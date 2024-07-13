import torch
import ray
import os

import torch.distributed

from entrypoints.api import RequestStage, WorkerType
from entrypoints.llm import LLM
from ray.util.queue import Queue, Empty

from models.llama3.model import ModelArgs


@ray.remote(num_cpus=4, num_gpus=1, runtime_env={"nsight": {"s": "none"}})  # Disable CPU profiling
class Decoder:
    def __init__(
        self,
        name,
        config,
        input_queue: Queue,
        output_queue: Queue,
        coordinator
    ):
        torch.set_default_device("cuda")
        self.rank = ray.get_gpu_ids()[0]
        self.name = name

        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.coordinator = coordinator

        self.model_args = ModelArgs()

        # NOTE: max_requests_in_scheduler is NOT the max pending queue size.
        # Pending queue size is just how big the prefill "write buffer" is.
        # It prevents prefills from going to fast and overrunning the scheduler.
        # On the other hand, the free slots here is how many requests we get to choose from in the scheduler.
        # If k = 1, then all schedulers converge to FCFS!
        # Total KV cache buffer needed is max_requests_in_scheduler * max_seq_len * dim * 2 * 4 bytes

        assert self.config.decode_scheduler.max_requests_in_scheduler >= self.config.decode_scheduler.batch_size
        self.num_scheduler_slots = self.config.decode_scheduler.max_requests_in_scheduler

    def setup(self, rank, world_size):
        print(f"{self.name} initializing LLM...")
        self.llm = LLM(
            self.config.model,
            self.config.decode_scheduler,
            self.config.seed,
            worker_type=WorkerType.DECODE,
        )
        print(f"{self.name} done initializing LLM!")

        print(f"{self.name} initializing distributed environment...")
        # TODO: change this
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size) 
        print(f"{self.name} done initializing distributed environment!")
    
    def __repr__(self):
        return self.name
    
    # NOTE: Assumes prefill is rank 0 for now
    def receive_k(self, src_rank, prompt_len):
        tensor = self.kv_cache_buffer[0, :prompt_len, :, :]
        print(f"Receiving K cache of size {tensor.shape}")
        torch.cuda.nvtx.range_push("receive_k_cache")
        torch.cuda.synchronize()
        torch.distributed.recv(tensor=tensor, src=src_rank)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    def receive_v(self, src_rank, prompt_len):
        tensor = self.kv_cache_buffer[1, :prompt_len, :, :]
        print(f"Receiving V cache of size {tensor.shape}")
        torch.cuda.nvtx.range_push("receive_v_cache")
        torch.cuda.synchronize()
        torch.distributed.recv(tensor=tensor, src=src_rank)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    def run(self):
        print(f"Starting on GPU {self.rank}")

        for epoch in range(self.config.decoder_epochs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            torch.cuda.nvtx.range_push(f"Epoch {epoch}")

            num_free_slots = self.num_scheduler_slots - self.llm.num_requests_in_progress
            requests_to_add = []
            for _ in range(num_free_slots):
                try:
                    # Wait for a very small amount of time, otherwise move on
                    request = self.input_queue.get(
                        block=True,
                        timeout=self.config.coordinator.dequeue_timeout
                    )
                    requests_to_add.append(request)
                except Empty:
                    break
            
            # Pull KV caches from prefiller and add to scheduler
            torch.cuda.nvtx.range_push("receive_kv_cache")
            for request in requests_to_add:
                if request.stage is not RequestStage.DECODE:
                    raise ValueError(f"Decoder should only receive decode requests! Received: {request.stage}. Try setting max_prompt_len lower.")

                # TODO: modify model config to all be in the same file
                self.kv_cache_buffer = torch.zeros(
                    (2, self.config.model.max_seq_len, self.model_args.n_layers, self.model_args.dim),
                    dtype=torch.bfloat16,
                    device="cuda"
                )

                torch.cuda.nvtx.range_push("coordinator_call")
                # TODO: support more than 1 prefiller
                ray.get(self.coordinator.send_tensor.remote("prefiller#0", f"{self.name}", request.request_id, len(request.prompt_tokens)))
                torch.cuda.nvtx.range_pop()

                # Use unbind to set cache_k, cache_v to the same data_ptr as self.kv_cache_buffer
                request.cache_k, request.cache_v = torch.unbind(self.kv_cache_buffer, dim=0)
                request.idx_in_data_batch = None

                # No need to GC kv_cache_buffer here, since it is GC'ed in Request.free_cache in llm.py
                del self.kv_cache_buffer

                print(f"Decoder received request {request.request_id} pending scheduling...")
                self.llm.add_request(request)
            
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("llm_step_decode")
            done_requests, request_batch = self.llm.step_decode()
            torch.cuda.nvtx.range_pop()

            for request in request_batch:
                request.epochs.append(epoch)

            for request in done_requests:
                # NOTE: Important to block until queue is free
                print(f"Decoder finished request {request.request_id}")
                self.output_queue.put(request)

            torch.cuda.nvtx.range_pop()

            end_event.record()
            epoch_time = start_event.elapsed_time(end_event)
            print(f"Epoch {epoch} took {epoch_time} ms")