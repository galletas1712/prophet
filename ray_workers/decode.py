import gc
import torch
import ray
from entrypoints.api import RequestStage, WorkerType
from entrypoints.databatch import PrefillDataBatch
from entrypoints.llm import LLM
from ray.util.queue import Queue, Empty

import ray.util.collective as collective
from models.llama3.model import ModelArgs


@ray.remote(num_cpus=4, num_gpus=1)
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

    def setup(self):
        print(f"{self.name} initializing LLM...")
        self.llm = LLM(
            self.config.model,
            self.config.decode_scheduler,
            self.config.seed,
            worker_type=WorkerType.DECODE,
        )
        print(f"{self.name} done initializing LLM!")
    
    def __repr__(self):
        return self.name
    
    # NOTE: Assumes prefill is rank 0 for now
    def receive_kv(self, src_rank):
        collective.recv(self.kv_cache_buffer, src_rank)
        torch.cuda.synchronize()
        print("Received in function")

    def run(self):
        print(f"Starting on GPU {self.rank}")
        while True:
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
            for request in requests_to_add:
                if request.stage is not RequestStage.DECODE:
                    raise ValueError(f"Decoder should only receive decode requests! Received: {request.stage}. Try setting max_prompt_len lower.")

                self.kv_cache_buffer = torch.zeros(
                    (2, len(request.prompt_tokens), self.model_args.n_layers, self.model_args.dim),
                    dtype=torch.bfloat16
                ).cuda()

                # TODO: support more than 1 prefiller
                ray.get(self.coordinator.send_tensor.remote("prefiller#0", f"{self.name}", request.request_id))
                request.cache_k, request.cache_v = torch.unbind(self.kv_cache_buffer, dim=0)

                del self.kv_cache_buffer
                gc.collect()
                torch.cuda.empty_cache()

                print(f"Decoder received request {request.request_id} pending scheduling...")
                self.llm.add_request(request)

                # First token from prefill
                request.benchmark_metrics.received_token()

            # Do work
            done_requests, request_batch = self.llm.step_decode()

            # Update benchmarks
            torch.cuda.synchronize()
            for request in request_batch:
                request.benchmark_metrics.received_token()

            for request in done_requests:
                # NOTE: Important to block until queue is free
                print(f"Decoder finished request {request.request_id}")
                self.output_queue.put(request)