import torch
import asyncio
import ray
from entrypoints.api import WorkerType
from entrypoints.llm import LLM
from ray.util.queue import Queue, Empty

@ray.remote(num_cpus=4, num_gpus=1)
class Decoder:
    def __init__(
        self,
        config,
        input_queue: Queue,
        output_queue: Queue,
    ):
        torch.set_default_device("cuda")
        self.rank = ray.get_gpu_ids()[0]

        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue

        # NOTE: max_requests_in_scheduler is NOT the max pending queue size.
        # Pending queue size is just how big the prefill "write buffer" is.
        # It prevents prefills from going to fast and overrunning the scheduler.
        # On the other hand, the free slots here is how many requests we get to choose from in the scheduler.
        # If k = 1, then all schedulers converge to FCFS!
        # Total KV cache buffer needed is max_requests_in_scheduler * max_seq_len * dim * 2 * 4 bytes

        assert self.config.decode_scheduler.max_requests_in_scheduler >= self.config.decode_scheduler.batch_size
        self.num_scheduler_slots = self.config.decode_scheduler.max_requests_in_scheduler

    def load_llm(self):
        print(f"Decoder(rank={self.rank}) initializing LLM...")
        self.llm = LLM(
            self.config.model,
            self.config.decode_scheduler,
            self.config.seed,
            worker_type=WorkerType.DECODE,
        )
        print(f"Decoder(rank={self.rank}) done initializing LLM!")

    def __repr__(self):
        return f"Decoder(rank={self.rank})"

    async def dequeue_request(self):
        # Wait for a very small amount of time, otherwise move on
        request = await self.input_queue.get_async(
            block=True,
            timeout=self.config.coordinator.dequeue_timeout
        )

        # Retrieve KV cache from main memory after deserialization
        request.cache_k = request.cache_k.cuda()
        request.cache_v = request.cache_v.cuda()

        request.max_gen_len = min(self.config.model.max_seq_len - len(request.prompt_tokens), request.max_gen_len)

        return request

    async def run(self):
        print(f"Starting on GPU {ray.get_gpu_ids()}")
        while True:
            # TODO: Pipeline! This blocks on I/O from prefill process right now

            num_free_slots = self.num_scheduler_slots - self.llm.num_requests_in_progress
            dequeue_cors = [self.dequeue_request()
                            for _ in range(num_free_slots)]
            dequeue_cors_results = await asyncio.gather(*dequeue_cors, return_exceptions=True)
            requests_to_add = filter(lambda x: not isinstance(
                x, Empty), dequeue_cors_results)
            for request in requests_to_add:
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
                await self.output_queue.put_async(request)

