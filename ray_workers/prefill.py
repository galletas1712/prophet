import torch
import gc
import asyncio
import ray
from entrypoints.api import WorkerType
from entrypoints.llm import LLM
from ray.util.queue import Queue, Empty


# class PrefillDataBatchManager:
#     def __init__(self):
#         self.prefill_data_batches = {}
#         self.request_id_to_batch_id = {}
#         self.requests_still_in_batch = {}

#     def add_prefill_data_batch(self, prefill_data_batch: PrefillDataBatch):
#         self.prefill_data_batches[prefill_data_batch.batch_id] = prefill_data_batch
#         self.request_id_to_batch_id.update({r.request_id: prefill_data_batch.batch_id for r in prefill_data_batch.requests})
#         self.num_requests_still_in_batch = len(prefill_data_batch.requests)

#     def pop_request_kv(self, request: Request):
#         batch_id = self.request_id_to_batch_id.pop(request.request_id)

#         cache_k = self.prefill_data_batches[batch_id].cache_k[request.idx_in_data_batch, :len(request.prompt_tokens)]
#         cache_v = self.prefill_data_batches[batch_id].cache_v[request.idx_in_data_batch, :len(request.prompt_tokens)]

#         self.num_requests_still_in_batch[batch_id] -= 1
#         if self.num_requests_still_in_batch[batch_id] == 0:
#             prefill_data_batch = self.prefill_data_batches.pop(batch_id)
#             del prefill_data_batch.cache_k
#             del prefill_data_batch.cache_v
#             del prefill_data_batch
#             # NOTE: doesn't free GPU memory yet because we still have a reference to the tensor in cache_k and cache_v
#             self.requests_still_in_batch.pop(batch_id)

#         return cache_k, cache_v


@ray.remote(num_cpus=4, num_gpus=1)
class Prefiller:
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

        self.num_scheduler_slots = int(self.config.max_in_progress_factor *
                                       self.config.prefill_scheduler.batch_size)

    def load_llm(self):
        print(f"Prefiller(rank={self.rank}) initializing LLM...")
        self.llm = LLM(
            self.config.model,
            self.config.prefill_scheduler,
            self.config.seed,
            worker_type=WorkerType.PREFILL,
        )
        print(f"Prefiller(rank={self.rank}) done initializing LLM!")

    def __repr__(self):
        return f"Prefiller(rank={self.rank})"

    async def dequeue_request(self):
        # Wait for a very small amount of time, otherwise move on
        request = await self.input_queue.get_async(
            block=True,
            timeout=self.config.coordinator.dequeue_timeout
        )
        return request

    async def run(self):
        print(f"Starting on GPU {ray.get_gpu_ids()}")
        while True:
            num_free_slots = self.num_scheduler_slots - self.llm.num_requests_in_progress
            dequeue_cors = [self.dequeue_request()
                            for _ in range(num_free_slots)]
            dequeue_cors_results = await asyncio.gather(*dequeue_cors, return_exceptions=True)
            requests_to_add = filter(lambda x: not isinstance(
                x, Empty), dequeue_cors_results)
            for request in requests_to_add:
                print(f"Received request {request.request_id} pending scheduling...")
                self.llm.add_request(request)

            prefill_data_batch = self.llm.step_prefill()
            if prefill_data_batch is None:
                continue

            for request in prefill_data_batch.requests:
                print(f"Prefilled {request.request_id}")

            # Add to pending queue
            for request in prefill_data_batch.requests:
                # NOTE: Important to block until queue is free
                request.cache_k = request.cache_k.detach().cpu()
                request.cache_v = request.cache_v.detach().cpu()
                await self.output_queue.put_async(request)

            # Free memory from prefill data batch
            del prefill_data_batch.cache_k
            del prefill_data_batch.cache_v
            gc.collect()
            torch.cuda.empty_cache()
