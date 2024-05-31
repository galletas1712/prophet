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

    def get_next_batch_ref(self):
        self.next_batch_ref = [
            self.input_queue.get_async(
                block=True,
                timeout=self.config.coordinator.dequeue_timeout
            )
            for _ in range(self.config.prefill_scheduler.batch_size)
        ]

    async def run(self):
        print(f"Starting on GPU {ray.get_gpu_ids()}")
        self.get_next_batch_ref()
        requests_to_process = []
        while True:
            dequeue_results = await asyncio.gather(*self.next_batch_ref, return_exceptions=True)
            requests_to_process = list(
                filter(lambda x: not isinstance(x, Empty), dequeue_results))
            self.get_next_batch_ref()

            # Do work
            for request in requests_to_process:
                self.llm.add_request(request)

            if len(requests_to_process) > 0:
                print(
                    f"Prefiller processing batch of {len(requests_to_process)} requests")
                print(
                    f"Prefiller requests: {[request.request_id for request in requests_to_process]}")

            prefill_data_batch = self.llm.step_prefill()

            if prefill_data_batch is None:
                continue

            # Add to pending queue
            for request in requests_to_process:
                # NOTE: Important to block until queue is free
                request.cache_k = request.cache_k.detach().cpu()
                request.cache_v = request.cache_v.detach().cpu()
                await self.output_queue.put_async(request)

            # Free memory from prefill data batch
            del prefill_data_batch.cache_k
            del prefill_data_batch.cache_v
            gc.collect()
            torch.cuda.empty_cache()

