import torch
import gc
import ray
import threading
from entrypoints.api import WorkerType
from entrypoints.databatch import PrefillDataBatch
from entrypoints.llm import LLM
from ray.util.queue import Queue, Empty

import ray.util.collective as collective


class SizeLimitedThreadSafeDict:
    def __init__(self, max_size):
        self.dict = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def add_item(self, key, value):
        with self.condition:
            self.dict[key] = value
            self.condition.notify_all()

    def pop_item(self, key):
        with self.condition:
            value = self.dict.pop(key)
            self.condition.notify_all()
            return value

    def wait_until_below_size(self, target_size):
        with self.condition:
            while len(self.dict) >= target_size:
                self.condition.wait() 


class KVCacheManager:
    def __init__(self, batch_size: int):
        # NOTE: There cannot be more than batch_size requests pending transfer
        # This is SEPARATE from the number of requests in the scheduler
        self.batch_size = batch_size
        self.kv_cache = SizeLimitedThreadSafeDict(batch_size)
    
    def new_prefill_batch(self, prefill_data_batch: PrefillDataBatch):
        for request in prefill_data_batch.requests:
            self.kv_cache.add_item(
                request.request_id,
                torch.stack([
                    request.cache_k,
                    request.cache_v
                ])
            )
            del request.cache_k
            del request.cache_v
        
    def pop_request(self, request_id: str):
        return self.kv_cache.pop_item(request_id)


@ray.remote(num_cpus=4, num_gpus=1)
class Prefiller:
    def __init__(
        self,
        name,
        config,
        input_queue: Queue,
        output_queue: Queue,
    ):
        torch.set_default_device("cuda")
        self.rank = ray.get_gpu_ids()[0]

        self.name = name
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        assert config.prefill_scheduler.max_requests_in_scheduler >= config.prefill_scheduler.batch_size
        self.num_scheduler_slots = config.prefill_scheduler.max_requests_in_scheduler

        self.kv_cache_manager = KVCacheManager(config.prefill_scheduler.batch_size)
    
    def setup(self):
        print(f"{self.name} initializing LLM...")
        self.llm = LLM(
            self.config.model,
            self.config.prefill_scheduler,
            self.config.seed,
            worker_type=WorkerType.PREFILL,
        )
        print(f"{self.name} done initializing LLM!")

    def __repr__(self):
        return self.name

    def send_kv(self, request_id: str, target_rank: int):
        kv_cache = self.kv_cache_manager.pop_request(request_id)
        assert(kv_cache.dtype == torch.bfloat16)
        collective.send(kv_cache, target_rank)
        torch.cuda.synchronize()

        # Free memory
        del kv_cache
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Sent {request_id} to decoder")

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

            for request in requests_to_add:
                print(f"Received request {request.request_id} pending scheduling...")
                self.llm.add_request(request)

            prefill_data_batch = self.llm.step_prefill()
            if prefill_data_batch is None:
                continue
            
            # Add reference of each request's prefill KV cache to data batch
            self.kv_cache_manager.new_prefill_batch(prefill_data_batch)
            
            del prefill_data_batch.cache_k
            del prefill_data_batch.cache_v

            gc.collect()
            torch.cuda.empty_cache()

            # Log successful prefill + immediately remove from prefill scheduler
            # as no more prefills for these requests will be done.
            for request in prefill_data_batch.requests:
                self.llm.get_scheduler().remove_request(request)
                print(f"Prefilled {request.request_id}")

            # NOTE: Important to block until pending queue is free
            for request in prefill_data_batch.requests:
                self.output_queue.put(request, block=True)
