import torch
import gc
import ray
import threading
from entrypoints.api import WorkerType
from entrypoints.databatch import PrefillDataBatch
from entrypoints.llm import LLM
from ray.util.queue import Queue, Empty

import ray.util.collective as collective
from models.llama3.model import ModelArgs


@ray.remote(num_cpus=4, num_gpus=1)
class ConsumerTest:
    def __init__(
        self,
        config,
        input_queue: Queue,
        output_queue: Queue,
        coordinator
    ):
        torch.set_default_device("cuda")
        self.rank = ray.get_gpu_ids()[0]

        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.coordinator = coordinator

        self.model_args = ModelArgs()

    def setup(self):
        pass
    
    def __repr__(self):
        return f"ConsumerTest(rank={self.rank})"
    
    # NOTE: Assumes prefill is rank 0 for now
    def receive_kv(self, src_rank):
        collective.recv(self.kv_cache_buffer, src_rank)
        torch.cuda.synchronize()
        print("Received in function")

    def run(self):
        print(f"Starting on GPU {self.rank}")
        while True:
            try:
                # Wait for a very small amount of time, otherwise move on
                request = self.input_queue.get(
                    block=True,
                    timeout=self.config.coordinator.dequeue_timeout
                )
            except Empty:
                continue
        
            self.kv_cache_buffer = torch.zeros((2, len(request.prompt_tokens), self.model_args.n_layers, self.model_args.dim), dtype=torch.bfloat16).cuda()
            ray.get(self.coordinator.send_tensor.remote("prefiller#0", "consumer_test", request.request_id))
            print("Received")
            print(self.kv_cache_buffer[0, 3, 10, 0])
            # request.cache_k, request.cache_v = torch.unbind(self.kv_cache_buffer, dim=0)
            # print(request.cache_k, request.cache_v)