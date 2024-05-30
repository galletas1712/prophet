import asyncio
import json
import ray

import numpy as np

from entrypoints.api import CompletionType, Request
from models.llama3.tokenizer import Tokenizer, LlamaFormatter
from ray.util.queue import Queue, Empty
from shareGPT_ray import ShareGPTRequestGenerator

shareGPTPath = '/Users/schwinn/shareGPT.json'
tokenizer_path = '/Users/schwinn/tokenizer.model'
@ray.remote(num_cpus=1)
class Prefiller:
    def __init__(self, input_queue: Queue, output_queue: Queue, rank: int, dequeue_timeout: float = 0.0005):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.rank = rank
        self.dequeue_timeout = dequeue_timeout
        self.max_batch_size = 8 # TODO: set batch size with config
    
    def __repr__(self):
        return f"Prefiller(rank:{self.rank})"
    
    def get_next_batch_ref(self):
        self.next_batch_ref = [self.input_queue.get_async(block=True, timeout=self.dequeue_timeout) for _ in range(self.max_batch_size)]
    
    async def run(self):
        print(f"Starting...")
        self.get_next_batch_ref()
        requests_to_process = []
        while True:
            dequeue_results = await asyncio.gather(*self.next_batch_ref, return_exceptions=True)
            requests_to_process = list(filter(lambda x: not isinstance(x, Empty), dequeue_results))
            self.get_next_batch_ref()

            # Do work
            print(f"Prefiller processing batch of {len(requests_to_process)} requests")
            print(f"Prefiller requests: {[request.request_id for request in requests_to_process]}")
            await asyncio.sleep(2)
            # NOTE: important to block until queue is free
            for request in requests_to_process:
                await self.output_queue.put_async(request)


@ray.remote(num_cpus=1)
class Decoder:
    def __init__(self, input_queue: Queue, output_queue: Queue, rank: int, dequeue_timeout: float = 0.005):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.rank = rank
        self.dequeue_timeout = dequeue_timeout
        self.max_batch_size = 8 # TODO: set batch size with config
    
    def __repr__(self):
        return f"Decoder(rank:{self.rank})"
    
    def get_next_batch_ref(self):
        self.next_batch_ref = [self.input_queue.get_async(block=True, timeout=self.dequeue_timeout) for _ in range(self.max_batch_size)]
    
    async def run(self):
        print(f"Starting...")
        self.get_next_batch_ref()
        requests_to_process = []
        while True:
            dequeue_results = await asyncio.gather(*self.next_batch_ref, return_exceptions=True)
            requests_to_process = list(filter(lambda x: not isinstance(x, Empty), dequeue_results))
            self.get_next_batch_ref()

            # Do work
            print(f"Decoder processing batch of {len(requests_to_process)} requests")
            print(f"Decoder requests: {[request.request_id for request in requests_to_process]}")
            await asyncio.sleep(4)
            # NOTE: important to block until queue is free
            for request in requests_to_process:
                await self.output_queue.put_async(request)

    
if __name__ == '__main__':
    ray.init()
    num_prefill_instances = 2
    num_decode_instances = 1
    request_queue = Queue(maxsize=16)
    pending_queue = Queue(maxsize=4)
    result_queue = Queue()
    request_generator = ShareGPTRequestGenerator.remote(shareGPTPath, tokenizer_path, request_queue)
    prefillers = [Prefiller.remote(request_queue, pending_queue, i) for i in range(num_prefill_instances)]
    decoders = [Decoder.remote(pending_queue, result_queue, i) for i in range(num_decode_instances)]

    ray.get([request_generator.run.remote()] + [prefiller.run.remote() for prefiller in prefillers] + [decoder.run.remote() for decoder in decoders])