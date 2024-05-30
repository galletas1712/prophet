import asyncio
import json
import ray

import numpy as np

from entrypoints.api import CompletionType, Request
from models.llama3.tokenizer import Tokenizer, LlamaFormatter
from ray.util.queue import Queue, Empty

shareGPTPath = '/Users/schwinn/shareGPT.json'
tokenizer = Tokenizer('/Users/schwinn/tokenizer.model')
formatter = LlamaFormatter(tokenizer)


def read_shareGPTJSON():
    f = open(shareGPTPath)
    shareGPTJSON = json.load(f)
    f.close()
    return shareGPTJSON


def shareGPT_to_llama_format(entry):
    if entry['from'] == 'human':
        return {
            'role': 'user',
            'content': entry['value']
        }
    else:
        return {
            'role': 'assistant',
            'content': entry['value']
        }


def preprocess_shareGPT_dialog(dialog, max_tokens):
    dialog = list(map(shareGPT_to_llama_format, dialog))
    dialog_len = formatter.get_max_dialog_len(dialog, max_tokens)
    while dialog_len > 0 and dialog[dialog_len - 1]['role'] == 'assistant':
        dialog_len -= 1
    return dialog[:dialog_len]


def preprocess_shareGPT_dialogs(corpus, max_tokens):
    return filter(
        lambda x: len(x) > 0,
        map(
            lambda convo: preprocess_shareGPT_dialog(
                convo['conversations'], max_tokens),
            corpus
        )
    )


@ray.remote(num_cpus=1)
class RequestGenerator:
    def __init__(
            self,
            request_queue: Queue,
            num_secs: int = 20,
            arrivals_per_sec: int = 10,
            max_gen_len_interval: tuple[int, int] = (5, 450),
        ):
        self.request_queue = request_queue

        shareGPTJSON = read_shareGPTJSON()
        self.dialogs = preprocess_shareGPT_dialogs(shareGPTJSON, 300)

        self.num_secs = num_secs
        self.arrivals_per_sec = arrivals_per_sec
        self.max_gen_len_interval = max_gen_len_interval
    
    def __repr__(self):
        return f"shareGPT RequestGenerator"

    async def run(self):
        print("Begin request generation")
        for _ in range(self.num_secs):
            await asyncio.sleep(1)
            num_requests = np.random.poisson(self.arrivals_per_sec)
            for _ in range(num_requests):
                request = Request(
                    next(self.dialogs),
                    CompletionType.CHAT_COMPLETION,
                    np.random.randint(*self.max_gen_len_interval)
                )

                # Put into random prefill queue. Block until queue is free
                await self.request_queue.put_async(request)

                print(f"Request {request.request_id} added to request queue")
        

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
    request_generator = RequestGenerator.remote(request_queue)
    prefillers = [Prefiller.remote(request_queue, pending_queue, i) for i in range(num_prefill_instances)]
    decoders = [Decoder.remote(pending_queue, result_queue, i) for i in range(num_decode_instances)]

    ray.get([request_generator.run.remote()] + [prefiller.run.remote() for prefiller in prefillers] + [decoder.run.remote() for decoder in decoders])