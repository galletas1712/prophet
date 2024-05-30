import asyncio
import json
import ray

import numpy as np

from entrypoints.api import CompletionType, Request
from models.llama3.tokenizer import Tokenizer, LlamaFormatter
from ray.util.queue import Queue

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