import asyncio
import json
import ray

import numpy as np

from entrypoints.api import CompletionType, Request
from models.llama3.tokenizer import Tokenizer, LlamaFormatter
from ray.util.queue import Queue


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


class ShareGPTCorpus:
    def __init__(
        self,
        corpus_path: str,
        tokenizer_path: str,
        max_prompt_tokens: int,
    ):
        # Read corpus
        f = open(corpus_path)
        self.raw_corpus = json.load(f)
        f.close()

        # Load tokenizer
        self.tokenizer = Tokenizer(tokenizer_path)
        self.formatter = LlamaFormatter(self.tokenizer)
        self.max_prompt_tokens = max_prompt_tokens

        self.dialogs = self._preprocess_shareGPT_dialogs()

    def _preprocess_shareGPT_dialog(self, dialog):
        dialog = list(map(shareGPT_to_llama_format, dialog))
        dialog_len = self.formatter.get_max_dialog_len(dialog, self.max_prompt_tokens)
        while dialog_len > 0 and dialog[dialog_len - 1]['role'] == 'assistant':
            dialog_len -= 1
        return dialog[:dialog_len]

    def _preprocess_shareGPT_dialogs(self):
        return filter(
            lambda x: len(x) > 0,
            map(
                lambda convo: self._preprocess_shareGPT_dialog(convo['conversations']),
                self.raw_corpus
            )
    )

    def sample(self):
        return next(self.dialogs)

@ray.remote(num_cpus=1)
class ShareGPTRequestGenerator:
    def __init__(
            self,
            corpus_path: str,
            tokenizer_path: str,
            request_queue: Queue,
            max_prompt_tokens: int = 300,
            num_secs: int = 20,
            arrivals_per_sec: int = 10,
            max_gen_len_interval: tuple[int, int] = (5, 450),
        ):
        self.corpus_path = corpus_path
        self.tokenizer_path = tokenizer_path
        self.request_queue = request_queue

        # Generation parameters
        self.max_prompt_tokens = max_prompt_tokens
        self.num_secs = num_secs
        self.arrivals_per_sec = arrivals_per_sec
        self.max_gen_len_interval = max_gen_len_interval
    
    def load_corpus(self):
        print("Loading shareGPT corpus...")
        self.corpus = ShareGPTCorpus(self.corpus_path, self.tokenizer_path, self.max_prompt_tokens)
        print("Done loading shareGPT corpus!")
    
    async def run(self):
        print("Begin request generation")
        for _ in range(self.num_secs):
            await asyncio.sleep(1)
            num_requests = np.random.poisson(self.arrivals_per_sec)
            for _ in range(num_requests):
                request = Request(
                    self.corpus.sample(),
                    CompletionType.CHAT_COMPLETION,
                    np.random.randint(*self.max_gen_len_interval)
                )

                # Put into random prefill queue. Block until queue is free
                await self.request_queue.put_async(request)

                print(f"Request {request.request_id} added to request queue")