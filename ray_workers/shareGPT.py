import asyncio
import json
import ray

import numpy as np

from entrypoints.api import Request
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
            config,
            tokenizer_path: str,
            request_queue: Queue,
        ):
        self.config = config
        self.prompt_suffix = config.prompt_suffix
        self.tokenizer_path = tokenizer_path
        self.request_queue = request_queue

    def load_corpus(self):
        print("Loading shareGPT corpus...")
        self.corpus = ShareGPTCorpus(self.config.corpus_path, self.tokenizer_path, self.config.max_prompt_tokens)
        print("Done loading shareGPT corpus!")
    
    def _append_prompt_suffix(self, prompt):
        prompt.append({
            "role": "user",
            "content": self.prompt_suffix
        })
        return prompt
    
    async def run(self):
        print("Begin request generation")
        while True:
            await asyncio.sleep(1)
            num_requests = np.random.poisson(self.config.arrivals_per_sec)
            for _ in range(num_requests):
                prompt = self.corpus.sample()
                
                if self.prompt_suffix is not None and len(self.prompt_suffix) > 0:
                    self._append_prompt_suffix(prompt)
                
                request = Request(
                    prompt,
                    self.corpus.formatter.encode_chat_completion(prompt),
                    np.random.randint(self.config.max_gen_len_low, self.config.max_gen_len_high)
                )

                # Put into request queue. Block until queue is free
                await self.request_queue.put_async(request)

                print(f"Request {request.request_id} added to request queue")