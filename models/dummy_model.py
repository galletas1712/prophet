from typing import Optional, List, Tuple

import numpy as np

from entrypoints.api import Request, RequestStage


class CharacterTokenizer:
    def encode(self, prompt_str):
        return [ord(c) for c in prompt_str]

    def decode(self, tokens):
        return "".join(
            [chr(token) if token != 0 else "[EOS]" for token in tokens]
        )


class DummyModel:
    def __init__(self, eos_prob, max_seq_len, **kwargs) -> None:
        self.tokenizer = CharacterTokenizer()  # EOS is 0.
        self.eos_prob = eos_prob

        # Inclusive of EOS token.
        self.max_seq_len = max_seq_len

    # Returns an arbitrary shape to test scheduler KV cache allocation.
    def kv_cache_shape(self):
        return (self.max_seq_len, 1, 64)

    def step(self, requests: List[Request]):
        for request in requests:
            # Tokenize prompt string if needed.
            if request.prompt_tokens is None:
                request.prompt_tokens = self.tokenizer.encode(
                    request.prompt_str
                )

            # Set request to DONE if EOS emitted or max seq len reached.
            curr_len = len(request.prompt_tokens) + len(request.output_tokens)

            if (
                np.random.uniform() < self.eos_prob
                or curr_len == self.max_seq_len - 1
            ):
                request.output_tokens.append(0)  # EOS token.
                request.output_str = self.tokenizer.decode(
                    request.output_tokens
                )
                request.stage = RequestStage.DONE

            # If not terminating sequence, sample a random letter to append.
            else:
                curr_new_token = np.random.randint(
                    low=ord("A"), high=ord("z") + 1
                )
                request.output_tokens.append(curr_new_token)
                request.stage = RequestStage.DECODE
