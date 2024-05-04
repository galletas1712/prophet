from typing import Optional, List, Tuple

import numpy as np
import torch

from models.utils import ModelOutputs


class CharacterTokenizer:
    def tokenize(self, x):
        return [ord(c) for c in x]


class DummyModel:
    def __init__(self, eos_prob, **kwargs) -> None:
        self.vocab = [chr(i) for i in range(ord("A"), ord("z") + 1)]
        self.vocab.append("[EOS]")

        self.tokenizer = CharacterTokenizer()
        self.eos_prob = eos_prob

    def step(
        self,
        prompts: List[str],
        kv_caches: List[Optional[torch.tensor]],
    ) -> ModelOutputs:
        prompt_tokens = [self.tokenizer.tokenize(prompt) for prompt in prompts]

        new_tokens = []
        new_tokens_decoded = []
        sequences_complete = []

        for _ in range(len(prompt_tokens)):
            if np.random.uniform() < self.eos_prob:
                new_tokens.append(0)
                new_tokens_decoded.append("[EOS]")
                sequences_complete.append(True)
            else:
                curr_new_token = np.random.randint(
                    low=ord("A"), high=ord("z") + 1
                )
                new_tokens.append(curr_new_token)
                new_tokens_decoded.append(chr(curr_new_token))
                sequences_complete.append(False)

        output = ModelOutputs(
            new_tokens, new_tokens_decoded, sequences_complete, kv_caches
        )

        return output
