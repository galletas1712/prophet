from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import torch


class RequestStage(Enum):
    PREFILL = 0
    DECODE = 1
    DONE = 2


@dataclass
class Request:
    request_id: int
    stage: RequestStage

    prompt_str: str
    output_str: Optional[str]  # Populated when request stage set to DONE.

    prompt_tokens: Optional[List[int]]  # Populated at prefill.
    output_tokens: List[int]

    cache_k: torch.Tensor
    cache_v: torch.Tensor

    def __init__(self, request_id, prompt_str, kv_cache_shape):
        self.request_id = request_id
        self.stage = RequestStage.PREFILL

        self.prompt_str = prompt_str
        self.output_str = None

        self.prompt_tokens = None
        self.output_tokens = []

        self.cache_k = torch.zeros(
            kv_cache_shape, dtype=torch.bfloat16, device="cuda:0"
        )

        self.cache_v = torch.zeros(
            kv_cache_shape, dtype=torch.bfloat16, device="cuda:0"
        )
