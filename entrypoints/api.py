from enum import Enum
from typing import Optional, List

import torch


class RequestStage(Enum):
    PREFILL = 0
    DECODE = 1
    DONE = 2


class Request:
    def __init__(
        self,
        request_id: int,
        prompt: str,
        prompt_tokens: Optional[List[int]] = None,
        stage: RequestStage = RequestStage.PREFILL,
        kv_cache: Optional[torch.tensor] = None,
    ):
        self.request_id = request_id
        
        self.prompt = prompt
        self.prompt_tokens = prompt_tokens

        self.output = None
        self.output_tokens = []
        
        self.stage = stage
        self.kv_cache = kv_cache

    def move_kv_cache(self, target_rank):
        self.stage = RequestStage.DECODE
        # TODO: move kv cache to deivce with target_rank
