from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class RequestStage(Enum):
    PREFILL = 0
    DECODE = 1
    DONE = 2


@dataclass
class Request:
    prompt_tokens: List[int]
    curr_idx_in_batch: Optional[int] = None
    stage: RequestStage = RequestStage.PREFILL
