from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Any

import torch
import uuid

from entrypoints.benchmark import RequestBenchmarkMetrics

class WorkerType (Enum):
    PREFILL = 0
    DECODE = 1

class CompletionType(Enum):
    CHAT_COMPLETION = 0
    TEXT_COMPLETION = 1


class RequestStage(Enum):
    PREFILL = 0
    DECODE = 1
    DONE = 2


@dataclass
class Request:
    prompt: str | Any
    completion_type: CompletionType
    max_gen_len: int

    request_id: uuid.UUID = field(default_factory=uuid.uuid4)
    idx_in_data_batch: Optional[int] = None

    stage: RequestStage = RequestStage.PREFILL

    prompt_tokens: Optional[List[int]] = None  # Populated at prefill.
    output_tokens: List[int] = field(default_factory=list)

    # Populated when request stage set to DONE.
    output: Optional[str | Any] = None

    # Populated on prefill
    cache_k: torch.Tensor | None = None
    cache_v: torch.Tensor | None = None

    # For benchmarking
    benchmark_metrics: RequestBenchmarkMetrics = field(default_factory=RequestBenchmarkMetrics)

