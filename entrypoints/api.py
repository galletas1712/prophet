from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Any

import gc
import torch

from entrypoints.benchmark import RequestBenchmarkMetrics

class WorkerType (Enum):
    PREFILL = 0
    DECODE = 1


class RequestStage(Enum):
    PREFILL = 0
    DECODE = 1
    DONE = 2


@dataclass
class Request:
    prompt: str | Any
    prompt_tokens: List[int]
    max_gen_len: int
    request_id: str

    oracle_request_len: Optional[int] = None

    idx_in_data_batch: Optional[int] = None

    stage: RequestStage = RequestStage.PREFILL
    output_tokens: List[int] = field(default_factory=list)

    # Populated when request stage set to DONE.
    output: Optional[str | Any] = None

    # Populated on prefill
    cache_k: Optional[torch.Tensor] = None
    cache_v: Optional[torch.Tensor] = None

    # For profiling
    estimated_token_length: Optional[int] = None

    # For benchmarking
    benchmark_metrics: RequestBenchmarkMetrics = field(default_factory=RequestBenchmarkMetrics)

    def free_cache(self):
        del self.cache_k
        del self.cache_v

        # NOTE: Just for profiling purposes. Remove in production
        gc.collect()
        torch.cuda.empty_cache()
