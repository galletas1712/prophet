from dataclasses import dataclass
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

    idx_in_data_batch: Optional[int]

    stage: RequestStage
    output_tokens: List[int]

    # Populated when request stage set to DONE.
    output: Optional[str | Any]

    # Populated on prefill
    cache_k: Optional[torch.Tensor]
    cache_v: Optional[torch.Tensor]

    # For benchmarking
    benchmark_metrics: RequestBenchmarkMetrics

    def __init__(
        self,
        prompt: str | Any,
        prompt_tokens: List[int],
        max_gen_len: int,
        request_id: str,
    ):
        self.prompt = prompt
        self.prompt_tokens = prompt_tokens
        self.max_gen_len = max_gen_len
        self.request_id = request_id

        self.idx_in_data_batch = None

        self.stage = RequestStage.PREFILL
        self.output_tokens = []

        self.output = None

        self.cache_k = None
        self.cache_v = None

        self.benchmark_metrics = RequestBenchmarkMetrics(request_id)


    def free_cache(self):
        del self.cache_k
        del self.cache_v

        # NOTE: Just for profiling purposes. Remove in production
        gc.collect()
        torch.cuda.empty_cache()
