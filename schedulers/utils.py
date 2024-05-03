from enum import Enum
from typing import Optional
import torch
import itertools


class RequestStage(Enum):
    PREFILL = 0
    DECODE = 1
    DONE = 2


SCHEDULERS = {}

def register_scheduler(name):
    def register_curr_scheduler(scheduler_class):
        SCHEDULERS[name] = scheduler_class
        return scheduler_class
def create_scheduler(scheduler_config):
    return None


class Request():
    def __init__(self, request_id: int, prompt: str, stage: RequestStage = RequestStage.PREFILL,
        kv_cache: Optional[torch.tensor] = None):
        self.request_id = request_id
        self.prompt = prompt
        self.output = []
        self.stage = stage
        self.kv_cache = kv_cache

    def move_kv_cache(self, target_rank):
        self.stage = RequestStage.DECODE   
        # TODO: move kv cache to deivce with target_rank
    