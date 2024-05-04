from typing import Optional, List, Tuple
from dataclasses import dataclass

import torch


@dataclass
class ModelOutputs:
    new_tokens: List[int]
    new_tokens_decoded: List[str]
    sequences_complete: List[bool]
    new_kv_caches: List[torch.tensor]


MODELS = {}


def register_model(name):
    def register_curr_model(model_fn):
        MODELS[name] = model_fn
        return model_fn

    return register_curr_model


@register_model("dummy")
def build_dummy(model_config):
    from models.dummy_model import DummyModel

    return DummyModel(**model_config)


@register_model("llama_3")
def build_llama_3(model_config):
    from models.llama3 import Llama

    return Llama.build(
        ckpt_dir=model_config.ckpt_dir,
        tokenizer_path=model_config.tokenizer_path,
        max_seq_len=model_config.max_seq_len,
        max_batch_size=model_config.max_batch_size,
    )


def build_model(model_config):
    return MODELS[model_config.name](model_config)
