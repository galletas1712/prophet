# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

# import pdb
import fairscale.nn.model_parallel.initialize as fs_init

from dataclasses import dataclass
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from enum import Enum

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from entrypoints.api import Request, RequestStage, PrefillDataBatch, DecodeDataBatch
from models.llama3.model import ModelArgs, Transformer
from models.llama3.tokenizer import ChatFormat, Dialog, Message, Tokenizer


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


@dataclass
class GlobalGenerationParams:
    max_gen_len: Optional[int]
    temperature: float = (0.6,)
    top_p: float = (0.9,)
    logprobs: bool = (False,)
    echo: bool = (False,)


@dataclass
class ModelParams:
    max_batch_size: int
    max_seq_len: int
    n_layers: int
    n_local_kv_heads: int
    head_dim: int


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        glob_params: GlobalGenerationParams,
        model_parallel_size: Optional[int] = None,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"

        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"

        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
            model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                **params,
            )

        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words

        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        n_local_kv_heads = (
            model_args.n_kv_heads // fs_init.get_model_parallel_world_size()
        )
        head_dim = model_args.dim // model_args.n_heads

        model_params = ModelParams(
            max_batch_size,
            max_seq_len,
            model_args.n_layers,
            n_local_kv_heads,
            head_dim,
        )

        return Llama(model, tokenizer, model_params, model_args, glob_params)

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        model_params: ModelParams,  # TODO: remove model_params
        model_args: ModelArgs,
        glob_params: GlobalGenerationParams,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.model_params = model_params
        self.model_args = model_args

        self.glob_params = glob_params
        if self.glob_params.max_gen_len is None:  # TODO: redo this!
            self.glob_params.max_gen_len = self.model_params.max_seq_len - 1

        self.stop_tokens = torch.tensor(list(tokenizer.stop_tokens))
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def step_prefill(self, requests: List[Request]):
        # Tokenize inputs. # TODO: chat tokenizer as well
        for request in requests:
            request.prompt_tokens = self.tokenizer.encode(
                request.prompt_str, bos=True, eos=False
            )

        # Form the batch.
        prefill_batch = PrefillDataBatch(
            requests,
            self.model_args.max_seq_len,
            self.model_args.n_layers,
            self.model_args.dim,
            self.tokenizer.pad_id
        )

        # Run through model, populating KV caches.
        logits = self.model.forward(
            prefill_batch.input_tokens,
            prefill_batch.start_pos,
            prefill_batch.first_pad_idx,
            prefill_batch.cache_k,
            prefill_batch.cache_v,
        )

        self.sample_and_add_token(
            prefill_batch,
            logits,
        )

        return prefill_batch

    @torch.inference_mode()
    def step_decode(self, decode_batch: DecodeDataBatch):
        # Run through model, populating KV caches.
        logits = self.model.forward(
            decode_batch.input_tokens,
            decode_batch.start_pos,
            decode_batch.first_pad_idx,
            decode_batch.cache_k,
            decode_batch.cache_v,
        )

        self.sample_and_add_token(
            decode_batch,
            logits,
        )

    @torch.inference_mode()
    def sample_and_add_token(
        self,
        batch: PrefillDataBatch | DecodeDataBatch,
        logits: torch.Tensor,
    ):
        # Sample the next token.
        # NOTE: logits = (bsz, max_input_tokens_len, encoding_universe_size)
        assert torch.all(batch.first_pad_idx > 0).item()

        logits = logits[torch.arange(
            len(batch.requests)), torch.clamp(batch.first_pad_idx - 1, 0), :]

        if self.glob_params.temperature > 0:
            probs = torch.softmax(
                logits / self.glob_params.temperature, dim=-1)
            next_token = sample_top_p(probs, self.glob_params.top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = next_token.reshape(-1).cpu().numpy()

        # Mutate requests with new tokens.
        for request_idx, request in enumerate(batch.requests):
            if request.stage is RequestStage.DONE:  # NOTE: Important check!
                continue

            curr_next_token = next_token[request_idx]
            request.output_tokens.append(curr_next_token)

            if request.stage == RequestStage.DECODE:
                batch.input_tokens[request_idx] = curr_next_token
                batch.start_pos[request_idx] += 1

            # If generation sample EOS or hits max seq len, sets request stage
            # to DONE and sets request output_str.
            if (
                curr_next_token == self.tokenizer.eos_id
                or len(request.output_tokens) == self.glob_params.max_gen_len
            ):
                request.output_str = self.tokenizer.decode(
                    request.output_tokens
                )
                request.stage = RequestStage.DONE

            # Generation needs to continue so set stage to DECODE.
            else:
                request.stage = RequestStage.DECODE


@torch.inference_mode()
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
