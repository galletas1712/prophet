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

from entrypoints.api import Request, RequestStage
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


@dataclass
class BatchStage(Enum):
    PREFILL = 0
    DECODE = 1


@dataclass
class DataBatch:
    # Whether batch is prefill or infill.
    stage: BatchStage

    # Token ids of shape (batch_size, padded_seq_len). Only includes the new
    # tokens the model needs to process; does not include tokens for which KV
    # cache state is already known.
    input_tokens: torch.Tensor

    # Index of the first padding token in the input tokens for each sample in
    # the batch. If no padding, it is the index right after the last index of
    # the sample. Has shape (batch_size,).
    first_pad_idx: torch.Tensor

    # TODO: Log probs of the input_tokens and the output token. Has shape
    # (batch_size, padded_seq_len + 1).
    token_logprobs: Optional[torch.Tensor]

    # KV cache of shape (batch_size, max_seq_len, n_layers, model_dim).
    cache_k: torch.Tensor
    cache_v: torch.Tensor

    # Position in the corresponding sequences of each entry in input_tokens. Has
    # shape (batch_size,).
    start_pos: torch.Tensor

    # Shape (batch_size,). True if EOS has been generated.
    eos_reached: torch.Tensor

    def __init__(
        self, requests: List[Request], stage: BatchStage, pad_token: int
    ):
        self.stage = stage

        # Batch the KV caches. Caches in request are already padded with shape
        # (num_layers, max_seq_len, model_dim).
        self.cache_k = torch.stack([request.cache_k for request in requests])
        self.cache_v = torch.stack([request.cache_v for request in requests])

        self.first_pad_idx = torch.zeros(
            (len(requests),), dtype=torch.long, device="cuda"
        )

        # Build the input tokens tensor, consisting of tokens that haven't been
        # processed yet. If prefill, this is all input tokens. If decode, this
        # is the last outputted decode token.
        input_tokens = []
        max_input_tokens_len = 0

        for request in requests:
            if self.stage is BatchStage.PREFILL:
                input_tokens.append(request.prompt_tokens)
            elif self.stage is BatchStage.DECODE:
                input_tokens.append([request.output_tokens[-1]])

            max_input_tokens_len = max(
                max_input_tokens_len, len(input_tokens[-1])
            )

        batch_size = len(requests)

        self.input_tokens = torch.full(
            (batch_size, max_input_tokens_len),
            pad_token,
            dtype=torch.long,
            device="cuda",
        )

        for input_idx, token_seq in enumerate(input_tokens):
            self.input_tokens[input_idx, : len(token_seq)] = torch.tensor(
                token_seq, dtype=torch.long, device="cuda"
            )
            self.first_pad_idx[input_idx] = len(token_seq)

        # Set the start pos.
        batch_size = len(requests)
        if self.stage is BatchStage.PREFILL:
            self.start_pos = torch.zeros(
                (batch_size,), dtype=torch.long, device="cuda"
            )
        elif self.stage is BatchStage.DECODE:
            self.start_pos = torch.tensor(
                [
                    len(request.prompt_tokens) + len(request.output_tokens) - 1
                    for request in requests
                ],
                dtype=torch.long,
                device="cuda",
            )

        # EOS reached is default false when constructing the batch since a
        # request is assumed to be never scheduled if EOS already reached.
        self.eos_reached = torch.zeros((batch_size,), dtype=torch.bool)


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

        return Llama(model, tokenizer, model_params, glob_params)

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        model_params: ModelParams,
        glob_params: GlobalGenerationParams,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.model_params = model_params

        self.glob_params = glob_params
        if self.glob_params.max_gen_len is None:  # TODO: redo this!
            self.glob_params.max_gen_len = self.model_params.max_seq_len - 1

        self.stop_tokens = torch.tensor(list(tokenizer.stop_tokens))
        self.formatter = ChatFormat(tokenizer)

    def kv_cache_shape(self):
        return self.model.kv_cache_shape

    @torch.inference_mode()
    def step(self, requests: List[Request]):
        prefill_requests = []
        decode_requests = []

        for request in requests:
            if request.stage is RequestStage.PREFILL:
                prefill_requests.append(request)
            elif request.stage is RequestStage.DECODE:
                decode_requests.append(request)

        if len(prefill_requests) > 0:
            self.step_prefill(prefill_requests)

        if len(decode_requests) > 0:
            self.step_decode(decode_requests)

    @torch.inference_mode()
    def step_prefill(self, requests: List[Request]):
        # Tokenize inputs.
        for request in requests:
            request.prompt_tokens = self.tokenizer.encode(
                request.prompt_str, bos=True, eos=False
            )

        # Form the batch.
        prefill_batch = DataBatch(
            requests, BatchStage.PREFILL, self.tokenizer.pad_id
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
            requests,
            logits,
            prefill_batch.first_pad_idx,
            prefill_batch.start_pos,
        )
        self.update_kv_cache(requests, prefill_batch)

    @torch.inference_mode()
    def step_decode(self, requests: List[Request]):
        # Form the batch.
        decode_batch = DataBatch(
            requests, BatchStage.DECODE, self.tokenizer.pad_id
        )

        # Run through model, populating KV caches.
        logits = self.model.forward(
            decode_batch.input_tokens,
            decode_batch.start_pos,
            decode_batch.first_pad_idx,
            decode_batch.cache_k,
            decode_batch.cache_v,
        )

        self.sample_and_add_token(
            requests, logits, decode_batch.first_pad_idx, decode_batch.start_pos
        )
        self.update_kv_cache(requests, decode_batch)

    @torch.inference_mode()
    def sample_and_add_token(
        self,
        requests: List[Request],
        logits: torch.Tensor,
        first_pad_idx: torch.Tensor,
        start_pos: torch.Tensor,
    ):
        # Sample the next token. TODO: check how this would work for scatter.
        # NOTE: logits = (bsz, max_input_tokens_len, encoding_universe_size)
        assert torch.all(first_pad_idx > 0).item()

        logits = logits[torch.arange(len(requests)), first_pad_idx - 1, :]

        if self.glob_params.temperature > 0:
            probs = torch.softmax(
                logits / self.glob_params.temperature, dim=-1)
            next_token = sample_top_p(probs, self.glob_params.top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = next_token.reshape(-1).cpu().numpy()

        # Mutate requests with new tokens.
        for request_idx, request in enumerate(requests):
            curr_next_token = next_token[request_idx]
            request.output_tokens.append(curr_next_token)

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

    def update_kv_cache(self, requests: List[Request], batch: DataBatch):
        for request_idx, request in enumerate(requests):
            request.cache_k = batch.cache_k[request_idx, ...]
            request.cache_v = batch.cache_v[request_idx, ...]


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
