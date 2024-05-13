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
    temperature: float = 0.6,
    top_p: float = 0.9,
    logprobs: bool = False,
    echo: bool = False,


@dataclass
class ModelParams:
    max_batch_size: int
    max_seq_len: int
    n_layers: int
    n_local_kv_heads: int
    head_dim: int


@dataclass
class LlamaPrefillBatchState:
    next_pos: torch.Tensor  # (bsz) next position to fill in the prompt
    min_prompt_len: int
    max_prompt_len: int

    tokens: torch.Tensor  # (bsz, max_seq_len) token IDs
    # (bsz, max_seq_len) log probabilities
    token_logprobs: Optional[torch.Tensor]
    # (bsz, max_seq_len) true is token is not padding.
    input_mask: torch.Tensor
    eos_reached: torch.Tensor  # (bsz) true if EOS token has been generated
    # Global k cache. We will always read/write to this tensor and pass entries over RPC
    cache_k: torch.Tensor
    cache_v: torch.Tensor  # Same for v cache

    def __init__(
        self,
        prompt_tokens: List[List[int]],
        logprobs: bool,
        params: ModelParams,
        tokenizer: Tokenizer,
    ):
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # NOTE: TODO: need to actually USE this. We're just gonna preallocate a max_prompt_len in decode for now
        self.next_pos = torch.tensor([len(t) for t in prompt_tokens])

        self.min_prompt_len = torch.min(self.next_pos).item()
        self.max_prompt_len = torch.max(self.next_pos).item()
        assert self.max_prompt_len <= params.max_seq_len

        # NOTE: Here we only allocate for this batch size and for max_prompt_len + 1 (to add next token after prefill) within this batch
        self.tokens = torch.full(
            (bsz, self.max_prompt_len + 1), tokenizer.pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            self.tokens[k, : len(t)] = torch.tensor(
                t, dtype=torch.long, device="cuda")
        if logprobs:
            self.token_logprobs = torch.zeros_like(
                self.tokens, dtype=torch.float)
        self.input_mask = self.tokens != tokenizer.pad_id
        self.eos_reached = torch.tensor([False] * bsz, device="cuda")
        kv_dim = (
            params.max_batch_size,
            params.max_seq_len,
            params.n_layers,
            params.n_local_kv_heads,
            params.head_dim,
        )
        self.cache_k = torch.zeros(kv_dim).cuda()
        self.cache_v = torch.zeros(kv_dim).cuda()


@dataclass
class LlamaDecodeBatchState:
    logprobs: bool
    params: ModelParams
    tokenizer: Tokenizer

    # Fixed size list of requests in the batch. Some entries may be inactive
    requests: List[Optional[Request]]
    tokens: torch.Tensor  # (bsz, max_seq_len) token IDs
    # (bsz, max_seq_len) log probabilities
    token_logprobs: Optional[torch.Tensor]
    # (bsz, max_seq_len) true is token is not padding.
    input_mask: torch.Tensor
    eos_reached: torch.Tensor  # (bsz) true if EOS token has been generated
    # Global k cache. We will always read/write to this tensor and pass entries over RPC
    cache_k: torch.Tensor
    cache_v: torch.Tensor  # Same for v cache

    def __init__(self, logprobs: bool, params: ModelParams, tokenizer: Tokenizer):
        self.logprobs = logprobs
        self.params = params
        self.tokenizer = tokenizer

        self.requests = [None for _ in range(params.max_batch_size)]
        # NOTE: Here we allocate for MAX batch size and MAX seq len
        self.tokens = torch.full((params.max_batch_size, params.max_seq_len),
                                 tokenizer.pad_id, dtype=torch.long, device="cuda")
        if logprobs:
            self.token_logprobs = torch.zeros_like(
                self.tokens, dtype=torch.float)
        self.input_mask = torch.zeros_like(
            self.tokens, dtype=torch.bool, device="cuda")
        self.eos_reached = torch.tensor(
            [True] * params.max_batch_size, device="cuda")  # NOTE: important
        kv_dim = (
            params.max_batch_size,
            params.max_seq_len,
            params.n_layers,
            params.n_local_kv_heads,
            params.head_dim,
        )
        self.cache_k = torch.zeros(kv_dim).cuda()
        self.cache_v = torch.zeros(kv_dim).cuda()

    def fill_slot(
        self,
        idx: int,
        request: Request,
        input_len_after_prefill: int,
        prompt_tokens: List[int],
        tokens: torch.Tensor,
        input_mask: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):  # TODO: recheck sending KV cache over
        assert len(prompt_tokens) <= self.tokens.shape[1]
        request.stage = RequestStage.DECODE
        request.curr_idx_in_batch = idx
        self.requests[idx] = request
        # TODO: over RPC
        self.tokens[idx, :input_len_after_prefill] = tokens[:input_len_after_prefill].clone()
        # TODO: omit entirely to optimize
        self.input_mask[idx,
                        :input_len_after_prefill] = input_mask[:input_len_after_prefill].clone()
        # TODO: over RPC
        self.cache_k[idx, :input_len_after_prefill] = cache_k[:input_len_after_prefill].clone()
        # TODO: over RPC
        self.cache_v[idx, :input_len_after_prefill] = cache_v[:input_len_after_prefill].clone()
        self.eos_reached[idx] = False  # TODO: assert EOS is not reached yet
        return request

    def clear_slot(self, idx: int):
        self.requests[idx].curr_idx_in_batch = None
        self.requests[idx].stage = RequestStage.DONE
        self.requests[idx] = None
        self.tokens[idx, :] = self.tokenizer.pad_id
        self.input_mask[idx, :] = False
        self.eos_reached[idx] = True
        self.cache_k[idx, :] = 0
        self.cache_v[idx, :] = 0


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        glob_params: GlobalGenerationParams,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
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

        # seed must be the same in all processes
        torch.manual_seed(seed)

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

        n_local_kv_heads = model_args.n_kv_heads // fs_init.get_model_parallel_world_size()
        head_dim = model_args.dim // model_args.n_heads

        model_params = ModelParams(
            max_batch_size,
            max_seq_len,
            model_args.n_layers,
            n_local_kv_heads,
            head_dim,
        )

        return Llama(model, tokenizer, model_params, glob_params)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, model_params: ModelParams, glob_params: GlobalGenerationParams):
        self.model = model
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.glob_params = glob_params
        if self.glob_params.max_gen_len is None:  # TODO: redo this!
            self.glob_params.max_gen_len = self.model_params.max_seq_len - 1
        self.stop_tokens = torch.tensor(list(tokenizer.stop_tokens))
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def step_prefill(self, prompt_tokens: List[List[int]]):
        batch_state = LlamaPrefillBatchState(
            prompt_tokens, self.glob_params.logprobs, self.model_params, self.tokenizer)
        # NOTE: Prefill only up to min_prompt_len. In decode, we're going to wait until it catches up to the others
        # TODO: just fill to max_prompt_len and pad KV cache
        logits = self.model.forward(
            batch_state.tokens[:, :batch_state.min_prompt_len], 0, batch_state.cache_k, batch_state.cache_v)
        if self.glob_params.temperature > 0:  # TODO: check how this would work for scatter
            probs = torch.softmax(
                logits[:, -1] / self.glob_params.temperature, dim=-1)
            next_token = sample_top_p(probs, self.glob_params.top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        next_token = torch.where(
            batch_state.input_mask[:, batch_state.min_prompt_len], batch_state.tokens[:,
                                                                                      batch_state.min_prompt_len], next_token
        )
        if self.glob_params.max_gen_len > 0 or batch_state.min_prompt_len < self.model_params.max_seq_len:
            # TODO: replace max_prompt_len for scatter
            batch_state.tokens[:, batch_state.min_prompt_len] = next_token
            batch_state.eos_reached |= (~batch_state.input_mask[:, batch_state.min_prompt_len]) & (
                torch.isin(next_token, self.stop_tokens)
            )
        else:
            batch_state.eos_reached[:] = True
        # TODO: logprobs
        return batch_state

    @torch.inference_mode()
    def step_decode(self, batch_state: LlamaDecodeBatchState, prev_pos: int):
        logits = self.model.forward(
            # TODO: check OOB
            batch_state.tokens[:, prev_pos:prev_pos + 1], prev_pos, batch_state.cache_k, batch_state.cache_v)
        if self.glob_params.temperature > 0:  # TODO: check how this would work for scatter
            probs = torch.softmax(
                logits[:, -1] / self.glob_params.temperature, dim=-1)
            next_token = sample_top_p(probs, self.glob_params.top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            batch_state.input_mask[:, prev_pos +
                                   1], batch_state.tokens[:, prev_pos+1], next_token
        )
        # TODO: redo boundary conditions
        # TODO: replace max_prompt_len for scatter
        batch_state.tokens[:, prev_pos + 1] = next_token
        batch_state.eos_reached |= (~batch_state.input_mask[:, prev_pos + 1]) & (
            torch.isin(next_token, self.stop_tokens)
        )
        # TODO: logprobs EVERYWHERE!

    @torch.inference_mode()
    def postprocess(self, batch: LlamaDecodeBatchState):
        if self.glob_params.logprobs:
            token_logprobs = batch.token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(batch.tokens.tolist()):
            if batch.requests[i] is None:
                out_tokens.append([])
                out_logprobs.append([])
                continue
            # Truncate to max gen len
            start = 0 if self.glob_params.echo else len(
                batch.requests[i].prompt_tokens)
            toks = toks[start: len(
                batch.requests[i].prompt_tokens) + self.glob_params.max_gen_len]
            probs = None
            if self.glob_params.logprobs:
                probs = token_logprobs[i][start: len(
                    batch.requests[i].prompt_tokens) + self.glob_params.max_gen_len]
            # Truncate after eos if any
            for stop_token in self.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if self.glob_params.logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if self.glob_params.logprobs else None)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        requests = [Request(prompt_tokens[i])
                    for i in range(len(prompt_tokens))]
        prefill_batch_state = self.step_prefill(prompt_tokens)
        bsz = len(prompt_tokens)

        decode_batch_state = LlamaDecodeBatchState(
            self.glob_params.logprobs, self.model_params, self.tokenizer)
        for b in range(bsz):
            if not prefill_batch_state.eos_reached[b]:
                decode_batch_state.fill_slot(
                    b,
                    requests[b],
                    # NOTE: In original Llama code, we're going to wait until it the slowest one catches up in the decode pool
                    max(prefill_batch_state.min_prompt_len + \
                        1, len(prompt_tokens[b])),
                    prompt_tokens[b],
                    prefill_batch_state.tokens[b, :],
                    prefill_batch_state.input_mask[b, :],
                    prefill_batch_state.cache_k[b, :],
                    prefill_batch_state.cache_v[b, :]
                )

        prev_pos = prefill_batch_state.min_prompt_len
        # NOTE: EOS and boundary conditions might have to be completely redone here and above
        while not torch.all(decode_batch_state.eos_reached) and prev_pos < self.model.params.max_seq_len - 1:
            self.step_decode(decode_batch_state, prev_pos)
            # NOTE: Can't do this yet because it would just clear and we won't get any results.
            # for b in range(bsz):
            #     if decode_batch_state.requests[b] is not None and decode_batch_state.eos_reached[b]: # TODO: what about when max len reached?
            # decode_batch_state.clear_slot(b)
            prev_pos += 1

        return self.postprocess(decode_batch_state)

    def text_completion(
        self,
        prompts: List[str],
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
        )
        if self.glob_params.logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        prompt_tokens = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
        )
        if self.glob_params.logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t),
                },
            }
            for t in generation_tokens
        ]


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
