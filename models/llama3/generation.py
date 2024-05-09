# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

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
class LlamaBatchParams:
    prompt_tokens: List[List[int]]
    max_gen_len: int
    temperature: float = 0.6,
    top_p: float = 0.9,
    logprobs: bool = False,
    echo: bool = False,

@dataclass
class LlamaBatch:
    min_prompt_len: int
    max_prompt_len: int
    total_len: int
    tokens: torch.Tensor
    token_logprobs: Optional[torch.Tensor]
    prev_pos: int
    eos_reached: torch.Tensor
    input_text_mask: torch.Tensor

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
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

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_tokens = torch.tensor(list(tokenizer.stop_tokens))
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def prepare_batch(
        self,
        batch_params: LlamaBatchParams,
    ):
        params = self.model.params
        bsz = len(batch_params.prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in batch_params.prompt_tokens)
        max_prompt_len = max(len(t) for t in batch_params.prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, batch_params.max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        token_logprobs = None
        for k, t in enumerate(batch_params.prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if batch_params.logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id

        batch = LlamaBatch(
            min_prompt_len=min_prompt_len,
            max_prompt_len=max_prompt_len,
            total_len=total_len,
            tokens=tokens,
            token_logprobs=token_logprobs,
            prev_pos=prev_pos,
            eos_reached=eos_reached,
            input_text_mask=input_text_mask,
        )
        return batch

    # def batch_tensors(seq_len, aux_dim, tensors: List[torch.Tensor]):
    #     bsz = len(tensors)

    #     batched_tensor = torch.zeros(
    #         (bsz, seq_len, aux_dim),
    #         dtype=tensors[0].dtype,
    #         device=tensors[0].device    # TODO: Disaggregate.
    #     )

    #     for idx in range(bsz):
    #         batched_tensor[idx, :tensors[idx].size[0], :] =  tensors[idx]
    #     return batched_tensor

    # @torch.inference_mode()
    # def step(
    #     self,
    #     tokens: List[List[int]],
    #     k_caches: List[torch.tensor],
    #     v_caches: List[torch.tensor],
    #     batch_params: LlamaBatchParams,
    # ):
    #     """
    #     tokens:
    #     - The ith element is the new tokens for the ith request in the batch. If
    #       we are doing a decode step, it is just one token (the last one that 
    #       was sampled).
    #     """
    #     # Pack the kv caches into a batch.
    #     params = self.model.params
    #     bsz = len(batch_params.prompt_tokens)
    #     assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    #     max_prompt_len = max(len(t) for t in batch_params.prompt_tokens)
    #     assert max_prompt_len <= params.max_seq_len
    #     total_max_len = params.max_seq_len + batch_params.max_gen_len

    #     batched_k_caches = torch.stack(k_caches)
    #     batched_v_caches = torch.stack(v_caches)
    #             # Pack the input tokens into a batch.
    #     max_prompt_len = max([len(seq_tokens) for seq_tokens in tokens])

    #     pad_id = self.tokenizer.pad_id
    #     padded_tokens = torch.full((bsz, max_prompt_len), pad_id, dtype=torch.long, device="cuda")
    #     for k, t in enumerate(tokens):
    #         padded_tokens[k, : len(t)] = \
    #         torch.tensor(t, dtype=torch.long, device="cuda")
    #         # Keep track of the indices of the first padding tokens in the kv cache.
    #     padding_start = []
    #     for t in prompt_tokens:
    #         padding_start.append(len(t))
    #             # Keep track of the start positions.
    #     # start_pos = 
    @torch.inference_mode()
    def step(
        self,
        batch: LlamaBatch,
        batch_params: LlamaBatchParams,
        right_edge: int,
    ):
        logits = self.model.forward(batch.tokens[:, batch.prev_pos:right_edge], batch.prev_pos)
        if batch_params.temperature > 0:
            probs = torch.softmax(logits[:, -1] / batch_params.temperature, dim=-1)
            next_token = sample_top_p(probs, batch_params.top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            batch.input_text_mask[:, right_edge], batch.tokens[:, right_edge], next_token
        )
        batch.tokens[:, right_edge] = next_token
        if batch_params.logprobs:
            batch.token_logprobs[:, batch.prev_pos + 1 : right_edge + 1] = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=batch.tokens[:, batch.prev_pos + 1 : right_edge + 1],
                reduction="none",
                ignore_index=self.tokenizer.pad_id,
            )
        batch.eos_reached |= (~batch.input_text_mask[:, right_edge]) & (
            torch.isin(next_token, self.stop_tokens)
        )
        batch.prev_pos = right_edge

    @torch.inference_mode()
    def prefill(self, batch: LlamaBatch, batch_params: LlamaBatchParams):
        if batch.min_prompt_len == batch.total_len: # No need to generate
            # NOTE: this is a bit intricate. Don't produce next token because we didn't allocate space for it!
            logits = self.model.forward(batch.tokens, batch.prev_pos)
            batch.token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=batch.tokens,
                reduction="none",
                ignore_index=batch.pad_id,
            )
        else:
            # NOTE: no chunked prefilled here.
            self.step(batch, batch_params, batch.min_prompt_len)

    @torch.inference_mode()
    def decode(self, batch: LlamaBatch, batch_params: LlamaBatchParams):
        if all(batch.eos_reached) or batch.prev_pos == batch.total_len - 1:
            return False
        self.step(batch, batch_params, batch.prev_pos + 1)
        return True

    @torch.inference_mode()
    def postprocess(self, batch: LlamaBatch, batch_params: LlamaBatchParams):
        if batch_params.logprobs:
            batch.token_logprobs = batch.token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(batch.tokens.tolist()):
            # Truncate to max gen len
            start = 0 if batch_params.echo else len(batch_params.prompt_tokens[i])
            toks = toks[start : len(batch_params.prompt_tokens[i]) + batch_params.max_gen_len]
            probs = None
            if batch_params.logprobs:
                probs = batch.token_logprobs[i][start : len(batch_params.prompt_tokens[i]) + batch_params.max_gen_len]
            # Truncate after eos if any
            for stop_token in self.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if batch_params.logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if batch_params.logprobs else None)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
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

        batch_params = LlamaBatchParams(prompt_tokens, max_gen_len, temperature, top_p, logprobs, echo)
        batch = self.prepare_batch(batch_params) 
        self.prefill(batch, batch_params)
        while True:
            should_continue = self.decode(batch, batch_params)
            if not should_continue:
                break
        return self.postprocess(batch, batch_params)


    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
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
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
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
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
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
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        prompt_tokens = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
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