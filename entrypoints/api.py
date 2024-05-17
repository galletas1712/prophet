from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import torch


class RequestStage(Enum):
    PREFILL = 0
    DECODE = 1
    DONE = 2


@dataclass
class Request:
    request_id: int
    stage: RequestStage

    prompt_str: str
    output_str: Optional[str]  # Populated when request stage set to DONE.

    prompt_tokens: Optional[List[int]]  # Populated at prefill.
    output_tokens: List[int]

    idx_in_data_batch: Optional[int] = None

    def __init__(self, request_id, prompt_str):
        self.request_id = request_id
        self.stage = RequestStage.PREFILL

        self.prompt_str = prompt_str
        self.output_str = None

        self.prompt_tokens = None
        self.output_tokens = []


@dataclass
class PrefillDataBatch:
    # The list of requests the comprise this prefill batch
    requests: List[Request]

    # Token ids of shape (batch_size, padded_seq_len). Only includes the new
    # tokens the model needs to process; does not include tokens for which KV
    # cache state is already known.
    input_tokens: torch.Tensor

    # TODO: Log probs of the input_tokens and the output token. Has shape
    # (batch_size, padded_seq_len + 1).
    token_logprobs: Optional[torch.Tensor]

    # Position in the corresponding sequences of each entry in input_tokens. Has
    # shape (batch_size,).
    start_pos: torch.Tensor

    # Index of the first padding token in the input tokens for each sample in
    # the batch. If no padding, it is the index right after the last index of
    # the sample. Has shape (batch_size,).
    first_pad_idx: torch.Tensor

    # Shape (batch_size,). True if EOS has been generated.
    eos_reached: torch.Tensor

    # KV cache of shape (batch_size, max_seq_len, n_layers, model_dim).
    cache_k: torch.Tensor
    cache_v: torch.Tensor

    def __init__(
        self,
        requests: List[Request],
        max_seq_len: int,
        n_layers: int,
        dim: int,
        pad_token: int
    ):
        batch_size = len(requests)

        self.requests = requests

        # Build the input tokens tensor, consisting of tokens that haven't been
        # processed yet. If prefill, this is all input tokens. If decode, this
        # is the last outputted decode token.
        input_tokens = []
        max_input_tokens_len = 0

        for idx, request in enumerate(requests):
            input_tokens.append(request.prompt_tokens)
            request.idx_in_data_batch = idx
            max_input_tokens_len = max(
                max_input_tokens_len, len(input_tokens[-1])
            )

        # Set input tokens, padded to the maximum input length in the batch
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

        # Prefill always starts at index 0
        self.start_pos = torch.zeros(
            (batch_size,), dtype=torch.long, device="cuda"
        )

        self.first_pad_idx = torch.tensor([
            len(token_seq) for token_seq in input_tokens
        ], dtype=torch.long, device="cuda")

        # EOS reached is default false when constructing the batch since a
        # request is assumed to be never scheduled if EOS already reached.
        self.eos_reached = torch.zeros((batch_size,), dtype=torch.bool)

        # Preallocate max size kv cache TODO: preallocate only up to max_input_tokens_len?
        kv_dim = (batch_size, max_seq_len, n_layers, dim)
        self.cache_k = torch.zeros(kv_dim, dtype=torch.bfloat16, device="cuda")
        self.cache_v = torch.zeros(kv_dim, dtype=torch.bfloat16, device="cuda")


@dataclass
class DecodeDataBatch:
    # List of current requests, starts of as all None
    # We know we can replace a slot when the slot becomes None
    # NOTE: If the stage changes to DONE, we still have to clear the slot
    # from the scheduler's side
    requests: List[Optional[Request]]

    # Token ids of shape (batch_size, padded_seq_len). Only includes the new
    # tokens the model needs to process; does not include tokens for which KV
    # cache state is already known.
    input_tokens: torch.Tensor

    # TODO: Log probs of the input_tokens and the output token. Has shape
    # (batch_size, padded_seq_len + 1).
    token_logprobs: Optional[torch.Tensor]

    # Position in the corresponding sequences of each entry in input_tokens. Has
    # shape (batch_size,).
    start_pos: torch.Tensor

    # Index of the first padding token in the output tokens for each sample in
    # the batch. If no padding, it is the index right after the last index of
    # the sample. Has shape (batch_size,).
    first_pad_idx: torch.Tensor

    # Shape (batch_size,). True if EOS has been generated.
    eos_reached: torch.Tensor

    # KV cache of shape (batch_size, max_seq_len, n_layers, model_dim).
    cache_k: torch.Tensor
    cache_v: torch.Tensor

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_layers: int,
        dim: int,
        pad_token: int,
    ):
        self.pad_token = pad_token

        self.requests = [None for _ in range(max_batch_size)]

        self.input_tokens = torch.full(
            (max_batch_size, 1),
            pad_token,
            device="cuda"
        )
        self.start_pos = torch.zeros(
            (max_batch_size,), dtype=torch.long, device="cuda"
        )

        # NOTE: our sample length is always 1 for decode.
        # This field really is only useful for prefill
        # TODO: get rid of this entirely
        self.first_pad_idx = torch.ones(
            (max_batch_size,), dtype=torch.long, device="cuda"
        )

        # We set eos_reached to **true** first to handle empty slots
        self.eos_reached = torch.ones(
            (max_batch_size,), dtype=torch.bool, device="cuda"
        )

        # Preallocate max size kv cache
        kv_dim = (max_batch_size, max_seq_len, n_layers, dim)
        self.cache_k = torch.zeros(kv_dim, device="cuda")
        self.cache_v = torch.zeros(kv_dim, device="cuda")

    def fill_slot(self, idx: int, request: Request, cache_k: torch.Tensor, cache_v: torch.Tensor):
        # This property allows us to extract the kv cache at the right position
        request.idx_in_data_batch = idx
        self.requests[idx] = request

        # Set input_tokens to be the prompt_tokens, plus output_tokens so far
        # Output_tokens should only be one element from prefill
        prompt_len = len(request.prompt_tokens)
        output_len = len(request.output_tokens)
        assert output_len == 1
        self.input_tokens[idx] = torch.tensor(
            request.output_tokens, dtype=torch.long, device="cuda")

        # Only token we process
        self.start_pos[idx] = prompt_len + output_len - 1

        # EOS should now be false so we allow this slot to make progress
        self.eos_reached[idx] = False

        # Copy KV cache over
        # TODO: clone over RPC, also think about lengths
        self.cache_k[idx] = cache_k.clone()
        self.cache_v[idx] = cache_v.clone()

    def clear_slot(self, idx: int):
        self.requests[idx].idx_in_batch = None
        self.requests[idx] = None
        self.input_tokens[idx] = self.pad_token
        self.start_pos[idx] = 0
        # Again, set this to True to handle empty slots
        self.eos_reached[idx] = True
        self.cache_k[idx] = 0
        self.cache_v[idx] = 0
