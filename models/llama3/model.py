# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = (
        256  # make SwiGLU hidden layer size multiple of large power of 2
    )
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_seq_len: int = 2048

class VocabEmbedding(torch.nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
    
    def forward(self, input_):
        # Build the mask.
        input_mask = (input_ < 0) | (input_ >= self.num_embeddings)
        # Mask the input.
        masked_input = input_.clone()
        masked_input[input_mask] = 0
        # Get the embeddings.
        output = super().forward(masked_input)
        # Mask the output embedding.
        output[input_mask, :] = 0.0
        return output

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # ndim = x.ndim
    # assert 0 <= 1 < ndim
    # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # return freqs_cis.view(*shape)
    return freqs_cis.unsqueeze(2)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = (
            args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        )
        model_parallel_size = 1 # TODO: parallelize
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # NOTE: This was ColumnParallelLinear in the original code. See args.
        self.wq = torch.nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = torch.nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False
        )

        # NOTE: This was RowParallelLinear in the original code. See args.
        self.wo = torch.nn.Linear(
            args.n_heads * self.head_dim, args.dim, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        mask: Optional[torch.Tensor],
        layer_idx: int,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # TODO: Vectorized implementation.
        for sample_idx in range(bsz):
            curr_start_pos = start_pos[sample_idx]

            cache_k[
                sample_idx,
                curr_start_pos: curr_start_pos + seqlen,
                layer_idx,
                :,
            ] = xk[sample_idx].view(seqlen, -1)

            cache_v[
                sample_idx,
                curr_start_pos: curr_start_pos + seqlen,
                layer_idx,
                :,
            ] = xv[sample_idx].view(seqlen, -1)

        # Invalid indices will be ignored due to mask.
        keys = cache_k[:, :, layer_idx, :].view(
            bsz, -1, self.n_local_heads, self.head_dim
        )
        values = cache_v[:, :, layer_idx, :].view(
            bsz, -1, self.n_local_heads, self.head_dim
        )

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(
            1, 2
        )  # (bs, n_local_heads, max_seq_len, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, max_seq_len, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if mask is not None:
            scores = (
                scores + mask
            )  # (bs, n_local_heads, seqlen, cache_len + seqlen)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq).nan_to_num()

        output = torch.matmul(
            scores, values
        )  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        # NOTE: This was ColumnParallelLinear in the original code. See args.
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        # NOTE: This was RowParallelLinear in the original code. See args.
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        # NOTE: This was ColumnParallelLinear in the original code. See args.
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(
            self.attention_norm(x),
            start_pos,
            freqs_cis,
            cache_k,
            cache_v,
            mask,
            self.layer_id,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # NOTE: This was VocabParallelEmbedding in the original code. See args.
        self.tok_embeddings = VocabEmbedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # NOTE: This was ColumnParallelLinear in the original code. See args.
        self.output = torch.nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    # Mask returned will be added to attention scores of shape
    # (bs, n_local_heads, seqlen, max_seq_len).
    @torch.inference_mode()
    def build_attention_mask(
        self, prompt_len, cache_len, start_pos, first_pad_idx
    ):
        batch_size = start_pos.shape[0]

        mask = torch.zeros(
            batch_size,
            1,
            prompt_len,
            cache_len,
            device="cuda",
        )

        # Add mask for input tokens. TODO: Vectorized implementation.
        for sample_idx in range(batch_size):
            curr_pad_idx = first_pad_idx[sample_idx]
            curr_start_pos = start_pos[sample_idx]
            for input_seq_idx in range(prompt_len):
                mask[
                    sample_idx,
                    :,
                    input_seq_idx,
                    curr_start_pos + min(input_seq_idx + 1, curr_pad_idx):,
                ] = float("-inf")

        return mask

    @torch.inference_mode()
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        first_pad_idx: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        batch_size, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        # NOTE: scheduled requests might have different numbers of tokens
        # already outputted, so we need a different start_pos for each.

        # TODO: Vectorized implementation.
        freqs_cis = []
        for sample_idx in range(batch_size):
            curr_start_pos = start_pos[sample_idx]
            freqs_cis.append(
                self.freqs_cis[curr_start_pos: curr_start_pos + seqlen]
            )

        freqs_cis = torch.stack(freqs_cis)

        # NOTE: tokens.shape[1] is the maximum token length in current batch (decode = 1)
        mask = self.build_attention_mask(
            tokens.shape[1], cache_k.shape[1], start_pos, first_pad_idx
        )

        for layer_id, layer in enumerate(self.layers):
            h = layer(h, start_pos, freqs_cis, cache_k, cache_v, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
