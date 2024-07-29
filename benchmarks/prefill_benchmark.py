from typing import List
import uuid
import json
import torch

from pathlib import Path
from entrypoints.api import Request, RequestStage
from entrypoints.databatch import PrefillDataBatch
from models.llama3.model import ModelArgs, Transformer


class DummyModel:
    def __init__(
        self,
        ckpt_dir: str = "/root/model_weights/Meta-Llama-3-8B-Instruct",
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
            model_args: ModelArgs = ModelArgs(
                max_seq_len=max_seq_len,
                **params,
            )

        self.model_args = model_args

        self.kv_dim = (max_batch_size, max_seq_len, model_args.n_layers, model_args.dim)
        print(model_args)

        self.model = Transformer(model_args)

        # Print out size of KV cache for each query
        kv_size = model_args.max_seq_len * model_args.n_layers * model_args.dim * (torch.finfo(torch.bfloat16).bits // 8)
        print(f"Model Initialized. Each query's K/V size is: {kv_size} bytes, combined is  {kv_size * 2} bytes. Total batch kv size is {2 * kv_size * max_batch_size} bytes")

    def forward(self, input_tokens, start_pos, first_pad_idx, cache_k, cache_v, mode):
        # Run through model, populating KV caches.
        logits = self.model.forward(
            input_tokens,
            start_pos,
            first_pad_idx,
            cache_k,
            cache_v,
            mode,
        )

        return logits


def test_prefill_time(
    batch_size: int,
    seqlen: int,
    num_iterations: int = 10):

    print(f"Testing batch size: {batch_size}, sequence length: {seqlen}")
    torch.cuda.nvtx.range_push(f"bs: {batch_size}, seqlen: {seqlen}")
    
    model = DummyModel(max_batch_size=batch_size, max_seq_len=seqlen)
    requests = [
        Request(
            stage=RequestStage.PREFILL,
            prompt=None,
            prompt_tokens=[x for x in range(seqlen)],
            output_tokens=[],
            max_gen_len=seqlen,  # Max it out
            request_id=uuid.uuid4()
        )
    ]

    prefill_batch = PrefillDataBatch(requests, seqlen, model.model_args.n_layers, model.model_args.dim, -1)

    def forward():
        model.forward(
            prefill_batch.input_tokens,
            prefill_batch.start_pos,
            prefill_batch.first_pad_idx,
            prefill_batch.cache_k,
            prefill_batch.cache_v,
            RequestStage.PREFILL,
        )

    # Warmup
    forward()
    
    torch.cuda.cudart().cudaProfilerStart()
    batch_start_event = torch.cuda.Event(enable_timing=True)
    batch_end_event = torch.cuda.Event(enable_timing=True)
    batch_start_event.record()

    import gc

    for it in range(num_iterations):
        torch.cuda.nvtx.range_push(f"Iteration {it}")

        # Just to get rid of any potential GC stalls in the pipeline later on in the iteration
        # TODO: REMOVE! This isn't realistic, but we want to isolate GC stalls for now
        torch.cuda.nvtx.range_push("GC")
        gc.collect()
        torch.cuda.nvtx.range_pop()

        # Do the forward pass
        torch.cuda.nvtx.range_push("Forward pass")
        forward()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()

    batch_end_event.record()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    elapsed_time_ms = batch_start_event.elapsed_time(batch_end_event)
    elapsed_time_per_iter = elapsed_time_ms / num_iterations
    print(f"Total elapsed time: {elapsed_time_ms} ms")
    print(f"Elapsed time per iteration: {elapsed_time_per_iter} ms")


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    torch.set_default_device("cuda")

    for bs in [1, 2, 4, 8, 16]:
        for seqlen in [512, 1024, 2048, 4096]:
            test_prefill_time(bs, seqlen)
