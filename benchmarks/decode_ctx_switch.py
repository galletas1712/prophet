from typing import List
import uuid
import json
import torch

from pathlib import Path
from entrypoints.api import Request, RequestStage
from entrypoints.databatch import DecodeDataBatch
from models.llama3.model import ModelArgs, Transformer


class DummyModel:
    def __init__(
        self,
        ckpt_dir: str = "/home/ubuntu/model_weights/Meta-Llama-3-8B-Instruct",
        max_batch_size: int = 8,
        max_seq_len: int = 512,
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
        print(f"Model Initialized. Each query's KV size is: {kv_size} bytes")

    def forward(self, input_tokens, start_pos, first_pad_idx, cache_k, cache_v):
        # Run through model, populating KV caches.
        logits = self.model.forward(
            input_tokens,
            start_pos,
            first_pad_idx,
            cache_k,
            cache_v,
        )

        return logits


def test_rotating_preemption(
    model: DummyModel,
    preempt: bool,
    num_queries_in_sched: int = 16,
    num_iterations: int = 200):
    request_bank = []

    request_input_lengths = list(torch.randint(1, model.max_seq_len, (num_queries_in_sched,)))
    print(model.kv_dim[1:])
    for q in range(num_queries_in_sched):
        request_bank.append(
            Request(
                stage=RequestStage.DECODE,
                prompt=None,
                prompt_tokens=[x for x in range(request_input_lengths[q])],
                output_tokens=[0],
                max_gen_len=model.max_seq_len,  # Max it out
                request_id=uuid.uuid4(),
                cache_k=torch.zeros(model.kv_dim[1:], dtype=torch.bfloat16),
                cache_v=torch.zeros(model.kv_dim[1:], dtype=torch.bfloat16),
            )
        )

    decode_batch = DecodeDataBatch(
        model.max_batch_size,
        model.max_seq_len,
        model.model_args.n_layers,
        model.model_args.dim,
        -1 # TODO: jank
    )

    def forward():
        model.forward(decode_batch.input_tokens, decode_batch.start_pos, decode_batch.first_pad_idx, decode_batch.cache_k, decode_batch.cache_v)

    # Populate initial batch
    decode_batch.batch_preempt_slots(decode_batch.get_free_slots(), request_bank[:model.max_batch_size])

    # Warmup
    forward()
    
    torch.cuda.cudart().cudaProfilerStart()
    batch_start_event = torch.cuda.Event(enable_timing=True)
    batch_end_event = torch.cuda.Event(enable_timing=True)
    batch_start_event.record()

    for it in range(num_iterations):
        torch.cuda.nvtx.range_push(f"Iteration {it}")

        # Preemption logic
        if preempt:
            preempt_indices = [i for i in range(model.max_batch_size)]
            preempt_requests = [request_bank[(it * model.max_batch_size + i) % num_queries_in_sched] for i in preempt_indices]
            decode_batch.batch_preempt_slots(preempt_indices, preempt_requests)

        # Do the forward pass
        forward()

        torch.cuda.nvtx.range_pop()

    batch_end_event.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    elapsed_time_ms = batch_start_event.elapsed_time(batch_end_event)
    elapsed_time_per_iter = elapsed_time_ms / num_iterations
    print(f"Total elapsed time: {elapsed_time_ms} ms")
    print(f"Elapsed time per iteration: {elapsed_time_per_iter} ms")


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    torch.set_default_device("cuda")
    model=DummyModel()
    test_rotating_preemption(model, True)
    print()
    test_rotating_preemption(model, False)


# Measuring cost of context switches
# We're running a forward pass over a batch of queries, then changing the constituents of the batch in each iteration
# We're concerned with the cost of copying memory in/out of the batch
# How big are cache lines on the GPU? If we copy memory in/out of the batch, we should ideally copy in/out whole cache lines at a time otherwise we waste bandwidth
# The size of each copy is the number of queries we remove * the KV size of each query
# If a cache line is bigger than one query's KV size, we need to be more careful about which queries we evict, because then we may evict an entire cache line at a time.
# How big should the size of the copy be so that we can still hide latency? cudaAsync...

# TODO: change pattern in how we load requests. Different amounts of preemption, positions or preemption, etc

# TODO (Jack said): what's contributing to the overhead? Is it copying? Cache misses? How long does the copying actually take within a GPU? How long does CPU overhead take? Synchronization?