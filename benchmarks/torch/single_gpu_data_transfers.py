from types import FunctionType
from typing import List
import torch
import pandas as pd

def block_transfer_wrapper(
    num_iterations: int,
    byte_sizes: List[int],
    dtype: torch.dtype,
    devices: List[torch.device],
    pin_memories: List[bool],
    warmup: FunctionType,
    op: FunctionType,
):
    bits = torch.finfo(dtype).bits
    elapsed_time = []
    elapsed_time_per_iter = []
    for byte_size in byte_sizes:
        dim = byte_size // (bits * 8)

        tensors = [
            torch.rand((dim,), dtype=dtype, device=device, pin_memory=pin_memory)
            for device, pin_memory in zip(devices, pin_memories)
        ]

        # Warmup
        warmup(tensors)

        # For single-gpu stuff
        torch.cuda.cudart().cudaProfilerStart()
        batch_start_event = torch.cuda.Event(enable_timing=True)
        batch_end_event = torch.cuda.Event(enable_timing=True)
        batch_start_event.record()
        for it in range(num_iterations):
            torch.cuda.nvtx.range_push(f"Iteration {it}")
            op(tensors)
            torch.cuda.nvtx.range_pop()

        batch_end_event.record()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        elapsed_time_ms = batch_start_event.elapsed_time(batch_end_event)
        elapsed_time_per_iter_ms = elapsed_time_ms / num_iterations

        elapsed_time.append(elapsed_time_ms)
        elapsed_time_per_iter.append(elapsed_time_per_iter_ms)
    
    df = pd.DataFrame({
        "Byte Size": byte_sizes,
        "Elapsed Time (ms)": elapsed_time,
        "Elapsed Time per Iteration (ms)": elapsed_time_per_iter
    })

    return df


def gpu2cpu_transfer_no_pin(num_iterations: int, byte_sizes: List[int], dtype: torch.dtype):
    print("GPU to CPU transfer without pinned memory")
    return block_transfer_wrapper(
        num_iterations,
        byte_sizes,
        dtype,
        devices=[torch.device("cuda"), torch.device("cpu")],
        pin_memories=[False, False],
        warmup=lambda tensors: tensors[1].copy_(tensors[0]),
        op=lambda tensors: tensors[1].copy_(tensors[0])
    )


def gpu2cpu_transfer_pinned(num_iterations: int, byte_sizes: List[int], dtype: torch.dtype):
    print("GPU to CPU transfer with pinned memory")
    return block_transfer_wrapper(
        num_iterations,
        byte_sizes,
        dtype,
        devices=[torch.device("cuda"), torch.device("cpu")],
        pin_memories=[False, True],
        warmup=lambda tensors: tensors[1].copy_(tensors[0]),
        op=lambda tensors: tensors[1].copy_(tensors[0])
    )

def cpu2gpu_transfer_no_pin(num_iterations: int, byte_sizes: List[int], dtype: torch.dtype):
    print("CPU to GPU transfer without pinned memory")
    return block_transfer_wrapper(
        num_iterations,
        byte_sizes,
        dtype,
        devices=[torch.device("cpu"), torch.device("cuda")],
        pin_memories=[False, False],
        warmup=lambda tensors: tensors[1].copy_(tensors[0]),
        op=lambda tensors: tensors[1].copy_(tensors[0])
    )


def cpu2gpu_transfer_pinned(num_iterations: int, byte_sizes: List[int], dtype: torch.dtype):
    print("CPU to GPU transfer with pinned memory")
    return block_transfer_wrapper(
        num_iterations,
        byte_sizes,
        dtype,
        devices=[torch.device("cpu"), torch.device("cuda")],
        pin_memories=[True, False],
        warmup=lambda tensors: tensors[1].copy_(tensors[0]),
        op=lambda tensors: tensors[1].copy_(tensors[0])
    )


def intragpu_transfer(num_iterations: int, byte_sizes: List[int], dtype: torch.dtype):
    print("Intra-GPU transfer")
    return block_transfer_wrapper(
        num_iterations,
        byte_sizes,
        dtype,
        devices=[torch.device("cuda"), torch.device("cuda")],
        pin_memories=[False, False],
        warmup=lambda tensors: tensors[1].copy_(tensors[0]),
        op=lambda tensors: tensors[1].copy_(tensors[0])
    )
    

if __name__ == '__main__':
    num_iterations = 100
    byte_sizes = [1, 16, 64, 1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]
    dtype = torch.bfloat16

    dfs = {
        "gpu2cpu_transfer_no_pin": gpu2cpu_transfer_no_pin(num_iterations, byte_sizes, dtype),
        "gpu2cpu_transfer_pinned": gpu2cpu_transfer_pinned(num_iterations, byte_sizes, dtype),
        "cpu2gpu_transfer_no_pin": cpu2gpu_transfer_no_pin(num_iterations, byte_sizes, dtype),
        "cpu2gpu_transfer_pinned": cpu2gpu_transfer_pinned(num_iterations, byte_sizes, dtype),
        "intragpu_transfer": intragpu_transfer(num_iterations, byte_sizes, dtype),
    }

    agg_df = pd.concat(dfs.values(), keys=dfs.keys())
    print(agg_df)


# TODO (Jack said): cpu intervention at each iteration of gpu2gpu, but also within the same gpu