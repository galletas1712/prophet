import pandas as pd
import ray
import ray.util.collective as collective
import time
import torch

from typing import List


@ray.remote(num_gpus=1)
class Source:
    def __init__(self, dtype):
        torch.set_default_device("cuda")
        self.dtype = dtype
    
    def preallocate(self, size):
        self.tensor = torch.rand((size,), dtype=self.dtype, device="cuda")

    def send_tensor(self, target_rank: int):
        collective.send(self.tensor, target_rank)
        torch.cuda.synchronize()


@ray.remote(num_gpus=1)
class Target:
    def __init__(self, dtype):
        torch.set_default_device("cuda")
        self.dtype = dtype
    
    def preallocate(self, size):
        self.tensor = torch.rand((size,), dtype=self.dtype, device="cuda")
    
    def receive_tensor(self, source_rank: int):
        collective.recv(self.tensor, source_rank)
        torch.cuda.synchronize()


def ray_setup(source_device, target_device, dtype):
    if not ray.is_initialized():
        ray.init()
    
    source = Source.options(name="source").remote(dtype)
    target = Target.options(name="target").remote(dtype)

    # Initialize NCCL
    collective.create_collective_group([source, target], world_size=2, ranks=[source_device, target_device])

    return source, target


def gpu2gpu_transfer(num_iterations: int, byte_sizes: List[int], dtype: torch.dtype, source_device: int, target_device: int):
    print("GPU to GPU transfer")
    elapsed_time = []
    elapsed_time_per_iter = []

    source, target = ray_setup(source_device, target_device, dtype)

    bits = torch.finfo(dtype).bits
    for byte_size in byte_sizes:
        dim = byte_size // (bits * 8)

        # Wait for target to preallocate memory
        ray.get([
            source.preallocate.remote(dim),
            target.preallocate.remote(dim)
        ])

        # Warmup
        ray.get([
            source.send_tensor.remote(target_device),
            target.receive_tensor.remote(source_device)
        ])

        start_time = time.perf_counter_ns()
        for i in range(num_iterations):
            ray.get([
                source.send_tensor.remote(target_device),
                target.receive_tensor.remote(source_device)
            ])
        end_time = time.perf_counter_ns()
        elapsed_time_ms = (end_time - start_time) / 1e6
        elapsed_time_per_iter_ms = elapsed_time_ms / num_iterations

        elapsed_time.append(elapsed_time_ms)
        elapsed_time_per_iter.append(elapsed_time_per_iter_ms)
    
    df = pd.DataFrame({
        "Byte Size": byte_sizes,
        "Elapsed Time (ms)": elapsed_time,
        "Elapsed Time per Iteration (ms)": elapsed_time_per_iter
    })

    return df


if __name__ == '__main__':
    num_iterations = 50
    byte_sizes = [1, 16, 64, 1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]
    dtype = torch.bfloat16
    source_device = 0
    target_device = 1
    df = gpu2gpu_transfer(num_iterations, byte_sizes, dtype, source_device, target_device)
    print(df)
