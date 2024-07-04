import gc
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
        del self.tensor
        gc.collect()


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
        del self.tensor
        gc.collect()


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

    source, target = ray_setup(source_device, target_device, dtype)
    byte_sizes = [1, 16, 64, 1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]

    bits = torch.finfo(dtype).bits
    for byte_size in byte_sizes:
        dim = byte_size // (bits * 8)

        times = []
        for i in range(num_iterations):
            # Wait for target to preallocate memory
            ray.get([
                source.preallocate.remote(dim),
                target.preallocate.remote(dim)
            ])

            start_time = time.time()
            ray.get([
                source.send_tensor.remote(target_device),
                target.receive_tensor.remote(source_device)
            ])
            end_time = time.time()
            if i > 1:  # A few warmup iterations
                times.append(end_time - start_time)
        print(f"Average time for {byte_size} bytes: {sum(times) / len(times)} seconds")