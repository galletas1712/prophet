import gc
import ray
import time
import torch

import ray.util.collective as collective


@ray.remote(num_gpus=1)
class Source:
    def __init__(self, dtype):
        torch.set_default_device("cuda")
        self.dtype = dtype
    
    def preallocate(self, byte_size):
        self.tensor = torch.rand((byte_size,), dtype=self.dtype, device="cuda")

    def send_tensor(self, target_rank: int):
        collective.send(self.tensor, target_rank)
        torch.cuda.synchronize()

        # Free memory
        # del self.tensor
        # gc.collect()
        # torch.cuda.empty_cache()


@ray.remote(num_gpus=1)
class Target:
    def __init__(self, dtype):
        torch.set_default_device("cuda")
        self.dtype = dtype
    
    def preallocate(self, max_byte_size):
        self.tensor = torch.zeros((max_byte_size,), dtype=self.dtype, device="cuda")
    
    def receive_tensor(self, source_rank: int):
        collective.recv(self.tensor, source_rank)
        torch.cuda.synchronize()

        # Free memory
        # del self.tensor
        # gc.collect()
        # torch.cuda.empty_cache()


def setup(source_device, target_device, dtype):
    if not ray.is_initialized():
        ray.init()
    
    source = Source.options(name="source").remote(dtype)
    target = Target.options(name="target").remote(dtype)

    # Initialize NCCL
    collective.create_collective_group([source, target], world_size=2, ranks=[source_device, target_device])

    return source, target


def gpu2gpu_transfer(source_device, target_device, dtype=torch.bfloat16):
    source, target = setup(source_device, target_device, dtype)
    byte_sizes = [1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]

    bits = torch.finfo(dtype).bits
    for byte_size in byte_sizes:
        dim = byte_size // (bits * 8)

        times = []
        for i in range(10):
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
    
# NOTE: No need to run more than once because we clear the cache every time

if __name__ == '__main__':
    gpu2gpu_transfer(0, 1, torch.bfloat16)