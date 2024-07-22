import pandas as pd
import ray
import time
import torch
import os

from typing import List


@ray.remote
class Coordinator:
    def __init__(self, rank_mapping):
        self.rank_mapping = rank_mapping

    def send_tensor(self, sender_name, receiver_name):
        sender = ray.get_actor(sender_name)
        receiver = ray.get_actor(receiver_name)
        ray.get([
            sender.send_tensor.remote(target_rank=self.rank_mapping[receiver_name]),
            receiver.receive_tensor.remote(src_rank=self.rank_mapping[sender_name])
        ])


@ray.remote(num_cpus=4, num_gpus=1, runtime_env={"nsight": {"s": "none"}})  # Disable CPU profiling
class Source:
    def __init__(self, dtype):
        torch.set_default_device("cuda")
        self.dtype = dtype
        self.i = 0
    
    def ddp_setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size) 

    def preallocate(self, tensor_size):
        self.tensor = torch.rand((4, tensor_size), dtype=self.dtype, device="cuda")

    def send_tensor(self, target_rank: int):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.nvtx.range_push("send")
        start_event.record()
        torch.cuda.synchronize()
        torch.distributed.send(tensor=self.tensor[self.i], dst=target_rank)
        end_event.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        self.i = (self.i + 3) % 4

        print(f"Send tensor took {start_event.elapsed_time(end_event)} ms")


@ray.remote(num_cpus=4, num_gpus=1, runtime_env={"nsight": {"s": "none"}})  # Disable CPU profiling
class Target:
    def __init__(self, dtype, coordinator):
        torch.set_default_device("cuda")
        self.dtype = dtype
        self.coordinator = coordinator
        self.i = 0
    
    def ddp_setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size) 
    
    def preallocate(self, tensor_size):
        self.tensor = torch.rand((4, tensor_size), dtype=self.dtype, device="cuda")
    
    def receive_tensor(self, src_rank: int):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.nvtx.range_push("receive")
        start_event.record()
        torch.cuda.synchronize()
        torch.distributed.recv(tensor=self.tensor[self.i], src=src_rank)
        end_event.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        self.i = (self.i + 3) % 4

        print(f"Receive tensor took {start_event.elapsed_time(end_event)} ms")
    
    def pull_tensor(self):
        start_pc = time.perf_counter_ns()
        ray.get(self.coordinator.send_tensor.remote("source", "target"))
        end_pc = time.perf_counter_ns()

        print(f"Pull tensor took {(end_pc - start_pc) * 1e6} ms")


def ray_setup(source_device, target_device, dtype):
    if not ray.is_initialized():
        ray.init()

    rank_mapping = {
        "source": source_device,
        "target": target_device
    }

    coordinator = Coordinator.remote(rank_mapping)

    source = Source.options(name="source", max_concurrency=2).remote(dtype)
    target = Target.options(name="target", max_concurrency=2).remote(dtype, coordinator)

    # Initialize DDP
    ray.get([
        source.ddp_setup.remote(source_device, 2),
        target.ddp_setup.remote(target_device, 2),
    ])

    return source, target, coordinator


def gpu2gpu_transfer(num_iterations: int, byte_sizes: List[int], dtype: torch.dtype, source_device: int, target_device: int):
    print("GPU to GPU transfer")
    elapsed_time = []
    elapsed_time_per_iter = []

    source, target, coordinator = ray_setup(source_device, target_device, dtype)

    bits = torch.finfo(dtype).bits
    for byte_size in byte_sizes:
        dim = byte_size * 8 // bits

        # Wait for target to preallocate memory
        ray.get([
            source.preallocate.remote(dim),
            target.preallocate.remote(dim)
        ])

        # Warmup
        for _ in range(4):
            ray.get(target.pull_tensor.remote())
        
        start_time = time.perf_counter_ns()
        for _ in range(num_iterations):
            ray.get(target.pull_tensor.remote())
        end_time = time.perf_counter_ns()
            
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
    num_iterations = 100
    byte_sizes = [1024, 4*1024, 16*1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024, 1024*1024*1024]
    dtype = torch.bfloat16
    source_device = 0
    target_device = 1
    df = gpu2gpu_transfer(num_iterations, byte_sizes, dtype, source_device, target_device)
    print(df)
