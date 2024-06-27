import gc
import ray
import torch

import ray.util.collective as collective


BYTE_SIZES = [1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]


def mem_transfer_inputs(dtype, device):
    bits = torch.finfo(dtype).bits

    for byte_size in BYTE_SIZES:
        dim = byte_size // (bits * 8)

        tensor = torch.rand((dim,), dtype=dtype, device=device)
        yield tensor
        del tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@ray.remote(num_gpus=1)
class Source:
    def __init__(self, dtype):
        torch.set_default_device("cuda")
        self.tensors_to_send = mem_transfer_inputs(dtype, "cuda")

    def send_tensor(self, target_rank: int):
        tensor = next(self.tensors_to_send)
        collective.send(tensor, target_rank)
        torch.cuda.synchronize()

        # Free memory
        del kv_cache
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Sent tensor to target")


@ray.remote(num_gpus=1)
class Target:
    def __init__(self, dtype):
        torch.set_default_device("cuda")
        self.dtype = dtype
    
    def preallocate(self):
        self.tensor = torch.zeros((max(BYTE_SIZES),), dtype=self.dtype)
    
    def receive_tensor(self, source_rank: int):
        collective.recv(self.tensor, source_rank)
        torch.cuda.synchronize()


def gpu2gpu_transfer(source_device, target_device, dtype=torch.bfloat16):
    if not ray.is_initialized():
        ray.init()
    
    source = Source.options(name="source").remote(dtype)
    target = Target.options(name="target").remote(dtype)

    # Wait for target to preallocate memory
    ray.get(target.preallocate.remote())

    # Initialize NCCL
    collective.create_collective_group([source, target], world_size=2, ranks=[source_device, target_device])

    ray.get([
        source.send_tensor.remote(target_device),
        target.receive_tensor.remote(source_device)])

# TODO: run each > once because cache misses