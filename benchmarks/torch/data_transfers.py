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


def ray_setup(source_device, target_device, dtype):
    if not ray.is_initialized():
        ray.init()
    
    source = Source.options(name="source").remote(dtype)
    target = Target.options(name="target").remote(dtype)

    # Initialize NCCL
    collective.create_collective_group([source, target], world_size=2, ranks=[source_device, target_device])

    return source, target


def gpu2gpu_transfer(source_device, target_device, dtype):
    print("GPU to GPU transfer")
    source, target = ray_setup(source_device, target_device, dtype)
    byte_sizes = [1, 16, 64, 1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]

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


def gpu2cpu_transfer_no_pin(dtype):
    print("GPU to CPU transfer without pinned memory")
    byte_sizes = [1, 16, 64, 1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]
    num_iterations = 50

    bits = torch.finfo(dtype).bits
    for byte_size in byte_sizes:
        dim = byte_size // (bits * 8)

        cpu_tensor = torch.rand((dim,), dtype=dtype, device="cpu", pin_memory=False)
        gpu_tensor = torch.rand((dim,), dtype=dtype, device="cuda", pin_memory=False)

        # Warmup
        cpu_tensor.copy_(gpu_tensor)

        torch.cuda.cudart().cudaProfilerStart()
        batch_start_event = torch.cuda.Event(enable_timing=True)
        batch_end_event = torch.cuda.Event(enable_timing=True)
        batch_start_event.record()
        for it in range(num_iterations):
            torch.cuda.nvtx.range_push(f"Iteration {it}")
            cpu_tensor.copy_(gpu_tensor)
            torch.cuda.nvtx.range_pop()

        batch_end_event.record()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        elapsed_time_ms = batch_start_event.elapsed_time(batch_end_event)
        elapsed_time_per_iter = elapsed_time_ms / num_iterations
        print(f"Total elapsed time: {elapsed_time_ms} ms")
        print(f"Elapsed time per iteration: {elapsed_time_per_iter} ms")


def gpu2cpu_transfer_pinned(dtype):
    print("GPU to CPU transfer without pinned memory")
    byte_sizes = [1, 16, 64, 1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]
    num_iterations = 50

    bits = torch.finfo(dtype).bits
    for byte_size in byte_sizes:
        dim = byte_size // (bits * 8)

        cpu_tensor = torch.rand((dim,), dtype=dtype, device="cpu", pin_memory=True)
        gpu_tensor = torch.rand((dim,), dtype=dtype, device="cuda")

        # Warmup
        cpu_tensor.copy_(gpu_tensor)

        torch.cuda.cudart().cudaProfilerStart()
        batch_start_event = torch.cuda.Event(enable_timing=True)
        batch_end_event = torch.cuda.Event(enable_timing=True)
        batch_start_event.record()
        for it in range(num_iterations):
            torch.cuda.nvtx.range_push(f"Iteration {it}")
            cpu_tensor.copy_(gpu_tensor)
            torch.cuda.nvtx.range_pop()

        batch_end_event.record()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


        elapsed_time_ms = batch_start_event.elapsed_time(batch_end_event)
        elapsed_time_per_iter = elapsed_time_ms / num_iterations
        print(f"Total elapsed time: {elapsed_time_ms} ms")
        print(f"Elapsed time per iteration: {elapsed_time_per_iter} ms")

def cpu2gpu_transfer_no_pin(dtype):
    print("CPU to GPU transfer without pinned memory")
    byte_sizes = [1, 16, 64, 1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]
    num_iterations = 50

    bits = torch.finfo(dtype).bits
    for byte_size in byte_sizes:
        dim = byte_size // (bits * 8)

        cpu_tensor = torch.rand((dim,), dtype=dtype, device="cpu", pin_memory=False)
        gpu_tensor = torch.rand((dim,), dtype=dtype, device="cuda", pin_memory=False)

        # Warmup
        gpu_tensor.copy_(cpu_tensor)

        torch.cuda.cudart().cudaProfilerStart()
        batch_start_event = torch.cuda.Event(enable_timing=True)
        batch_end_event = torch.cuda.Event(enable_timing=True)
        batch_start_event.record()
        for it in range(num_iterations):
            torch.cuda.nvtx.range_push(f"Iteration {it}")
            gpu_tensor.copy_(cpu_tensor)
            torch.cuda.nvtx.range_pop()

        batch_end_event.record()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        elapsed_time_ms = batch_start_event.elapsed_time(batch_end_event)
        elapsed_time_per_iter = elapsed_time_ms / num_iterations
        print(f"Total elapsed time: {elapsed_time_ms} ms")
        print(f"Elapsed time per iteration: {elapsed_time_per_iter} ms")


def cpu2gpu_transfer_pinned(dtype):
    print("CPU to GPU transfer with pinned memory")
    byte_sizes = [1, 16, 64, 1024, 16*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024, 1024*1024*1024, 16*1024*1024*1024]
    num_iterations = 50

    bits = torch.finfo(dtype).bits
    for byte_size in byte_sizes:
        dim = byte_size // (bits * 8)

        cpu_tensor = torch.rand((dim,), dtype=dtype, device="cpu", pin_memory=True)
        gpu_tensor = torch.rand((dim,), dtype=dtype, device="cuda")

        # Warmup
        gpu_tensor.copy_(cpu_tensor)

        torch.cuda.cudart().cudaProfilerStart()
        batch_start_event = torch.cuda.Event(enable_timing=True)
        batch_end_event = torch.cuda.Event(enable_timing=True)
        batch_start_event.record()
        for it in range(num_iterations):
            torch.cuda.nvtx.range_push(f"Iteration {it}")
            gpu_tensor.copy_(cpu_tensor)
            torch.cuda.nvtx.range_pop()

        batch_end_event.record()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


        elapsed_time_ms = batch_start_event.elapsed_time(batch_end_event)
        elapsed_time_per_iter = elapsed_time_ms / num_iterations
        print(f"Total elapsed time: {elapsed_time_ms} ms")
        print(f"Elapsed time per iteration: {elapsed_time_per_iter} ms")

if __name__ == '__main__':
    dtype = torch.bfloat16
    gpu2gpu_transfer(0, 1, dtype)
    print()
    gpu2cpu_transfer_no_pin(dtype)
    print()
    gpu2cpu_transfer_pinned(dtype)
    print()
    cpu2gpu_transfer_no_pin(dtype)
    print()
    cpu2gpu_transfer_pinned(dtype)

# TODO (Jack said): cpu intervention at each iteration of gpu2gpu, but also within the same gpu
# TODO (Jack said): data transfers within gpu