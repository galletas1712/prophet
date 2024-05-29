import hydra
import torch
import torch.multiprocessing as mp

from queue import Empty

from entrypoints.api import WorkerType
from entrypoints.llm import LLM
from shareGPT import request_generator


def prefill_worker(
    config,
    device: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    model_load_barrier: mp.Barrier,
):
    torch.cuda.set_device(device)
    llm = LLM(
        config.model,
        config.prefill_scheduler,
        config.seed,
        worker_type=WorkerType.PREFILL,
    )
    model_load_barrier.wait()
    num_pending_batching = 0
    while True:
        while num_pending_batching < config.prefill_scheduler.batch_size:
            try:
                request = input_queue.get_nowait()
                # Terminate on None
                if request is None:
                    output_queue.put(None)
                    return
                llm.add_request(request)
                num_pending_batching += 1
            except Empty:
                # OK if no new requests, continue decoding old requests
                break

        prefill_batch_state = llm.step_prefill()
        if prefill_batch_state is None:
            continue

        for request in prefill_batch_state.requests:
            print(f"---- PREFILLED REQUEST {request.request_id} ----")
            output_queue.put(request)

        num_pending_batching = 0


def decode_worker(
    config,
    device: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    model_load_barrier: mp.Barrier,
):
    torch.cuda.set_device(device)
    llm = LLM(
        config.model,
        config.decode_scheduler,
        config.seed,
        worker_type=WorkerType.DECODE,
    )
    model_load_barrier.wait()
    while True:
        try:
            request = input_queue.get_nowait()
            # Terminate on None
            if request is None:
                output_queue.put(None)
                return
            # First token (from prefill)
            request.benchmark_metrics.received_token()
            llm.add_request(request)
        except Empty:
            # OK if no new requests, continue decoding old requests
            pass

        done_requests, request_batch = llm.step_decode()

        # Add benchmark times for decode tokens
        torch.cuda.synchronize()
        for request in request_batch:
            request.benchmark_metrics.received_token()

        # NOTE: KV cache was already deleted so it's fine to send whole request object
        for request in done_requests:
            output_queue.put(request)


@hydra.main(
    config_path="config/",
    config_name="disaggregated_llama_3",
    version_base=None,
)
def spawn_workers(config):
    mp.set_start_method("spawn")
    num_available_gpus = torch.cuda.device_count()

    # Assert disabled for single GPU testing.
    assert (
        config.coordinator.num_prefill_workers
        + config.coordinator.num_decode_workers
        <= num_available_gpus
    )

    prefill_queue = mp.Queue()
    decode_queue = mp.Queue()
    result_queue = mp.Queue()

    processes = []

    model_load_barrier = mp.Barrier(config.coordinator.num_prefill_workers + config.coordinator.num_decode_workers + 1)

    for i in range(config.coordinator.num_prefill_workers):
        worker_device = f"cuda:{min(i, num_available_gpus - 1)}"
        print("Starting prefill worker on", worker_device)
        worker = mp.Process(
            target=prefill_worker,
            args=(
                config,
                worker_device,
                prefill_queue,
                decode_queue,
                model_load_barrier,
            ),
        )
        processes.append(worker)

    for i in range(config.coordinator.num_decode_workers):
        worker_device = f"cuda:{min(config.coordinator.num_prefill_workers + i, num_available_gpus - 1)}"
        print("Starting decode worker on", worker_device)

        worker = mp.Process(
            target=decode_worker,
            args=(
                config,
                worker_device,
                decode_queue,
                result_queue,
                model_load_barrier
            ),
        )
        processes.append(worker)

    for p in processes:
        p.start()

    model_load_barrier.wait()

    # Start request generator after we start prefill and decode processes
    request_generator_process = mp.Process(
        target=request_generator,
        args=(prefill_queue,
                config.coordinator.num_prefill_workers
        )
    )
    processes.append(request_generator_process)
    request_generator_process.start()

    f = open('benchmark_results.csv', 'w')
    f.write('JCT,TTFT,TPOT,TTFPT,TPODT\n')
    f.close()

    # Main process waits for outputs to complete
    while True:
        try:
            request = result_queue.get_nowait()
            # Terminate on None
            if request is None:
                break
            # First token (from prefill)
            request.benchmark_metrics.finished_request()

            # Write benchmark results to file
            f = open('benchmark_results.csv', 'a')
            f.write(request.benchmark_metrics.to_csv_row())
            f.close()

            print(f"---- COMPLETED REQUEST {request.request_id} ----")
            print(request.output)
            print(request.benchmark_metrics)
            print()
        except Empty:
            # OK if no new requests, continue decoding old requests
            pass

    for p in processes:
        p.join()


if __name__ == "__main__":
    spawn_workers()
