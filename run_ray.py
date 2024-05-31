# Somehow we need to do this before importing Ray for no log dedup
import os
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['RAY_COLOR_PREFIX'] = '1'

import torch
import hydra
import gc
import asyncio
import ray
from entrypoints.api import WorkerType
from entrypoints.llm import LLM
from ray.util.queue import Queue, Empty
from shareGPT_ray import ShareGPTRequestGenerator


shareGPTPath = '/home/ubuntu/shareGPT.json'
tokenizer_path = '/home/ubuntu/model_weights/Meta-Llama-3-8B-Instruct/tokenizer.model'


# class PrefillDataBatchManager:
#     def __init__(self):
#         self.prefill_data_batches = {}
#         self.request_id_to_batch_id = {}
#         self.requests_still_in_batch = {}

#     def add_prefill_data_batch(self, prefill_data_batch: PrefillDataBatch):
#         self.prefill_data_batches[prefill_data_batch.batch_id] = prefill_data_batch
#         self.request_id_to_batch_id.update({r.request_id: prefill_data_batch.batch_id for r in prefill_data_batch.requests})
#         self.num_requests_still_in_batch = len(prefill_data_batch.requests)

#     def pop_request_kv(self, request: Request):
#         batch_id = self.request_id_to_batch_id.pop(request.request_id)

#         cache_k = self.prefill_data_batches[batch_id].cache_k[request.idx_in_data_batch, :len(request.prompt_tokens)]
#         cache_v = self.prefill_data_batches[batch_id].cache_v[request.idx_in_data_batch, :len(request.prompt_tokens)]

#         self.num_requests_still_in_batch[batch_id] -= 1
#         if self.num_requests_still_in_batch[batch_id] == 0:
#             prefill_data_batch = self.prefill_data_batches.pop(batch_id)
#             del prefill_data_batch.cache_k
#             del prefill_data_batch.cache_v
#             del prefill_data_batch
#             # NOTE: doesn't free GPU memory yet because we still have a reference to the tensor in cache_k and cache_v
#             self.requests_still_in_batch.pop(batch_id)

#         return cache_k, cache_v


@ray.remote(num_cpus=4, num_gpus=1)
class Prefiller:
    def __init__(
        self,
        config,
        input_queue: Queue,
        output_queue: Queue,
    ):
        torch.set_default_device("cuda")
        self.rank = ray.get_gpu_ids()[0]

        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue

    def load_llm(self):
        print(f"Prefiller(rank={self.rank}) initializing LLM...")
        self.llm = LLM(
            self.config.model,
            self.config.prefill_scheduler,
            self.config.seed,
            worker_type=WorkerType.PREFILL,
        )
        print(f"Prefiller(rank={self.rank}) done initializing LLM!")

    def __repr__(self):
        return f"Prefiller(rank={self.rank})"

    def get_next_batch_ref(self):
        self.next_batch_ref = [
            self.input_queue.get_async(
                block=True,
                timeout=self.config.coordinator.dequeue_timeout
            )
            for _ in range(self.config.prefill_scheduler.batch_size)
        ]

    async def run(self):
        print(f"Starting on GPU {ray.get_gpu_ids()}")
        self.get_next_batch_ref()
        requests_to_process = []
        while True:
            dequeue_results = await asyncio.gather(*self.next_batch_ref, return_exceptions=True)
            requests_to_process = list(
                filter(lambda x: not isinstance(x, Empty), dequeue_results))
            self.get_next_batch_ref()

            # Do work
            for request in requests_to_process:
                self.llm.add_request(request)

            if len(requests_to_process) > 0:
                print(
                    f"Prefiller processing batch of {len(requests_to_process)} requests")
                print(
                    f"Prefiller requests: {[request.request_id for request in requests_to_process]}")

            prefill_data_batch = self.llm.step_prefill()

            if prefill_data_batch is None:
                continue

            # Add to pending queue
            for request in requests_to_process:
                # NOTE: Important to block until queue is free
                request.cache_k = request.cache_k.detach().cpu()
                request.cache_v = request.cache_v.detach().cpu()
                await self.output_queue.put_async(request)

            # Free memory from prefill data batch
            del prefill_data_batch.cache_k
            del prefill_data_batch.cache_v
            gc.collect()
            torch.cuda.empty_cache()


@ray.remote(num_cpus=4, num_gpus=1)
class Decoder:
    def __init__(
        self,
        config,
        input_queue: Queue,
        output_queue: Queue,
    ):
        torch.set_default_device("cuda")
        self.rank = ray.get_gpu_ids()[0]

        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.num_scheduler_slots = int(self.config.max_in_progress_factor *
                                       self.config.decode_scheduler.batch_size)

    def load_llm(self):
        print(f"Decoder(rank={self.rank}) initializing LLM...")
        self.llm = LLM(
            self.config.model,
            self.config.decode_scheduler,
            self.config.seed,
            worker_type=WorkerType.DECODE,
        )
        print(f"Decoder(rank={self.rank}) done initializing LLM!")

    def __repr__(self):
        return f"Decoder(rank={self.rank})"

    async def dequeue_request(self):
        # Wait for a very small amount of time, otherwise move on
        request = await self.input_queue.get_async(
            block=True,
            timeout=self.config.coordinator.dequeue_timeout
        )

        # # NOTE: right now idx_in_batch is still the same as prefill
        # # TODO: make this not block
        # asyncio.gather(self.prefill_actor.send_kv(request), col.get())

        return request

    async def run(self):
        print(f"Starting on GPU {ray.get_gpu_ids()}")
        while True:
            # TODO: Pipeline! This blocks on I/O from prefill process right now

            # NOTE: The factor k * scheduler.batch_size is the maximum number of requests that can be pending in the scheduler
            # Note that this is NOT the max pending queue size.
            # Pending queue size is just how big the prefill "write buffer" is.
            # It prevents prefills from going to fast and overrunning the scheduler.
            # On the other hand, the free slots here is how many requests we get to choose from in the scheduler.
            # If k = 1, then all schedulers converge to FCFS!
            # Total KV cache buffer needed is k * scheduler.batch_size * max_seq_len * dim * 2 * 4 bytes

            num_free_slots = self.num_scheduler_slots - self.llm.num_requests_in_progress

            dequeue_cors = [self.dequeue_request()
                            for _ in range(num_free_slots)]
            dequeue_cors_results = await asyncio.gather(*dequeue_cors, return_exceptions=True)
            requests_to_add = filter(lambda x: not isinstance(
                x, Empty), dequeue_cors_results)
            for request in requests_to_add:
                print(f"Decoder received request {request.request_id}")
                self.llm.add_request(request)
                # First token from prefill
                request.benchmark_metrics.received_token()

            # Do work
            done_requests, request_batch = self.llm.step_decode()

            # Update benchmarks
            torch.cuda.synchronize()
            for request in request_batch:
                request.benchmark_metrics.received_token()

            for request in done_requests:
                # NOTE: Important to block until queue is free
                print(f"Decoder finished request {request.request_id}")
                await self.output_queue.put_async(request)


@ray.remote(num_cpus=2)
class OutputConsumer:
    def __init__(self, config, input_queue: Queue):
        self.config = config
        self.input_queue = input_queue

        if self.config.benchmark_csv_path is not None:
            f = open(self.config.benchmark_csv_path, 'w')
            f.write('gen_len,JCT,TTFT,TPOT,TTFPT,TPODT\n')
            f.close()

    def run(self):
        while True:
            request = self.input_queue.get(block=True)
            print(f"OutputConsumer received request {request.request_id}")
            print(
                f"Max Gen Len: {request.max_gen_len}, Output: {request.output}")

            # Write benchmarks
            request.benchmark_metrics.finished_request()

            # Write benchmark results to file
            if self.config.benchmark_csv_path is not None:
                f = open(self.config.benchmark_csv_path, 'a')
                f.write(request.benchmark_metrics.to_csv_row())
                f.close()


@hydra.main(
    config_path="config/",
    config_name="disaggregated_llama_3",
    version_base=None,
)
def driver(config):
    num_available_gpus = torch.cuda.device_count()

    # Assert disabled for single GPU testing.
    assert (
        config.coordinator.num_prefill_workers
        + config.coordinator.num_decode_workers
        <= num_available_gpus
    )

    ray.init()

    # (Probably) optimal write buffer lengths?
    max_request_queue_size = config.coordinator.num_prefill_workers * \
        config.prefill_scheduler.batch_size
    max_pending_queue_size = config.coordinator.num_decode_workers * \
        config.decode_scheduler.batch_size

    request_queue = Queue(maxsize=max_request_queue_size)
    pending_queue = Queue(maxsize=max_pending_queue_size)
    result_queue = Queue()

    request_generator = ShareGPTRequestGenerator.remote(
        shareGPTPath, tokenizer_path, request_queue)
    prefillers = [
        Prefiller.options(name=f"prefiller#{i}").remote(
            config,
            request_queue,
            pending_queue,
        )
        for i in range(config.coordinator.num_prefill_workers)
    ]
    decoders = [
        Decoder.options(name=f"decoder#{i}").remote(
            config,
            pending_queue,
            result_queue,
        )
        for i in range(config.coordinator.num_decode_workers)
    ]
    output_consumer = OutputConsumer.remote(config, result_queue)

    # Wait for all actors to initialize
    ray.get([
        request_generator.load_corpus.remote(),
        *[prefiller.load_llm.remote() for prefiller in prefillers],
        *[decoder.load_llm.remote() for decoder in decoders],
    ])

    # Wait for all actors to terminate
    ray.get(
        [request_generator.run.remote()] +
        [prefiller.run.remote() for prefiller in prefillers] +
        [decoder.run.remote() for decoder in decoders] +
        [output_consumer.run.remote()]
    )


if __name__ == '__main__':
    driver()
