# Somehow we need to do this before importing Ray for no log dedup
import os
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['RAY_COLOR_PREFIX'] = '1'

import torch
import hydra
import ray
from ray.util.queue import Queue
from ray_workers.shareGPT import ShareGPTRequestGenerator
from ray_workers.prefill import Prefiller
from ray_workers.decode import Decoder


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
    max_pending_queue_size = config.coordinator.num_decode_workers * \
        config.decode_scheduler.batch_size

    request_queue = Queue()
    pending_queue = Queue(maxsize=max_pending_queue_size)
    result_queue = Queue()

    request_generator = ShareGPTRequestGenerator.remote(
        config.request_generator_corpus_path,
        config.model.tokenizer_path,
        request_queue
    )
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
