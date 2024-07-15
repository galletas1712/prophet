import os  # noqa
os.environ['RAY_DEDUP_LOGS'] = '0'  # noqa
os.environ['RAY_COLOR_PREFIX'] = '1'  # noqa
# Somehow we need to do this before importing Ray for no log dedup

from ray_workers.coordinator import Coordinator
from ray_workers.decode import Decoder
from ray_workers.prefill import Prefiller
from ray_workers.shareGPT import ShareGPTRequestGenerator
from ray_workers.output_consumer import OutputConsumer
from ray.util.queue import Queue
import ray
import hydra
import torch


@hydra.main(
    config_path="config/",
    config_name="disaggregated_llama_3",
    version_base=None,
)
def driver(config):
    torch.cuda.nvtx.range_push("Driver")
    num_available_gpus = torch.cuda.device_count()

    # Assert disabled for single GPU testing.
    assert (
        config.coordinator.num_prefill_workers
        + config.coordinator.num_decode_workers
        <= num_available_gpus
    )

    ray.init()

    # For each prefill and decode worker pair, we should have only one pending request
    max_pending_queue_size = config.coordinator.num_prefill_workers * \
        config.coordinator.num_decode_workers

    request_queue = Queue()
    pending_queue = Queue(maxsize=max_pending_queue_size)
    result_queue = Queue()

    # TODO: change SIGNIFICANTTLY
    world_size = config.coordinator.num_prefill_workers + config.coordinator.num_decode_workers
    ranks = list(range(world_size))
    rank_mapping = {}
    for i in range(0, config.coordinator.num_prefill_workers):
        rank_mapping[f"prefiller#{i}"] = i
    for i in range(0, config.coordinator.num_decode_workers):
        rank_mapping[f"decoder#{i}"] = i + config.coordinator.num_prefill_workers

    request_generator = ShareGPTRequestGenerator.remote(
        config.request_generator,
        config.model.tokenizer_path,
        request_queue
    )
    coordinator = Coordinator.remote(rank_mapping)
    prefillers = [
        Prefiller.options(name=f"prefiller#{i}", max_concurrency=2).remote(
            f"prefiller#{i}",
            config,
            request_queue,
            pending_queue,
        )
        for i in range(config.coordinator.num_prefill_workers)
    ]
    decoders = [
        Decoder.options(name=f"decoder#{i}", max_concurrency=2).remote(
            f"decoder#{i}",
            config,
            pending_queue,
            result_queue,
            coordinator
        )
        for i in range(config.coordinator.num_decode_workers)
    ]
    output_consumer = OutputConsumer.options(name=f"output_consumer").remote(
        config,
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        result_queue
    )

    # Wait for all actors to initialize
    ray.get([
        request_generator.load_corpus.remote(),
        *[prefiller.setup.remote(0, 2) for prefiller in prefillers], # TODO: assign ranks properly
        *[decoder.setup.remote(1, 2) for decoder in decoders],
    ])
    # TODO: sync k and v cache transfers?

    # Wait for all actors to terminate
    ray.get(
        [request_generator.run.remote()] +
        [prefiller.run.remote() for prefiller in prefillers] +
        [decoder.run.remote() for decoder in decoders] +
        [output_consumer.run.remote()]
    )
    torch.cuda.nvtx.range_pop()


if __name__ == '__main__':
    driver()

