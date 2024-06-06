import os  # noqa
os.environ['RAY_DEDUP_LOGS'] = '0'  # noqa
os.environ['RAY_COLOR_PREFIX'] = '1'  # noqa
# Somehow we need to do this before importing Ray for no log dedup

from models.llama3.tokenizer import LlamaFormatter, Tokenizer
from ray_workers.decode import Decoder
from ray_workers.prefill import Prefiller
from ray_workers.shareGPT import ShareGPTRequestGenerator
from ray_workers.consumer_test import ConsumerTest
from ray.util.queue import Queue
import ray
import ray.util.collective as collective
import hydra
import torch


@ray.remote(num_cpus=2)
class OutputConsumer:

    def __init__(self, config, estimate_decode_lengths: bool, output_dir: str, input_queue: Queue):
        self.config = config
        self.estimate_decode_lengths = estimate_decode_lengths
        self.input_queue = input_queue
        self.benchmark_results_file = os.path.join(
            output_dir, 'benchmark_results.csv')
        
        print("Estimate decode lengths:", self.estimate_decode_lengths  )

        self.tokenizer = Tokenizer(config.model.tokenizer_path)
        self.formatter = LlamaFormatter(self.tokenizer)

        f = open(self.benchmark_results_file, 'w')
        if self.estimate_decode_lengths:
            f.write('request_hash,gen_len,JCT,TTFT,TPOT,TTFPT,TPODT,estimated_output_len,actual_output_len\n')
        else:
            f.write('request_hash,gen_len,JCT,TTFT,TPOT,TTFPT,TPODT\n')
        f.close()

        print("Started Output Consumer!")

    def run(self):
        while True:
            request = self.input_queue.get(block=True)
            request.output = self.formatter.decode_chat_completion(
                request.output_tokens,
                self.estimate_decode_lengths,
                None,
            )
            print(f"OutputConsumer received request {request.request_id}")
            # Remove length prediction prefix from output
            
            if self.estimate_decode_lengths:
                print(
                    f"Estimated Output Tokens: {request.estimated_token_length}, Actual Output Tokens: {len(request.output_tokens)}, Output: {request.output}")
            else:
                print(
                    f"Output tokens: {len(request.output_tokens)}, Output: {request.output}")


            # Write benchmarks
            request.benchmark_metrics.finished_request()
            if self.estimate_decode_lengths:
                csv_row = ','.join(
                    [str(request.request_id)] +
                    request.benchmark_metrics.get_stats_list() +
                    [str(request.estimated_token_length), str(len(request.output_tokens))]
                ) + '\n'
            else:
                csv_row = ','.join(
                    [str(request.request_id)] +
                    request.benchmark_metrics.get_stats_list()
                ) + '\n'


            # Write benchmark results to file
            f = open(self.benchmark_results_file, 'a')
            f.write(csv_row)
            f.close()

@ray.remote
class Coordinator:
    def __init__(self, rank_mapping):
        self.rank_mapping = rank_mapping

    def send_tensor(self, sender_name, receiver_name, request_id):
        sender = ray.get_actor(sender_name)
        receiver = ray.get_actor(receiver_name)
        ray.get([
            sender.send_kv.remote(request_id=request_id, target_rank=self.rank_mapping[receiver_name]),
            receiver.receive_kv.remote(src_rank=self.rank_mapping[sender_name])])


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

    # For each prefill and decode worker pair, we should have only one pending request
    max_pending_queue_size = config.coordinator.num_prefill_workers * \
        config.coordinator.num_decode_workers

    request_queue = Queue()
    pending_queue = Queue(maxsize=max_pending_queue_size)
    result_queue = Queue()

    estimate_decode_lengths = hasattr(config.decode_scheduler, 'scoring_method') and config.decode_scheduler.scoring_method == 'estimated_rpt'

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
        estimate_decode_lengths,
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
        estimate_decode_lengths,
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        result_queue
    )

    # Create the collective group
    collective.create_collective_group(
        [*prefillers, *decoders],
        world_size=world_size,
        ranks=ranks
    )

    # Wait for all actors to initialize
    ray.get([
        request_generator.load_corpus.remote(),
        *[prefiller.setup.remote() for prefiller in prefillers],
        *[decoder.setup.remote() for decoder in decoders],
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
