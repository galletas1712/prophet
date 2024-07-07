
import ray
import os 

from models.llama3.tokenizer import LlamaFormatter, Tokenizer
from ray.util.queue import Queue

@ray.remote(num_cpus=2)
class OutputConsumer:

    def __init__(self, config, output_dir: str, input_queue: Queue):
        self.config = config
        self.input_queue = input_queue
        self.benchmark_results_file = os.path.join(
            output_dir, 'benchmark_results.csv')
        
        self.tokenizer = Tokenizer(config.model.tokenizer_path)
        self.formatter = LlamaFormatter(self.tokenizer)

        print("Started Output Consumer!")

    def run(self):
        while True:
            request = self.input_queue.get(block=True)
            request.output = self.formatter.decode_chat_completion(
                request.output_tokens,
                None,
            )
            print(f"Request prompt: {request.prompt}")
            print(
                f"Output tokens: {len(request.output_tokens)}, Output: {request.output}")