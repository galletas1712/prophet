import random
import hydra
import torch
import torch.multiprocessing as mp

from queue import Empty

from entrypoints.api import CompletionType, Request, WorkerType
from entrypoints.llm import LLM

def sample_requests():
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:
        Hi everyone,
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]

    dialogs = [
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
    Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

    1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
    2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
    3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

    These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
    ]

    requests = [Request(prompt, CompletionType.TEXT_COMPLETION) for prompt in prompts] + [Request(dialog, CompletionType.CHAT_COMPLETION) for dialog in dialogs]
    random.shuffle(requests)
    return requests

def prefill_worker(
    config,
    device: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    torch.cuda.set_device(device)
    llm = LLM(config.model, config.scheduler, config.seed, worker_type=WorkerType.PREFILL)
    num_pending_batching = 0
    while True:
        request = input_queue.get()
        # Terminate on None
        if request is None:
            return
        llm.add_request(request)
        num_pending_batching += 1
        if num_pending_batching == 1: # TODO: CHANGE BATCHING SIZE (also check with scheduler batching size and max_batch_size) and timeout as well?
            # NOTE: no batching for now
            prefill_batch_state = llm.step_prefill()
            for request in prefill_batch_state.requests:
                output_queue.put(request)
            num_pending_batching = 0

def decode_worker(
    config,
    device: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    torch.cuda.set_device(device)
    llm = LLM(config.model, config.scheduler, config.seed, worker_type=WorkerType.DECODE)
    while True:
        try:
            request = input_queue.get_nowait()
            # Terminate on None
            if request is None:
                return
            llm.add_request(request)
        except Empty:
            # OK if no new requests, continue decoding old requests
            pass

        done_requests = llm.step_decode()
        for request_id, output in done_requests.items():
            output_queue.put((request_id, output))


@hydra.main(config_path="config/", config_name="llama_3_test", version_base=None)
def spawn_workers(config):
    mp.set_start_method("spawn")
    num_available_gpus = torch.cuda.device_count()
    assert config.coordinator.num_prefill_workers + config.coordinator.num_decode_workers <= num_available_gpus

    prefill_queue = mp.Queue()
    decode_queue = mp.Queue()
    result_queue = mp.Queue()

    processes = [
        mp.Process(
            target=prefill_worker,
            args=(config, f'cuda:{i}', prefill_queue, decode_queue),
        ) for i in range(config.coordinator.num_prefill_workers)
    ] + [
        mp.Process(
            target=decode_worker,
            args=(config, f'cuda:{config.coordinator.num_prefill_workers + i}', decode_queue, result_queue),
        ) for i in range(config.coordinator.num_decode_workers)
    ] # NOTE: using same config bretween prefill and decode workers for now

    for p in processes:
        p.start()

    # Send requests to workers
    requests = sample_requests()
    for request in requests:
        prefill_queue.put(request)

    for _ in range(len(requests)):
        request_id, output = result_queue.get()
        print("Received result for request", request_id)
        print(output)
    
    # Send terminate signal
    for _ in range(config.coordinator.num_prefill_workers):
        prefill_queue.put(None)
    for _ in range(config.coordinator.num_decode_workers):
        decode_queue.put(None)

    for p in processes:
        p.join()


if __name__ == '__main__':
    spawn_workers()