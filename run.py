import hydra
from entrypoints import LLM
from entrypoints.api import CompletionType
from models.llama3.tokenizer import Dialog
from typing import List


@hydra.main(config_path="config/", config_name="llama_3_test", version_base=None)
def run_model(config):
    llm = LLM(config.model, config.scheduler, config.seed, 'cuda:0')

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

    dialogs: List[Dialog] = [
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

    requests = {}
    total_requests = 0

    for prompt in prompts:
        request = llm.create_request(prompt, CompletionType.TEXT_COMPLETION)
        requests[request.request_id] = request
        total_requests += 1

    for dialog in dialogs:
        request = llm.create_request(dialog, CompletionType.CHAT_COMPLETION)
        requests[request.request_id] = request
        total_requests += 1
    
    outputs = []

    llm.step_prefill()
    while len(outputs) < total_requests:
        # for request in llm.decode_batch.requests:
        #     if request is not None:
        #         print(request.request_id)
        #         print(request.stage)
        #         print(llm.model.tokenizer.decode(request.output_tokens))
        #         print("--------------------------------------------")
        # print("===================================================")
        done_requests = llm.step_decode()
        for completed_request_id in done_requests:
            print("Completed request", completed_request_id)
            print(requests[completed_request_id].output)
            outputs.append(requests[completed_request_id].output)
        free_slots, _ = llm.decode_batch.get_free_slots()
        if len(free_slots) > 0:
            llm.step_prefill()
    
    print(f"Received outputs:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    run_model()
