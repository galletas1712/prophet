import hydra
from entrypoints import LLM
from entrypoints.api import CompletionType
from models.llama3.tokenizer import Dialog
from typing import List


@hydra.main(config_path="config/", config_name="dummy_test", version_base=None)
def run_model(config):
    llm = LLM(config.model, config.scheduler)

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
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]

    requests = {}

    for prompt in prompts:
        request = llm.create_request(prompt, CompletionType.TEXT_COMPLETION)
        requests[request.request_id] = request

    # for dialog in dialogs:
    #     request = llm.create_request(dialog, CompletionType.CHAT_COMPLETION)
    #     requests[request.request_id] = request

    llm.step_prefill()

    outputs = []
    while len(outputs) < len(prompts):
        llm.step_decode()
        for completed_request_ids in llm.done_requests:
            outputs.append(requests[completed_request_ids].output)

    print(f"Received outputs:")
    print(outputs)


if __name__ == "__main__":
    run_model()
