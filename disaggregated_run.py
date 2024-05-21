import hydra
from entrypoints.disaggregated_llm import Disaggregation_Coordinator
from entrypoints.api import CompletionType
from models.llama3.tokenizer import Dialog
from typing import List


@hydra.main(config_path="config/", config_name="dummy_test", version_base=None)
def run_model(config):
    llm = Disaggregation_Coordinator(
        config.coordinator, config.model, config.scheduler
    )

    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?",
            },
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
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?",
            },
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]

    for dialog in dialogs:
        llm.send_prompt(dialog, CompletionType.CHAT_COMPLETION)


if __name__ == "__main__":
    run_model()
