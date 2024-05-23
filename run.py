import hydra
from typing import List

import torch

from entrypoints.llm import LLM
from entrypoints.api import CompletionType, Request
from test_data import TEST_PROMPTS, TEST_DIALOGS


@hydra.main(
    config_path="config/", config_name="llama_3_test", version_base=None
)
def run_model(config):
    torch.cuda.set_device("cuda:0")

    llm = LLM(config.model, config.scheduler, config.seed, "cuda:0")

    total_requests = 0

    for prompt in TEST_PROMPTS:
        request = Request(prompt, CompletionType.TEXT_COMPLETION)
        llm.add_request(request)
        total_requests += 1

    for dialog in TEST_DIALOGS:
        request = Request(dialog, CompletionType.CHAT_COMPLETION)
        llm.add_request(request)
        total_requests += 1

    outputs = []

    llm.step_prefill()
    
    while len(outputs) < total_requests:
        done_requests = llm.step_decode()
        for request_id, output in done_requests.items():
            print(f"---- COMPLETED REQUEST {request_id} ----")
            print(output)
            print()
            outputs.append(output)
        
        if len(llm.decode_batch.get_free_slots()) > 0:
            llm.step_prefill()

if __name__ == "__main__":
    run_model()
