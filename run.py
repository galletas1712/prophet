import hydra
from entrypoints import LLM


@hydra.main(
    config_path="config/", config_name="llama_3_test", version_base=None
)
def run_model(config):
    llm = LLM(config.scheduler, config.model)

    prompts = [
        "test prompt 1",
        "test prompt 2",
        "test prompt 3",
    ]

    requests = {}

    for prompt in prompts:
        request = llm.create_request(prompt)
        requests[request.request_id] = request

    outputs = []
    while len(outputs) < len(prompts):
        llm.step()
        if llm.request_completed():
            completed_request_id = llm.pop_completed_request_id()
            outputs.append(requests[completed_request_id].output)

    print(f"Received outputs:")
    print(outputs)


if __name__ == "__main__":
    run_model()
