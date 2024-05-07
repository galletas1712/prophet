import hydra
from entrypoints import LLM


@hydra.main(
    config_path="config/", config_name="dummy_test", version_base=None
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
        for completed_request_ids in llm.curr_step_completed_request_ids:
            outputs.append(requests[completed_request_ids].output)

    print(f"Received outputs:")
    print(outputs)


if __name__ == "__main__":
    run_model()
