import hydra
from entrypoints import LLM


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

    requests = {}

    for prompt in prompts:
        request = llm.create_request(prompt)
        requests[request.request_id] = request

    outputs = []
    while len(outputs) < len(prompts):
        llm.step()
        for completed_request_ids in llm.curr_step_completed_request_ids:
            outputs.append(requests[completed_request_ids].output_str)

    print(f"Received outputs:")
    print(outputs)


if __name__ == "__main__":
    run_model()
