import hydra

from models import get_model_constructor
from schedulers import create_scheduler
from entrypoints import create_llm


@hydra.main(
    config_path="config/", config_name="llama_3_test", version_base=None
)
def run_model(config):
    model_constructor = get_model_constructor(config.model)
    scheduler = create_scheduler(config.scheduler)
    llm = create_llm(scheduler, model_constructor)

    requests = [
        "test prompt 1",
        "test prompt 2",
        "test prompt 3",
    ]


if __name__ == "__main__":
    run_model()
