from entrypoints.api import *
from schedulers import build_scheduler
from models import build_model


class LLM:

    def __init__(self, model_config, scheduler_config) -> None:
        self.model = build_model(model_config)

        self.scheduler = build_scheduler(
            scheduler_config, kv_cache_shape=self.model.kv_cache_shape()
        )

        self.request_id_to_response = {}
        self.curr_step_completed_request_ids = []

    def create_request(self, prompt: str):
        return self.scheduler.create_request(prompt)

    def step(self):
        self.curr_step_completed_request_ids = []

        batch = self.scheduler.schedule()
        self.model.step(batch)

        for _, request in enumerate(batch):
            if request.stage is RequestStage.DONE:
                self.curr_step_completed_request_ids.append(request.request_id)
                self.scheduler.remove_request(request.request_id)
