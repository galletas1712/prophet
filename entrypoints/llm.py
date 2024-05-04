from entrypoints.api import *
from schedulers import build_scheduler
from models import build_model


class LLM:

    def __init__(self, scheduler_config, model_config) -> None:
        self.scheduler = build_scheduler(scheduler_config)
        self.model = build_model(model_config)

        self.request_id_to_response = {}
        self.completed_request_ids = []

    def create_request(self, prompt: str):
        return self.scheduler.create_request(prompt)

    def request_completed(self):
        return len(self.completed_request_ids) > 0

    def pop_completed_request_id(self):
        return self.completed_request_ids.pop()

    def prepare_batch(self, batch):
        prompts = []
        kv_caches = []

        for request in batch:
            prompts.append(request.full_text)
            kv_caches.append(request.kv_cache)

        return prompts, kv_caches

    def step(self):
        batch = self.scheduler.schedule()
        prompts, kv_caches = self.prepare_batch(batch)

        model_outputs = self.model.step(prompts, kv_caches)

        for idx, request in enumerate(batch):
            request.output = (
                request.output + model_outputs.new_tokens_decoded[idx]
            )
            # request.output_tokens.append(model_outputs.new_tokens[idx])

            request.full_text = (
                request.full_text + model_outputs.new_tokens_decoded[idx]
            )
            # request.all_tokens.append(model_outputs.new_tokens[idx])

            request.kv_cache = model_outputs.new_kv_caches[idx]

            if model_outputs.sequences_complete[idx]:
                self.completed_request_ids.append(request.request_id)
                self.scheduler.remove_request(request.request_id)
