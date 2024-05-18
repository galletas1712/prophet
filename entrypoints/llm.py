from entrypoints.api import Request, RequestStage, PrefillDataBatch, DecodeDataBatch, CompletionType
from schedulers import build_scheduler
from models import build_model

from typing import Any


class LLM:

    def __init__(self, model_config, scheduler_config) -> None:
        self.model = build_model(model_config)

        self.scheduler = build_scheduler(
            scheduler_config
        )

        self.decode_batch = DecodeDataBatch(
            model_config.max_batch_size,
            model_config.max_seq_len,
            self.model.model_args.n_layers,
            self.model.model_args.dim,
            self.model.tokenizer.pad_id
        )

        self.done_requests = []
        self.cache_k = {}
        self.cache_v = {}

    def create_request(self, prompt: str | Any, completion_type: CompletionType):
        return self.scheduler.create_request(prompt, completion_type)

    def step_prefill(self):
        request_batch = self.scheduler.schedule(RequestStage.PREFILL)
        prefill_batch_state = self.model.step_prefill(request_batch)
        # NOTE: assumes idx is the same
        for idx, request in enumerate(request_batch):
            self.cache_k[request.request_id] = prefill_batch_state.cache_k[idx]
            self.cache_v[request.request_id] = prefill_batch_state.cache_v[idx]
        return prefill_batch_state

    def step_decode(self):
        #  Get set of slots we can replace
        replaceable_slots = []
        requests_already_in = set()
        for slot_idx, slot_request in enumerate(self.decode_batch.requests):
            if slot_request is None:  # NOTE: we clear to None to actually clear the slot
                replaceable_slots.append(slot_idx)
            else:
                requests_already_in.add(slot_request.request_id)

        # TODO: move to the actual decode batch class, and use a map instead
        request_batch = self.scheduler.schedule(RequestStage.DECODE)
        curr_replaceable_slot_idx = 0
        for scheduled_request in request_batch:
            if scheduled_request.request_id not in requests_already_in:
                self.decode_batch.fill_slot(
                    curr_replaceable_slot_idx,
                    scheduled_request,
                    self.cache_k[scheduled_request.request_id],
                    self.cache_v[scheduled_request.request_id]
                )
                curr_replaceable_slot_idx += 1

        self.model.step_decode(self.decode_batch)
        for slot_idx, slot_request in enumerate(self.decode_batch.requests):
            if slot_request is not None and slot_request.stage is RequestStage.DONE:
                self.done_requests.append(slot_request.request_id)
                self.scheduler.remove_request(slot_request.request_id)
                self.decode_batch.clear_slot(slot_idx)
