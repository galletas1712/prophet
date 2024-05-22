from entrypoints.api import Request, RequestStage, PrefillDataBatch, DecodeDataBatch, CompletionType
from schedulers import build_scheduler
from models import build_model

from typing import Any, List, Optional
import random
import torch


class LLM:

    def __init__(self, model_config, scheduler_config, seed: int) -> None:
        assert scheduler_config.batch_size <= model_config.max_batch_size

        random.seed(seed)
        torch.manual_seed(seed)

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

        self.cache_k = {}
        self.cache_v = {}

    def create_request(self, prompt: str | Any, completion_type: CompletionType):
        return self.scheduler.create_request(prompt, completion_type)

    def step_prefill(self) -> Optional[PrefillDataBatch]:
        # Returns a PrefillDataBatch if there were requests to prefill, None otherwise
        request_batch = self.scheduler.schedule(RequestStage.PREFILL)
        # print("Prefilling requests:", request_batch)
        if len(request_batch) == 0:
            return None

        prefill_batch_state = self.model.step_prefill(request_batch)
        # NOTE: assumes idx is the same
        for idx, request in enumerate(request_batch):
            self.cache_k[request.request_id] = prefill_batch_state.cache_k[idx]
            self.cache_v[request.request_id] = prefill_batch_state.cache_v[idx]
        return prefill_batch_state

    def step_decode(self) -> List[int]:
        # Returns a list of request ids that terminated after this decode step

        #  Get set of slots we can replace
        free_slots, requests_already_in = self.decode_batch.get_free_slots()
        assert len(requests_already_in) + len(free_slots) == len(self.decode_batch.requests)

        request_batch = self.scheduler.schedule(RequestStage.DECODE)

        # print("Scheduled: ", [r.request_id for r in request_batch], "Replaceable slots: ", free_slots)
        # print("Requests already in:", requests_already_in)

        # Allocate free decode batch slots for new requests that just finished prefilling
        new_requests = filter(lambda r: r.request_id not in requests_already_in, request_batch)
        for free_slot_idx, new_request in zip(free_slots, new_requests):
            # print("Filling slot", free_slot_idx, "with request", new_request.request_id, new_request.prompt[:15])
            self.decode_batch.fill_slot(
                free_slot_idx,
                new_request,
                self.cache_k[new_request.request_id],
                self.cache_v[new_request.request_id],
            )

        self.model.step_decode(self.decode_batch)

        done_requests = []
        for slot_idx, slot_request in enumerate(self.decode_batch.requests):
            if slot_request is not None and slot_request.stage is RequestStage.DONE:
                # NOTE: slot_request MUST become None after this (set in DecodeDataBatch)
                done_requests.append(slot_request.request_id)
                self.scheduler.remove_request(slot_request.request_id)
                self.decode_batch.clear_slot(slot_idx)

        return done_requests