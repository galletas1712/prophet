from entrypoints.api import Request, RequestStage, WorkerType
from entrypoints.databatch import PrefillDataBatch, DecodeDataBatch
from schedulers import build_scheduler
from models import build_model

from typing import Any, Optional
import random
import torch


class LLM:
    def __init__(
        self,
        model_config,
        scheduler_config,
        seed: int,
        # NOTE: None corresponds to both
        worker_type: Optional[WorkerType] = None,
    ) -> None:
        random.seed(seed)
        torch.manual_seed(seed)

        assert scheduler_config.batch_size <= model_config.max_batch_size

        self.worker_type = worker_type
        self.model = build_model(model_config)
        self.scheduler = build_scheduler(
            scheduler_config
        )
        if self.worker_type is not WorkerType.PREFILL:
            self.decode_batch = DecodeDataBatch(
                scheduler_config.batch_size,
                model_config.max_seq_len,
                self.model.model_args.n_layers,
                self.model.model_args.dim,
                self.model.tokenizer.pad_id
            )

        self.num_requests_in_progress = 0

    def step_prefill(self) -> Optional[PrefillDataBatch]:
        # NOTE: sometimes we might call step_prefill with nothing in the queue
        # Returns a PrefillDataBatch if there were requests to prefill, None otherwise
        assert self.worker_type is not WorkerType.DECODE

        request_batch = self.scheduler.schedule(RequestStage.PREFILL)
        # print("Prefilling requests:", request_batch)
        if len(request_batch) == 0:
            return None

        prefill_batch_state = self.model.step_prefill(request_batch)
        self.num_requests_in_progress -= len(request_batch)
        return prefill_batch_state

    def step_decode(self) -> dict[str, Any]:
        # Returns a list of request ids that terminated after this decode step
        assert self.worker_type is not WorkerType.PREFILL

        #  Get set of slots we can replace
        free_slots = self.decode_batch.get_free_slots()
        requests_already_in = self.decode_batch.get_requests_already_in()
        assert len(requests_already_in) + \
            len(free_slots) == len(self.decode_batch.requests)

        request_batch = self.scheduler.schedule(RequestStage.DECODE)
        # print("Scheduling decodes with prompt lengths", [(req.prompt, len(req.prompt)) for req in request_batch])

        # If there's nothing to process
        if len(request_batch) == 0 and len(requests_already_in) == 0:
            return {}, []

        # Allocate free decode batch slots for new requests that just finished prefilling
        new_requests = list(filter(
            lambda r: r.request_id not in requests_already_in, request_batch))

        # Preempt slots if we have to
        occupied_slots = self.decode_batch.get_occupied_slots()
        requests_to_preempt_with = new_requests[len(free_slots):]
        for occupied_slot_idx, new_request in zip(occupied_slots, requests_to_preempt_with):
            print(
                f"Preempting slot {occupied_slot_idx} kicking out {self.decode_batch.requests[occupied_slot_idx].request_id} for request {new_request.request_id}")
            self.decode_batch.preempt_slot(occupied_slot_idx, new_request)

        # Fill slots that can be filled
        # NOTE: This MUST come after preemption, because our occupied set changes
        requests_to_fill_with = new_requests[:len(free_slots)]
        for free_slot_idx, new_request in zip(free_slots, requests_to_fill_with):
            print(
                f"Filling slot {free_slot_idx} with request {new_request.request_id}")
            self.decode_batch.fill_slot(free_slot_idx, new_request)

        self.model.step_decode(self.decode_batch)

        # Process all requests that are done
        done_requests = []
        for slot_idx, slot_request in enumerate(self.decode_batch.requests):
            if slot_request is not None and slot_request.stage is RequestStage.DONE:
                # NOTE: Important otherwise we get memory leak
                slot_request.free_cache()

                # NOTE: slot_request MUST become None after this (set in DecodeDataBatch)
                self.scheduler.remove_request(slot_request.request_id)
                self.num_requests_in_progress -= 1

                self.decode_batch.clear_slot(slot_idx)
                done_requests.append(slot_request)

        return done_requests, request_batch

    def add_request(self, request: Request):
        self.scheduler.add_request(request)
        self.num_requests_in_progress += 1

    def get_num_free_decoder_slots(self):
        return len(self.decode_batch.free_slots)
