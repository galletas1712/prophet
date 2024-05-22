from entrypoints.api import Request, RequestStage, PrefillDataBatch, DecodeDataBatch, CompletionType, WorkerType
from schedulers import build_scheduler
from models import build_model

from typing import Any, List, Optional
import random
import torch


class LLM:
    def __init__(
        self,
        model_config,
        scheduler_config,
        seed: int,
        device: Optional[int] = None,
        worker_type: Optional[WorkerType] = None
    ) -> None:
        random.seed(seed)
        torch.manual_seed(seed)

        if device is not None:
            print("Loading model on device:", device)
            torch.cuda.set_device(device)
        else:
            print("Loading model on device:", torch.cuda.current_device())

        assert scheduler_config.batch_size <= model_config.max_batch_size

        self.worker_type = worker_type
        self.model = build_model(model_config)
        self.scheduler = build_scheduler(
            scheduler_config
        )
        if self.worker_type is not WorkerType.PREFILL:
            self.decode_batch = DecodeDataBatch(
                model_config.max_batch_size,
                model_config.max_seq_len,
                self.model.model_args.n_layers,
                self.model.model_args.dim,
                self.model.tokenizer.pad_id
            )

    def step_prefill(self) -> Optional[PrefillDataBatch]:
        # NOTE: sometimes we might call step_prefill with nothing in the queue
        # Returns a PrefillDataBatch if there were requests to prefill, None otherwise
        assert self.worker_type is not WorkerType.DECODE

        request_batch = self.scheduler.schedule(RequestStage.PREFILL)
        # print("Prefilling requests:", request_batch)
        if len(request_batch) == 0:
            return None

        prefill_batch_state = self.model.step_prefill(request_batch)
        return prefill_batch_state

    def step_decode(self) -> List[int]:
        # Returns a list of request ids that terminated after this decode step
        assert self.worker_type is not WorkerType.PREFILL

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
            self.decode_batch.fill_slot(free_slot_idx, new_request)

        self.model.step_decode(self.decode_batch)

        done_requests = []
        for slot_idx, slot_request in enumerate(self.decode_batch.requests):
            if slot_request is not None and slot_request.stage is RequestStage.DONE:
                # NOTE: slot_request MUST become None after this (set in DecodeDataBatch)
                done_requests.append(slot_request.request_id)
                self.scheduler.remove_request(slot_request.request_id)
                self.decode_batch.clear_slot(slot_idx)

        return done_requests

    def add_request(self, request: Request):
        self.scheduler.add_request(request)
