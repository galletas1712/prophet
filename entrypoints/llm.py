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
    
    def get_scheduler(self) -> Any:
        return self.scheduler

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

        request_batch = self.scheduler.schedule(RequestStage.DECODE)
        # print("Scheduling decodes with prompt lengths", [(req.prompt, len(req.prompt)) for req in request_batch])

        # If there's nothing to process
        if len(request_batch) == 0:
            return {}, []
        
        # Strategy: never preempt if request is already in batch, fill free slots first, then preempt
        new_requests = list(filter(lambda r: r.idx_in_data_batch is None, request_batch))
        old_requests = list(filter(lambda r: r.idx_in_data_batch is not None, request_batch))
        # print(f"Request batch: {[(r.request_id, r.idx_in_data_batch) for r in request_batch]}")
        # print(f"New requests: {[(r.request_id, r.idx_in_data_batch) for r in new_requests]}")
        # print(f"Old requests: {[(r.request_id, r.idx_in_data_batch) for r in old_requests]}")
        if len(new_requests) > 0:
            free_slots = self.decode_batch.get_free_slots()
            preempt_slots = self.decode_batch.get_occupied_slots_avoiding_requests([r.request_id for r in old_requests])
            # print("Free slots:", free_slots)
            # print("Preempt slots:", preempt_slots)
            # TODO: strategy should be to round-robin preemptions for fairness. Keep a LRU dequeue and pop from the back.

            slots = (free_slots + preempt_slots)[:len(new_requests)]
            # print(f"Filling/preempting slots {slots}")
            self.decode_batch.batch_preempt_slots(slots, new_requests)

        self.model.step_decode(self.decode_batch)

        # Process all requests that are done
        done_requests = []
        for slot_idx, slot_request in enumerate(self.decode_batch.requests):
            if slot_request is not None and slot_request.stage is RequestStage.DONE:
                # NOTE: slot_request MUST become None after this (set in DecodeDataBatch)
                slot_request.free_cache()
                self.scheduler.remove_request(slot_request)
                self.num_requests_in_progress -= 1

                self.decode_batch.clear_slot(slot_idx)
                done_requests.append(slot_request)

        return done_requests, request_batch

    def add_request(self, request: Request):
        self.scheduler.add_request(request)
        self.num_requests_in_progress += 1

    def get_num_free_decoder_slots(self):
        return len(self.decode_batch.free_slots)
