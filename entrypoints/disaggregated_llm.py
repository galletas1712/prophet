from multiprocessing import Process, Queue
from entrypoints.api import *

from schedulers import build_scheduler
from models import build_model

class Coordinator:
  def __init__ (self, model_config, scheduler_config):
     self.prefill_request_queue = Queue ()
     self.decode_request_queue = Queue ()


     self.prefill_worker = Worker (model_config, scheduler_config, WorkerType.PREFILL)
     self.decode_worker = Worker (model_config, scheduler_config, WorkerType.DECODE)
     
     self.prefill_process = Process (target= self.prefill_worker.worker_main, args = (self.prefill_request_queue))
     self.decode_process = Process (target= self.decode_worker.worker_main, args = (self.decode_request_queue))


class Worker:
    
    def __init__(self, model_config, scheduler_config, worker_type):
        # TODO: abstract scheduler config based on worker_type
        self.scheduler = build_scheduler(
            scheduler_config
        )
        self.type = worker_type
        self.model = build_model(model_config)

        self.active = True

        self.request_id = 0

    def worker_main(self, queue):
        self.queue = queue
        while self.active ()
            if len(self.queue) > 0:
                curr_prompt = self.queue.pop()
                curr_request = Request(self.request_id, curr_prompt, self.type)
                self.scheduler.add_request(curr_request)
                self.request_id += 1
            
            if len(self.queue) == 0:
                self.step()
          
          
    def step (self):
       if (self.type == WorkerType.PREFILL):
          self.prefill_step ()
       elif (self.type == WorkerType.DECODE):
          self.decode_step ()
    
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

        done_requests = []
        for slot_idx, slot_request in enumerate(self.decode_batch.requests):
            if slot_request is not None and slot_request.stage is RequestStage.DONE:
                # NOTE: slot_request MUST become None after this (set in DecodeDataBatch)
                done_requests.append(slot_request.request_id)
                self.scheduler.remove_request(slot_request.request_id)
                self.decode_batch.clear_slot(slot_idx)

        return done_requests