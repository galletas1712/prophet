import os
import torch
from multiprocessing import Process, Queue

from entrypoints.api import *
from schedulers import build_scheduler
from models import build_model

SHUTDOWN_PROMPT = "SHUTDOWN_WORKER()"


class DisaggregationCoordinator:
    def __init__(self, coordinator_config, model_config, scheduler_config):
        self.prefill_request_queue = Queue()
        self.decode_request_queue = Queue()
        self.output_queue = Queue()

        self.prefill_worker = Worker(
            model_config,
            scheduler_config,
            WorkerType.PREFILL,
            gpu_id=coordinator_config.prefill_gpu,
            master_port=coordinator_config.prefill_port,
        )
        self.decode_worker = Worker(
            model_config,
            scheduler_config,
            WorkerType.DECODE,
            gpu_id=coordinator_config.decode_gpu,
            master_port=coordinator_config.decode_port,
        )

        self.prefill_process = Process(
            target=self.prefill_worker.worker_main,
            args=(self.prefill_request_queue, self.decode_request_queue),
        )
        self.decode_process = Process(
            target=self.decode_worker.worker_main,
            args=(self.decode_request_queue, self.output_queue),
        )
        self.output_process = Process(target=self.monitor_output)

        self.prefill_process.start()
        self.decode_process.start()

    def send_prompt(self, prompt, completion_type):
        self.prefill_request_queue.put(prompt, completion_type)

    def monitor_output(self):
        while True:
            output = self.output_queue.get()
            if output["output"] == SHUTDOWN_PROMPT:
                break
            print(output)

    def shutdown(self):
        self.prefill_request_queue.put(SHUTDOWN_PROMPT, None)
        self.decode_request_queue.put(SHUTDOWN_PROMPT, None)
        self.output_queue.put({"output": SHUTDOWN_PROMPT})

        self.prefill_process.join()
        self.decode_process.join()


class Worker:

    def __init__(
        self, model_config, scheduler_config, worker_type, gpu_id, master_port
    ):
        self.gpu_id = gpu_id
        self.master_port = master_port
        self.model_config = model_config

        # TODO: abstract scheduler config based on worker_type
        self.scheduler = build_scheduler(scheduler_config)
        self.type = worker_type

        self.active = True
        self.request_id = 0

    def worker_main(self, in_queue, out_queue):
        # Set env variables for disaggregation. Currently assumes that both
        # workers use exactly one gpu.
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = f"{self.master_port}"

        os.environ["CUDA_AVAILABLE_DEVICES"] = f"{self.gpu_id}"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

        # Build model.
        self.model = build_model(self.model_config)

        self.cache_k = {}
        self.cache_v = {}
        if self.type == WorkerType.DECODE:
            self.decode_batch = DecodeDataBatch(
                self.model_config.max_batch_size,
                self.model_config.max_seq_len,
                self.model.model_args.n_layers,
                self.model.model_args.dim,
                self.model.tokenizer.pad_id,
            )

        # Set queues in object.
        self.in_queue = in_queue
        self.out_queue = out_queue

        # Main worker loop.
        while self.active:
            if len(self.in_queue) > 0:
                curr_prompt, curr_completion_type = self.in_queue.get()
                if curr_prompt == SHUTDOWN_PROMPT:
                    self.active = False
                    break
                curr_request = Request(
                    self.request_id, curr_prompt, curr_completion_type
                )
                curr_request.stage = (
                    RequestStage.PREFILL
                    if self.type == WorkerType.PREFILL
                    else RequestStage.DECODE
                )
                self.scheduler.add_request(curr_request)
                self.request_id += 1

            if len(self.in_queue) == 0:
                output = self.step()
                self.out_queue.put(output)

    def step(self):
        if self.type == WorkerType.PREFILL:
            return self.prefill_step()
        elif self.type == WorkerType.DECODE:
            return self.decode_step()

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
            if (
                slot_request is None
            ):  # NOTE: we clear to None to actually clear the slot
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
                    self.cache_v[scheduled_request.request_id],
                )
                curr_replaceable_slot_idx += 1

        self.model.step_decode(self.decode_batch)

        done_requests = []
        for slot_idx, slot_request in enumerate(self.decode_batch.requests):
            if (
                slot_request is not None
                and slot_request.stage is RequestStage.DONE
            ):
                # NOTE: slot_request MUST become None after this (set in DecodeDataBatch)
                done_requests.append(
                    {
                        "request_id": slot_request.request_id,
                        "output": slot_request.output,
                    }
                )
                self.scheduler.remove_request(slot_request.request_id)
                self.decode_batch.clear_slot(slot_idx)

        return done_requests
