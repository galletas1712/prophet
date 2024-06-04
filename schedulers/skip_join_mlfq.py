from typing import List
from dataclasses import dataclass

from entrypoints.api import Request, RequestStage
from schedulers.utils import register_scheduler
import logging

@dataclass
class MLFQRequestInfo:
    request: Request
    iteration_number: int
    last_batched_time: int

@register_scheduler("skip_join_mlfq")
class SkipJoinMLFQ_Scheduler:
    def __init__(
        self,
        batch_size,
        num_queues,
        queue_limits,
        starvation_limit,
        **kwargs,
    ) -> None:
        super(SkipJoinMLFQ_Scheduler, self).__init__()
        self.batch_size = batch_size
        self.starvation_limit = starvation_limit
        self.now = 0

        self.request_queues = [[] for _ in range(num_queues)]
        self.num_queues = num_queues
        self.queue_limits = queue_limits
        assert len(queue_limits) == num_queues

        self.logger = logging.getLogger("skip_join_mlfq")
        logging.basicConfig(level=logging.DEBUG)


    def add_request(self, request, auto_score=None) -> Request:
        # skip-join step
        # priority is set to the quantum larger than the first iteration quantum
        request_added = False
        score = len(request.prompt) if not auto_score else auto_score
        for i, limit in enumerate(self.queue_limits):
            if score <= limit:
                self.request_queues[i].append(
                    MLFQRequestInfo(request, 0, self.now)
                )
                request_added = True
                break
        if not request_added:
            self.request_queues[-1].append(
                MLFQRequestInfo(request, 0, self.now)
            )

        return request
    
    def schedule(self, stage: RequestStage) -> List[Request]:
        self.now += 1
        batch = []
        batch_done = False
        queue_idx = 0
        for queue_idx in range(self.num_queues):
            if batch_done:
                break
            req_idx = 0
            while req_idx < len(self.request_queues[queue_idx]):
                request_info = self.request_queues[queue_idx][req_idx]
                assert request_info.request.stage != RequestStage.DONE
                if request_info.request.stage != stage:
                    req_idx += 1
                    continue

                # preemption
                if request_info.iteration_number != 0 and request_info.iteration_number % self.queue_limits[queue_idx] == 0:
                    self._move_request_to_lower_queue(queue_idx, req_idx)
                    continue

                request_info.last_batched_time = self.now
                request_info.iteration_number += 1
                self.request_queues[queue_idx][req_idx] = request_info

                batch.append(request_info.request)

                if len(batch) == self.batch_size:
                    batch_done = True
                    break
                
                req_idx += 1

        # prevent starvation
        for queue_idx, queue in enumerate(self.request_queues):
            for req_idx, request_info in enumerate(queue):
                if (self.now - request_info.last_batched_time) >= self.starvation_limit:
                    self._reset_request(queue_idx, req_idx)
                    break
        return batch

    def _move_request_to_lower_queue(self, queue_idx, req_idx):
        request_info = self.request_queues[queue_idx].pop(req_idx)
        if queue_idx == self.num_queues - 1:
            self.request_queues[queue_idx].append(request_info)
        else:
            self.request_queues[queue_idx + 1].append(request_info)

    def _reset_request(self, queue_idx, req_idx):
        request_info = self.request_queues[queue_idx].pop(req_idx)
        assert (self.now - request_info.last_batched_time) >= self.starvation_limit
        request_info.last_batched_time = self.now
        request_info.iteration_number = 0
        self.request_queues[0].append(request_info)

    def remove_request(self, request: Request):
        for i in range(self.num_queues):
            for req_idx, request_info in enumerate(self.request_queues[i]):
                if request_info.request.request_id == request.request_id:
                    self.request_queues[i].pop(req_idx)
                    return
        raise ValueError(
            f"Request with id {request.request_id} not found in any queue"
        )