from typing import List

from entrypoints.api import Request, RequestStage
from schedulers.utils import register_scheduler
import logging

@register_scheduler("skip_join_mlfq")
class SkipJoinMLFQ_scheduler:
    def __init__(
        self,
        batch_size,
        num_queues,
        queue_limits,
        starvation_limit,
        **kwargs,
    ) -> None:
        super(SkipJoinMLFQ_scheduler, self).__init__()
        self.batch_size = batch_size
        self.num_queues = num_queues
        self.queue_limits = queue_limits
        assert len(queue_limits) == num_queues
        self.starvation_limit = starvation_limit

        self.request_queues = [[] for _ in range(num_queues)]

        self.logger = logging.getLogger("skip_join_mlfq")
        logging.basicConfig(level=logging.DEBUG)


    def add_request(self, request) -> Request:
        # skip-join step
        # priority is set to the quantum larger than the first iteration quantum
        request_added = False
        for i, limit in enumerate(self.queue_limits):
            if len(request.prompt) <= limit:
                self.request_queues[i].append(
                    (request, 0)
                )  # request, iteration_number
                request_added = True
                break
        if not request_added:
            self.request_queues[-1].append((request, 0))
        return request
    
    # TODO(cathy) this is not tested!!
    def schedule(self, stage: RequestStage) -> List[Request]:
        batch = []
        batch_done = False
        queue_idx = 0
        for queue_idx in range(self.num_queues):
            if batch_done:
                break
            req_idx = 0
            while req_idx < len(self.request_queues[queue_idx]):
                request, iteration_number = self.request_queues[queue_idx][req_idx]
                assert request.stage != RequestStage.DONE
                if request.stage != stage:
                    req_idx += 1
                    continue

                # preemption
                if iteration_number != 0 and iteration_number % self.queue_limits[queue_idx] == 0:
                    # print("PREEMPT BEFORE")
                    # for i, q in enumerate(self.request_queues):
                    #     print(f"Queue {i}: {[r[0].request_id for r in q]}")
                        # print(f"Queue {i}: {len(q)}")

                    self.logger.info(f'PREEMPT')

                   
                    self._move_request_to_lower_queue(queue_idx, req_idx)
                    
                    # print("PREEMPT AFTER", req_idx)
                    # for i, q in enumerate(self.request_queues):
                    #     print(f"Queue {i}: {[r[0].request_id for r in q]}")
                        # print(f"Queue {i}: {len(q)}")

                    continue

                self.request_queues[queue_idx][req_idx] = (
                    request,
                    iteration_number + 1,
                )
                batch.append(request)

                # print("scheduled:", request.request_id, "queue:", queue_idx, "iteration:", iteration_number)

                if len(batch) == self.batch_size:
                    batch_done = True
                    break
                
                req_idx += 1

        # prevent starvation
        for queue_idx, queue in enumerate(self.request_queues):
            for req_idx, (request, iteration_number) in enumerate(queue):
                if iteration_number >= self.starvation_limit:
                    self._reset_request(queue_idx, req_idx)
                    break

        return batch

    def _move_request_to_lower_queue(self, queue_idx, req_idx):
        if queue_idx == self.num_queues - 1:
            request, iteration_number = self.request_queues[queue_idx].pop(req_idx)
            self.request_queues[queue_idx].append((request, iteration_number+1))
        else:
            request, iteration_number = self.request_queues[queue_idx].pop(req_idx)
            self.request_queues[queue_idx + 1].append((request, iteration_number))

    def _reset_request(self, queue_idx, req_idx):
        request, iteration_number = self.request_queues[queue_idx].pop(req_idx)
        assert iteration_number >= self.starvation_limit
        self.request_queues[0].append((request, 0))

    def remove_request(self, finished_request_id):
        for i in range(self.num_queues):
            for req_idx, (request, _) in enumerate(self.request_queues[i]):
                if request.request_id == finished_request_id:
                    self.request_queues[i].pop(req_idx)
                    return
        raise ValueError(
            f"Request with id {finished_request_id} not found in any queue"
        )