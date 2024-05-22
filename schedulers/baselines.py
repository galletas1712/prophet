from typing import List, Any
from collections import OrderedDict
import heapq

from entrypoints.api import Request, RequestStage, CompletionType
from schedulers.utils import register_scheduler


@register_scheduler("fcfs")
class FCFS_Scheduler:

    def __init__(self, batch_size, **kwargs):
        super(FCFS_Scheduler, self).__init__()

        self.batch_size = batch_size

        self.request_dict = OrderedDict()
        self.next_id = 0

    def create_request(self, prompt: str | Any, completion_type: CompletionType) -> Request:
        request_id = self.next_id
        self.next_id += 1

        request = Request(request_id, prompt, completion_type)
        self.request_dict[request_id] = request

        return request

    def schedule(self, stage: RequestStage) -> List[Request]:
        # Iterate over the requests dict, popping items that have finished.
        batch = []

        for request_id, request in self.request_dict.items():
            assert request.stage is not RequestStage.DONE
            if request.stage is not stage:
                continue
            batch.append(request)
            if len(batch) == self.batch_size:
                break

        return batch

    def remove_request(self, finished_request_id):
        self.request_dict.pop(finished_request_id)


@register_scheduler("srtp")
class SRTP_Scheduler():
    def __init__(self, batch_size, **kwargs) -> None:
        super(SRTP_Scheduler, self).__init__()
        self.batch_size = batch_size

        self.request_heap = []
        self.next_id = 0
    
    def create_request(self, prompt: str) -> Request:
        request_id = self.next_id
        self.next_id += 1

        request = Request(request_id, prompt)
        heapq.heappush(self.request_heap, (len(prompt), request))

        return request

    def schedule(self) -> List[Request]:
        batch = []

        for _ in range(min(self.batch_size, len(self.request_heap))):
            _, request = self.request_heap[0]
            assert request.stage != RequestStage.DONE
            batch.append(request)

        return batch

    def remove_request(self, finished_request_id):
        self.request_heap = [(length, request) for length, request in self.request_heap if request.id != finished_request_id]


@register_scheduler("skip-join-mlfq")
class SJMLFQ_scheduler():
    def __init__(
            self, 
            batch_size, 
            num_queues = 4, # TODO parameterize
            queue_limits = [32, 64, 128, 256],
            **kwargs,
        ) -> None:
        super(SJMLFQ_scheduler, self).__init__()
        self.batch_size = batch_size
        self.num_queues = num_queues
        self.queue_limits = queue_limits

        self.request_queues = [[] for _ in range(num_queues)]
        self.next_id = 0

    def create_request(self, prompt: str) -> Request:
        request_id = self.next_id
        self.next_id += 1

        request = Request(request_id, prompt)

        # skip-join step
        # priority is set to the quantum larger than the first iteration quantum
        request_added = False
        for i, limit in enumerate(self.queue_limits):
            if len(prompt) < limit:
                self.request_queues[i].append((request, 0))# request, iteration_number
                request_added = True
                break
        if not request_added:
            self.request_queues[-1].append((request, 0))
        
        return request
    
    def schedule(self) -> List[Request]:
        batch = []

        for i in range(self.num_queues):
            for req_idx, (request, iteration_number) in enumerate(self.request_queues[i]):
                assert request.stage != RequestStage.DONE
                if iteration_number >= self.queue_limits[i]:
                    self._move_request_to_lower_queue(i, req_idx)
                batch.append(request)
                self.request_queues[i][req_idx] = (request, iteration_number+1)

                if len(batch) == self.batch_size:
                    break 
        
        return batch
    
    def _move_request_to_lower_queue(self, queue_idx, req_idx):
        if queue_idx == self.num_queues - 1:
            return
        request, iteration_number = self.request_queues[queue_idx].pop(req_idx)
        self.request_queues[queue_idx+1].append((request, iteration_number))

    # TODO(cathy) add starve limit

    def remove_request(self, finished_request_id):
        for i in range(self.num_queues):
            for req_idx, (request, _) in enumerate(self.request_queues[i]):
                if request.id == finished_request_id:
                    self.request_queues[i].pop(req_idx)
                    return
        raise ValueError(f"Request with id {finished_request_id} not found in any queue")


        