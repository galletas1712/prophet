from typing import List, Any
from collections import OrderedDict

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
