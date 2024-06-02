from entrypoints.api import Request, RequestStage
from typing import List

from schedulers.utils import register_scheduler


# TODO(cathy) change definition of promptlen for dialog completion
@register_scheduler("srpt")
class SRPTScheduler:
    def __init__(self, batch_size, **kwargs) -> None:
        super(SRPTScheduler, self).__init__()
        self.batch_size = batch_size
        self.request_list = []  # TODO(cathy) could ordered dict here

    def add_request(self, request) -> Request:
        left, right = 0, len(self.request_list)
        while left < right:
            mid = (left + right) // 2
            if len(self.request_list[mid].prompt) < len(request.prompt):
                left = mid + 1
            else:
                right = mid
        self.request_list.insert(left, request)
        return request

    def schedule(self, stage: RequestStage) -> List[Request]:
        batch = []

        idx = 0
        while idx < len(self.request_list):
            request = self.request_list[idx]
            assert request.stage is not RequestStage.DONE
            if request.stage == stage:
                batch.append(request)
                if len(batch) == self.batch_size:
                    break
            idx += 1

        return batch

    def remove_request(self, finished_request_id):
        self.request_list = [
            request
            for request in self.request_list
            if request.request_id != finished_request_id
        ]

