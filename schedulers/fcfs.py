from collections import OrderedDict
from entrypoints.api import Request, RequestStage
from schedulers.utils import register_scheduler
from typing import List


@register_scheduler("fcfs")
class FCFSScheduler:

    def __init__(self, batch_size, **kwargs):
        super(FCFSScheduler, self).__init__()

        self.batch_size = batch_size
        self.request_dict = OrderedDict()

    def add_request(self, request):
        self.request_dict[request.request_id] = request

    def schedule(self, stage: RequestStage) -> List[Request]:
        # Iterate over the requests dict, popping items that have finished.
        batch = []

        for _, request in self.request_dict.items():
            assert request.stage is not RequestStage.DONE
            if request.stage is not stage:
                continue
            batch.append(request)
            if len(batch) == self.batch_size:
                break

        return batch

    def remove_request(self, request: Request):
        self.request_dict.pop(request.request_id)
