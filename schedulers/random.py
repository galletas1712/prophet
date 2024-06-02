from entrypoints.api import Request, RequestStage
from schedulers.utils import register_scheduler
from typing import List

import random


@register_scheduler("random")
class RandomScheduler:

    def __init__(self, batch_size, **kwargs):
        super(RandomScheduler, self).__init__()

        self.batch_size = batch_size
        self.request_dict = {}

    def add_request(self, request):
        self.request_dict[request.request_id] = request

    def schedule(self, stage: RequestStage) -> List[Request]:
        # Iterate over the requests dict, popping items that have finished.
        batch = []

        items = list(self.request_dict.items())
        random.shuffle(items)
        for request_id, request in items:
            assert request.stage is not RequestStage.DONE
            if request.stage is not stage:
                continue
            batch.append(request)
            if len(batch) == self.batch_size:
                break

        return batch

    def remove_request(self, finished_request_id):
        self.request_dict.pop(finished_request_id)
