from typing import List
from collections import OrderedDict

from entrypoints.api import Request, RequestStage
from schedulers.utils import register_scheduler


@register_scheduler("fcfs")
class FCFS_Scheduler:

    def __init__(self, batch_size, kv_cache_shape, **kwargs):
        super(FCFS_Scheduler, self).__init__()

        self.kv_cache_shape = kv_cache_shape
        self.batch_size = batch_size

        self.request_dict = OrderedDict()
        self.next_id = 0

    def create_request(self, prompt: str) -> Request:
        request_id = self.next_id
        self.next_id += 1

        request = Request(request_id, prompt, self.kv_cache_shape)
        self.request_dict[request_id] = request

        return request

    def schedule(self) -> List[Request]:
        # Iterate over the requests dict, popping items that have finished.
        batch = []

        for request_id, request in self.request_dict.items():
            assert request.stage != RequestStage.DONE
            batch.append(request)
            if len(batch) == self.batch_size:
                break

        return batch

    def remove_request(self, finished_request_id):
        self.request_dict.pop(finished_request_id)
