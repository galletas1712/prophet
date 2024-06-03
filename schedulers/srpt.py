from entrypoints.api import Request, RequestStage
from typing import List
from sortedcontainers import SortedDict

from schedulers.utils import register_scheduler


@register_scheduler("srpt")
class SRPTScheduler:
    def __init__(self, batch_size, **kwargs) -> None:
        super(SRPTScheduler, self).__init__()
        self.batch_size = batch_size
        self.requests = SortedDict()

    def add_request(self, request) -> Request:
        # NOTE: Using prompt token length for now (even for decode)
        # NOTE: Important the we don't modify prompt_tokens after prefill!
        self.requests[(len(request.prompt_tokens), request.request_id)] = request

    def schedule(self, stage: RequestStage) -> List[Request]:
        batch = []

        for _, request in self.requests.items():
            assert request.stage is not RequestStage.DONE
            if request.stage is not stage:
                continue
            batch.append(request)
            if len(batch) == self.batch_size:
                break

        return batch

    def remove_request(self, request: Request):
        request_key = (len(request.prompt_tokens), request.request_id)
        del self.requests[request_key]

