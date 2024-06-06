from entrypoints.api import Request, RequestStage
from typing import List
from sortedcontainers import SortedDict

from schedulers.utils import register_scheduler


# TODO: scoring method
@register_scheduler("srpt")
class SRPTScheduler:
    def __init__(self, batch_size, scoring_method, starvation_limit, **kwargs) -> None:
        super(SRPTScheduler, self).__init__()
        self.batch_size = batch_size
        assert scoring_method == 'prefill_length'
        self.scoring_method = scoring_method
        self.requests = SortedDict()

        self.starvation_limit = starvation_limit
        self.now = 0

    def add_request(self, request) -> Request:
        # NOTE: Using prompt token length for now (even for decode)
        # NOTE: Important the we don't modify prompt_tokens after prefill!

        # storage: (priority, prompt length, request_id) -> (request, added_time or last_batched_time)
        # priority either 1 (default) or 0 (promoted after starvation)
        self.requests[(1, len(request.prompt_tokens),
                       request.request_id)] = (request, self.now)

    def schedule(self, stage: RequestStage) -> List[Request]:
        self.now += 1
        batch = []
        for _, (request, _) in self.requests.items():
            assert request.stage is not RequestStage.DONE
            if request.stage is not stage:
                continue
            batch.append(request)
            if len(batch) == self.batch_size:
                break

        if self.starvation_limit is not None:
            for _, (request, added_time) in self.requests.items():
                if self.now - added_time >= self.starvation_limit:
                    self.remove_request(request)
                    self.requests[(0, len(request.prompt_tokens), request.request_id)] = (request, self.now)

        return batch

    def remove_request(self, request: Request):
        request_key = (1, len(request.prompt_tokens), request.request_id)
        if request_key in self.requests.keys():
            del self.requests[request_key]
        else:
            request_key = (0, len(request.prompt_tokens), request.request_id)
            del self.requests[request_key]
