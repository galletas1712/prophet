from entrypoints.api import Request, RequestStage
from typing import List
from sortedcontainers import SortedDict

from schedulers.utils import register_scheduler


# TODO: scoring method
@register_scheduler("srpt")
class SRPTScheduler:
    def __init__(self, batch_size, scoring_method, starvation_limit=None, **kwargs) -> None:
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

        # storage: (prompt length, request_id) -> (request, added_time or last_batched_time)
        self.requests[(len(request.prompt_tokens),
                       request.request_id)] = (request, self.now)

    def schedule(self, stage: RequestStage) -> List[Request]:
        self.now += 1
        
        batch = []
        for _, (request, _) in self.requests.items():
            assert request.stage is not RequestStage.DONE
            if request.stage is not stage:
                continue
            batch.append(request) 
            # no need to update last time as always complet
            if len(batch) == self.batch_size:
                break
        
        print([r.request_id for r in batch])

        for _, (request, added_time) in self.requests.items():
            if self.now - added_time > self.starvation_limit:
                del self.requests[(len(request.prompt_tokens), request.request_id)]
                self.requests[(0, request.request_id)] = (request, self.now)
                print(f"Promoted starved request {request.request_id}")

        return batch

    def remove_request(self, request: Request):
        request_key = (len(request.prompt_tokens), request.request_id)
        del self.requests[request_key]
