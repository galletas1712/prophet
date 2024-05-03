from collections import OrderedDict

from utils import SCHEDULERS, register_scheduler, Request, RequestStage


@register_scheduler("fcfs")
class FCFS_Scheduler:

    def __init__(self, batch_size):
        super(FCFS_Scheduler, self).__init__()
        self.batch_size = batch_size

        self.requests = OrderedDict()
        self.next_id = 0
    
    def add_request(self, prompt):
        request_id = self.next_id
        self.next_id += 1

        request = Request(request_id, prompt)
        self.request_dict[request_id] = request

    def schedule(self):
        # Iterate over the requests dict, popping items that have finished.
        batch = []
        to_pop = []

        for request_id, request in self.requests.items():
            if request.stage != RequestStage.DONE:
                batch.append(request)
                if len(batch) == self.batch_size:
                    break
            else:
                to_pop.append(request_id)
        
        for finished_request in to_pop:
            self.requests.pop(finished_request)

        return batch
