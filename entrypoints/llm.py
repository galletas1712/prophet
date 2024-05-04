from entrypoints.api import *

class LLM:

    def __init__(self, scheduler, model_constructor) -> None:
        self.scheduler = scheduler
        self.model = model_constructor()
        
        self.request_id_to_response = {}
        self.completed_request_ids = []
    
    def create_request(self, prompt: str):
        return self.scheduler.create_request(prompt)

    def request_completed(self):
        return len(self.completed_request_ids) > 0

    def pop_completed_request_id(self):
        return self.completed_request_ids.pop()
    
    def prepare_batch(self, batch):
        pass

    def step(self):
        batch = self.scheduler.request()

