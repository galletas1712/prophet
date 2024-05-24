from typing import List

import torch
import time

class BenchmarkMetrics:

    def __init__(self):
        self.timestamps = [time.time()]
        self.request_finished = False
    
    def received_token(self):
        self.timestamps.append(time.time())
    
    def finished_request(self):
        all_timestamps = torch.tensor(self.timestamps, device="cpu", dtype=torch.float64)
        # Job completion time
        self.JCT = (all_timestamps)[-1] - all_timestamps[0]

        #  Adjacent difference of token times
        self.token_times = all_timestamps[1:] - all_timestamps[:-1]
        # Time to first *decode* (or prefill if no decodes) token
        self.TTFT = (self.token_times[0] + (self.token_times[1] if len(self.token_times) > 1 else 0)).item()
        # Time per output token (second decode token onwards)
        self.TPOT = self.token_times[2:].mean().item() if len(self.token_times) > 2 else None
        # Time to first *prefill* token
        self.TTFPT = self.token_times[0].item()
        # Time per output token (first decode token onwards)
        self.TPODT = self.token_times[1:].mean().item() if len(self.token_times) > 1 else None

        self.request_finished = True

    def __repr__(self):
        assert self.request_finished
        return f"JCT: {self.JCT}, TTFT: {self.TTFT}, TPOT: {self.TPOT}, TTFPT: {self.TTFPT}, TPODT: {self.TPODT}"