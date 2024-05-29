from typing import List

import torch
import time

class RequestBenchmarkMetrics:

    def __init__(self):
        self.timestamps = [time.time()]
        self.request_finished = False
        self.gen_len = 0
    
    def received_token(self):
        self.timestamps.append(time.time())
        self.gen_len += 1
    
    def finished_request(self):
        all_timestamps = torch.tensor(self.timestamps, device="cpu", dtype=torch.float64)
        # Job completion time
        self.JCT = ((all_timestamps)[-1] - all_timestamps[0]).item()

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

        self.stats_trunc = list(map("{:.3f}".format, [self.JCT, self.TTFT, self.TPOT, self.TTFPT, self.TPODT]))
        self.request_finished = True

    def __str__(self):
        assert self.request_finished
        return f"JCT: {self.JCT}, TTFT: {self.TTFT}, TPOT: {self.TPOT}, TTFPT: {self.TTFPT}, TPODT: {self.TPODT}"
    
    def to_csv_row(self):
        assert self.request_finished
        return ','.join([str(self.gen_len)] + self.stats_trunc) + '\n'