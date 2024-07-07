import pandas as pd
import time

class RequestBenchmarkMetrics:

    def __init__(self):
        self.request_finished = False
        self.perfcount_request_dispatched = time.perf_counter()
    
    def started_prefill(self):
        # When the requeest actually gets added into the prefill batch
        self.perfcount_dispatched_prefill = time.perf_counter()
    
    def finished_prefill(self):
        # When the request is done in the prefill batch, but waiting to transfer to the decoder
        self.perfcount_finished_prefill = time.perf_counter()
        
    def began_transfer(self):
        # When the request starts getting transferred to the decoder
        self.perfcount_begin_transfer = time.perf_counter()
    
    def decoder_received(self):
        # When the request is received by the decoder
        self.perfcount_decoder_received = time.perf_counter()
    
    def started_decoding_token(self, epoch):
        # When the decoder starts decoding a token
        self.perfcounts_started_decoding_token.append(time.perf_counter())
        self.epochs_decoded_token.append(epoch)

    def finished_decoding_token(self):
        # When the decoder finishes decoding a token
        self.perfcounts_finished_decoding_token.append(time.perf_counter())
    
    def finished_decode(self):
        # When the request is finished in the decoder
        self.request_finished = True
    
    def get_stats(self):
        assert self.request_finished
        assert len(self.perfcounts_started_decoding_token) == len(self.perfcounts_finished_decoding_token)
        decoded_tokens = len(self.perfcounts_started_decoding_token)
        for i in range(decoded_tokens):
            assert self.perfcounts_started_decoding_token[i] < self.perfcounts_finished_decoding_token[i]

        stats = {}
        stats["e2e_JCT"] = self.perfcount_finished_decoding_token[-1] - self.perfcount_request_dispatched

        # Transfer metrics. NOTE: Only valid for intra-node transfers!
        stats["raw_transfer_time"] = self.perfcounter_decode_receved - self.perfcount_begin_transfer
        stats["prefill_kv_manager_wait_time"] = self.perfcount_begin_transfer - self.perfcount_finished_prefill
        stats["transfer_wait_time"] = self.perfcount_decoder_received - self.perfcount_finished_prefill

        # Prefill metrics
        stats["e2e_prefill_latency"] = self.perfcount_finished_prefill - self.perfcount_request_dispatched

        # Decode metrics
        stats["decode_latency_nowait"] = self.perfcount_finished_decoding_token[-1] - self.perfcount_started_decoding_token[0]
        stats["decoder_TTFT"] = self.perfcount_started_decoding_token[0] - self.perfcount_decoder_received
        stats["e2e_decode_latency"] = self.perfcount_finished_decoding_token[-1] - self.perfcount_decoder_received

        stats["active_decode_time"] = sum(self.perfcounts_finished_decoding_token[i] - self.perfcounts_started_decoding_token[i] for i in range(decoded_tokens))
        stats["idle_decode_time"] = self.e2e_decode_latency - self.active_decode_time

        stats["TPOT"] = self.e2e_decode_latency / decoded_tokens
        stats["TPOT_active_only"] = self.active_decode_time / decoded_tokens

        df = pd.DataFrame(stats)
        return df
