from entrypoints.api import Request, CompletionType
from schedulers.score import SkipJoinMLFQ_Scheduler
from entrypoints.api import RequestStage


def test_preemption_promotion_thresholds():
    """
    Scheduler is initialized with:
        batch_size=1
        min_batches_before_preemption=4
        max_batches_before_promotion=8

    Schedule two requests:
        1. Long length, high score, low priority.
        2. Short length, low score, high priority.

    Short length request should be scheduled first as higher priority. Should
    run for max_batches_before_promotion = 8 since the only other request that
    can be scheduled is lower priority.

    After 8 iterations, request 1 is swapped in because of meeting promotion
    threshold. Runs for 4 iterations, and is then preempted because of a higher
    priority request (number 2).
    """
    print(f"---- TESTING PREEMPTION / PROMOTION THRESHOLDS ----\n")

    scheduler = SkipJoinMLFQ_scheduler(
        batch_size=1,
        num_queues=4,
        queue_limits=[2, 4, 8, 16],
        starvation_limit=32
    )

    # Low priority request.
    request_1 = Request("123", CompletionType.TEXT_COMPLETION)
    scheduler.add_request(request_1)

    # High priority request.
    request_2 = Request("2", CompletionType.TEXT_COMPLETION)
    scheduler.add_request(request_2)

    for __ in range(2):
        batch = scheduler.schedule(stage=RequestStage.PREFILL)
        assert len(batch) == 1        
        assert batch[0].request_id == request_2.request_id

    for __ in range(2):
        batch = scheduler.schedule(stage=RequestStage.PREFILL)
        assert len(batch) == 1        
        assert batch[0].request_id == request_1.request_id
    
    
    for i in [2, 4, 8]:
        # High priority request should be scheduled first, and should run for
        # max_batches_before_promotion threshold before request_1 is promoted 
        # into batch.
        # Now, request_1 should be promoted and run for 
        # min_batches_before_preemption.
        for __ in range(i):
            batch = scheduler.schedule(stage=RequestStage.PREFILL)
            assert len(batch) == 1        
            assert batch[0].request_id == request_1.request_id

        for __ in range(i):
            batch = scheduler.schedule(stage=RequestStage.PREFILL)
            assert len(batch) == 1        
            assert batch[0].request_id == request_2.request_id
        
        
    
    print(f"---- TEST PASSED ----\n")



def main():
    test_preemption_promotion_thresholds()


if __name__ == "__main__":
    main()
