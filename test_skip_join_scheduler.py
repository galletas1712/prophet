from entrypoints.api import Request, CompletionType
from schedulers.baselines import SkipJoinMLFQ_Scheduler


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

    min_batches_before_preemption = 4
    max_batches_before_promotion = 8

    scheduler = SkipJoinMLFQ_Scheduler(
        batch_size=1,
        num_queues=4,
        queue_limits=[2, 4, 8, 16],
        min_batches_before_preemption=min_batches_before_preemption,
        max_batches_before_promotion=max_batches_before_promotion,
        scoring_method="prefill_length",
        initial_score=0,
    )

    # Low priority request.
    request_1 = Request("12345678", CompletionType.TEXT_COMPLETION)
    scheduler.add_request(request_1)

    # High priority request.
    request_2 = Request("2", CompletionType.TEXT_COMPLETION)
    scheduler.add_request(request_2)
    
    for _ in range(5):
        # High priority request should be scheduled first, and should run for
        # max_batches_before_promotion threshold before request_1 is promoted 
        # into batch.
        for __ in range(max_batches_before_promotion):
            batch = scheduler.schedule()

            assert len(batch) == 1        
            assert batch[0].request_id == request_2.request_id
        
        # Now, request_1 should be promoted and run for 
        # min_batches_before_preemption.
        for __ in range(min_batches_before_preemption):
            batch = scheduler.schedule()

            assert len(batch) == 1        
            assert batch[0].request_id == request_1.request_id
    
    print(f"---- TEST PASSED ----\n")



def main():
    test_preemption_promotion_thresholds()


if __name__ == "__main__":
    main()
