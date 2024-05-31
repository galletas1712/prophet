from collections import OrderedDict, deque
from typing import List

from entrypoints.api import Request, RequestStage, CompletionType
from schedulers.utils import register_scheduler

import random


@register_scheduler("fcfs")
class FCFS_Scheduler:

    def __init__(self, batch_size, **kwargs):
        super(FCFS_Scheduler, self).__init__()

        self.batch_size = batch_size
        self.request_dict = OrderedDict()

    def add_request(self, request):
        self.request_dict[request.request_id] = request

    def schedule(self, stage: RequestStage) -> List[Request]:
        # Iterate over the requests dict, popping items that have finished.
        batch = []

        for request_id, request in self.request_dict.items():
            assert request.stage is not RequestStage.DONE
            if request.stage is not stage:
                continue
            batch.append(request)
            if len(batch) == self.batch_size:
                break

        return batch

    def remove_request(self, finished_request_id):
        self.request_dict.pop(finished_request_id)

@register_scheduler("random")
class Random_Scheduler:

    def __init__(self, batch_size, **kwargs):
        super(Random_Scheduler, self).__init__()

        self.batch_size = batch_size
        self.request_dict = {}

    def add_request(self, request):
        self.request_dict[request.request_id] = request

    def schedule(self, stage: RequestStage) -> List[Request]:
        # Iterate over the requests dict, popping items that have finished.
        batch = []

        items = list(self.request_dict.items())
        random.shuffle(items)
        for request_id, request in items:
            assert request.stage is not RequestStage.DONE
            if request.stage is not stage:
                continue
            batch.append(request)
            if len(batch) == self.batch_size:
                break

        return batch

    def remove_request(self, finished_request_id):
        self.request_dict.pop(finished_request_id)

# TODO(cathy) change definition of promptlen for dialog completion
@register_scheduler("srpt")
class SRPT_Scheduler:
    def __init__(self, batch_size, **kwargs) -> None:
        super(SRPT_Scheduler, self).__init__()
        self.batch_size = batch_size
        self.request_list = []  # TODO(cathy) could ordered dict here

    def add_request(self, request) -> Request:
        left, right = 0, len(self.request_list)
        while left < right:
            mid = (left + right) // 2
            if len(self.request_list[mid].prompt) < len(request.prompt):
                left = mid + 1
            else:
                right = mid
        self.request_list.insert(left, request)
        return request

    def schedule(self, stage: RequestStage) -> List[Request]:
        batch = []

        idx = 0
        while idx < len(self.request_list):
            request = self.request_list[idx]
            assert request.stage is not RequestStage.DONE
            if request.stage == stage:
                batch.append(request)
                if len(batch) == self.batch_size:
                    break
            idx += 1

        return batch

    def remove_request(self, finished_request_id):
        self.request_list = [
            request
            for request in self.request_list
            if request.request_id != finished_request_id
        ]


class PrefillLengthScorer:

    def __init__(self, *args) -> None:
        pass

    def __call__(self, request: Request):
        if request.prompt_tokens is None:
            return len(request.prompt)
        return len(request.prompt_tokens)


@register_scheduler("skip_join_mlfq")
class SkipJoinMLFQ_Scheduler:
    def __init__(
        self,
        batch_size,
        num_queues=4,
        queue_limits=[16, 32, 64, 128],
        min_batches_before_preemption=4,
        max_batches_before_promotion=256,
        scoring_method="prefill_length",
        initial_score=0,
        **kwargs,
    ) -> None:

        super(SkipJoinMLFQ_Scheduler, self).__init__()

        self.batch_size = batch_size

        # Scheduler 'timestep' / num batches it has scheduled.
        self.num_batches_scheduled = 0

        # Build queues.
        self.num_queues = num_queues
        self.queue_limits = queue_limits
        self.request_queues = [[] for _ in range(num_queues)]

        # Preemption / promotion thresholds.
        self.min_batches_before_preemption = min_batches_before_preemption
        self.max_batches_before_promotion = max_batches_before_promotion

        # Keep track of the previous batch so that we can update its scores.
        # prev_batch holds prev requests themselves.
        self.prev_batch: List[Request] = []

        # Set up function to score requests.
        assert scoring_method in ["prefill_length", "estimated_rpt"]

        if scoring_method == "prefill_length":
            self.scorer = PrefillLengthScorer(initial_score)
        elif scoring_method == "estimated_rpt_score":
            self.scorer = None

        # ---- EXTRA STATE FOR REQUESTS, KEY IS REQUEST ID ----

        # Maps request ids to requests.
        self.id_to_request = {}

        # Maps request id to (queue_idx, idx_in_queue); this is the current
        # position of the request in the MLFQ data structure.
        self.request_positions = {}

        # Requests with low scores are scheduled first. TODO: Use something like
        # a min heap to quickly find lowest scoring requests.
        self.request_scores = {}

        # The first timestep a request is scheduled such that the request has
        # also been scheduled for all subsequent timesteps. Used to decide
        # if requests can be preempted.
        self.first_timestep_scheduled = {}

        # The last timestep at which a prefill / decode iteration was run on
        # a request. Used to check if requests need promotion.
        self.last_timestep_scheduled = {}

    def add_request(self, request: Request):
        request_id = request.request_id
        self.id_to_request[request_id] = request

        # Initialize first / last scheduled time to curr timestep - 1. This
        # means that we will be eligible for promotion in
        # max_batches_before_promotion steps inclusive of the current iteration.
        self.last_timestep_scheduled[request_id] = (
            self.num_batches_scheduled - 1
        )

        self.first_timestep_scheduled[request_id] = (
            self.num_batches_scheduled - 1
        )

        # Score request.
        request_score = self.scorer(request)
        self.request_scores[request_id] = request_score

        # Skip-join step. Priority is set to the quantum larger than the first
        # iteration quantum.
        for i, limit in enumerate(self.queue_limits):
            if request_score < limit or i == len(self.queue_limits) - 1:
                idx_in_queue = len(self.request_queues[i])
                self.request_queues[i].append(request)
                self.request_positions[request_id] = (i, idx_in_queue)
                break

        return request

    def compute_queue_idx(self, request_id):
        """Returns the index of the MLFQ queue that request should be inside."""
        for queue_idx, queue_limit in enumerate(self.queue_limits):
            if queue_limit > self.request_scores[request_id]:
                return queue_idx
        return self.num_queues - 1

    def update_scores(self, requests):
        """Updates scores and MLFQ positions of requests in the input list."""

        for request_idx, request in enumerate(requests):
            request_id = request.request_id

            # Rescore request. If score is same, no change needed to MLFQ
            # position.
            old_score = self.request_scores[request_id]
            new_score = self.scorer(request)

            if new_score == old_score:
                continue

            # Score changed, so set the request's new score and move it to
            # correct queue.
            self.request_scores[request_id] = new_score

            old_pos = self.request_positions[request_idx]
            old_queue_idx, old_idx_in_queue = old_pos

            new_queue_idx = self.compute_queue_idx(request_id)
            new_idx_in_queue = len(self.request_queues[new_queue_idx])

            if new_queue_idx == old_queue_idx:
                continue

            self.request_queues[old_queue_idx].pop(old_idx_in_queue)
            self.request_queues[new_queue_idx].append(request)
            self.request_positions[request_id] = (
                new_queue_idx,
                new_idx_in_queue,
            )

    def check_preemption_eligible(self, request):
        """Returns if a request should be preempted."""
        request_id = request.request_id

        if request_id not in self.first_timestep_scheduled:
            return False

        # Num times request was consecutively scheduled assuming that it's also
        # scheduled at the current timestep (self.num_batches_scheduled).
        num_consecutive_scheduled = (
            self.num_batches_scheduled
            - self.first_timestep_scheduled[request_id]
            + 1
        )

        preemption_eligible = (
            num_consecutive_scheduled > self.min_batches_before_preemption
        )

        # print(f"prompt = {request.prompt}, preemption_eligible = {preemption_eligible}")

        return preemption_eligible

    def check_promotion_eligible(self, request):
        """Returns if a request should be promoted."""
        request_id = request.request_id

        if request_id not in self.last_timestep_scheduled:
            return False

        # Num times request was consecutively not scheduled assuming it's also
        # not scheduled at the current timestep (self.num_batches_scheduled).
        num_consecutive_not_scheduled = (
            self.num_batches_scheduled
            - self.last_timestep_scheduled[request_id]
        )

        promotion_eligible = (
            num_consecutive_not_scheduled > self.max_batches_before_promotion
        )

        # print(f"prompt = {request.prompt}, promotion_eligible = {promotion_eligible}")

        return promotion_eligible

    def promote_requests(self, batch):
        """Adds as many promotion eligible requests as possible to the batch."""
        curr_batch_request_ids = set([request.request_id for request in batch])

        for request_id, request in self.id_to_request.items():
            # If the batch is already full, don't try to promote more requests.
            if len(batch) == self.batch_size:
                break

            # Don't try to add to batch if request is not promotion eligible.
            if not self.check_promotion_eligible(request):
                continue

            # Only add to batch if the promotion candidate is not already in it.
            if request.request_id not in curr_batch_request_ids:
                batch.append(request)
                curr_batch_request_ids.add(request.request_id)

    def add_high_priority_requests(self, batch):
        curr_batch_request_ids = set([request.request_id for request in batch])

        candidate_request_ids = list(self.request_scores.keys())
        candidate_request_ids = sorted(
            candidate_request_ids, key=lambda x: self.request_scores[x]
        )

        for candidate_request_id in candidate_request_ids:
            if len(batch) == self.batch_size:
                break

            if candidate_request_id in curr_batch_request_ids:
                continue

            batch.append(self.id_to_request[candidate_request_id])
            curr_batch_request_ids.add(candidate_request_id)

    def update_ages(self, batch, prev_batch):
        """Updates first_timestep_scheduled and last_timestep_scheduled attrs.

        - Remove entries in first_timestep_scheduled for preempted / done
          requests.
        - Add entries in first_timestep_scheduled for newly scheduled requests.
        - Update last_timestep_scheduled for all requests in the new batch.
        """

        batch_request_ids = [request.request_id for request in batch]
        prev_batch_request_ids = [request.request_id for request in prev_batch]

        removed_request_ids = [
            request_id
            for request_id in prev_batch_request_ids
            if request_id not in batch_request_ids
        ]

        added_request_ids = [
            request_id
            for request_id in batch_request_ids
            if request_id not in prev_batch_request_ids
        ]

        # Remove entries in first_timestep_scheduled for preempted / done
        # requests.
        for removed_request_id in removed_request_ids:
            del self.first_timestep_scheduled[removed_request_id]

        # Add entries in first_timestep_scheduled for newly scheduled requests.
        for added_request_id in added_request_ids:
            self.first_timestep_scheduled[added_request_id] = (
                self.num_batches_scheduled
            )

        # Update last_timestep_scheduled for all requests in the new batch.
        for batch_request_id in batch_request_ids:
            self.last_timestep_scheduled[batch_request_id] = (
                self.num_batches_scheduled
            )

    def schedule(self) -> List[Request]:
        """Schedules a batch to run prefill / decode iterations on.

        - Update scores and MLFQ positions for previously scheduled requests.
        - Preempt eligible requests.
        - Promote starving requests.
        - Schedule high priority / low scoring requests.
        - Update metadata.
        """

        # Batch returned to user, initialized to the previously scheduled batch.
        batch = [request for request in self.prev_batch]

        # Update scores and queue positions for previously scheduled requests.
        self.update_scores(self.prev_batch)

        # Preempt all eligible requests.
        batch = [
            request
            for request in self.prev_batch
            if not self.check_preemption_eligible(request)
        ]

        # Add as many promotion eligible requests as possible.
        self.promote_requests(batch)

        # Fill up batch with high priority / low score requests.
        self.add_high_priority_requests(batch)

        # Update request metadata for scheduling.
        self.update_ages(batch, self.prev_batch)

        # Other scheduler state updates.
        self.prev_batch = batch
        self.num_batches_scheduled += 1

        return batch

    def remove_request(self, finished_request_id):
        if finished_request_id not in self.request_positions:
            raise ValueError(
                f"Request with id {finished_request_id} not found in any queue"
            )

        del self.id_to_request[finished_request_id]

        queue_idx, req_idx = self.request_positions[finished_request_id]
        self.request_queues[queue_idx].pop(req_idx)

        del self.request_positions[finished_request_id]

        del self.last_timestep_scheduled[finished_request_id]

        if finished_request_id in self.first_timestep_scheduled:
            del self.first_timestep_scheduled[finished_request_id]
