import re

from sortedcontainers import SortedKeyList
from typing import List

from entrypoints.api import Request, RequestStage
from models.llama3.tokenizer import Tokenizer
from schedulers.utils import register_scheduler


@register_scheduler("preemptible_srpt")
class PreemptibleSRPT_Scheduler:
    def __init__(
        self,
        batch_size,
        min_batches_before_preemption,
        max_batches_before_promotion,
        scoring_method,
        initial_score,               # Needed for EstimatedRPT_Scorer.
        tokens_per_word,             # Needed for EstimatedRPT_Scorer.
        fcr_ratio,               # Needed for EstimatedRPT_Scorer.
        tokenizer_path,             # Needed for EstimatedRPT_Scorer.
        num_tokens_before_scoring,   # Needed for EstimatedRPT_Scorer.
        **kwargs,
    ) -> None:

        super(PreemptibleSRPT_Scheduler, self).__init__()

        self.batch_size = batch_size

        # Scheduler 'timestep' / num batches it has scheduled.
        self.num_batches_scheduled = 0

        # Maps request id to request. Needed for remove_request().
        self.id_to_request = {}

        # Maps request id to score. Lower scores are scheduled first.
        self.request_scores = {}

        # Holds request ids ordered by scores.
        self.prioritized_request_ids = SortedKeyList(
            [], key=lambda x: self.request_scores[x]
        )

        # Preemption / promotion thresholds.
        self.min_batches_before_preemption = min_batches_before_preemption
        self.max_batches_before_promotion = max_batches_before_promotion

        # The first timestep a request is scheduled such that the request has
        # also been scheduled for all subsequent timesteps. Used to decide
        # if requests can be preempted.
        self.first_timestep_scheduled = {}

        # The last timestep at which a prefill / decode iteration was run on
        # a request. Used to check if requests need promotion.
        self.last_timestep_scheduled = {}

        # Keep track of the previous batch so that requests can remain in batch
        # until min_batches_before_preemption.
        self.prev_batch: List[Request] = []

        # Set up function to score requests.
        assert scoring_method in ["prefill_length", "estimated_rpt"]
        self.scoring_method = scoring_method

        if scoring_method == "estimated_rpt":
            self.initial_score = initial_score
            self.tokens_per_word = tokens_per_word
            self.num_tokens_before_scoring = num_tokens_before_scoring
            self.tokenizer = Tokenizer(tokenizer_path)
            self.fcr_ratio = fcr_ratio

            # Maps request id to predicted length. Does not change. Used for FCR.
            self.request_perceived_lengths = {}
    
    def compute_score(self, request: Request):
        if self.scoring_method == "prefill_length":
            assert request.prompt_tokens is not None
            return len(request.prompt_tokens)
        
        elif self.scoring_method == "estimated_rpt":
            if len(request.output_tokens) < self.num_tokens_before_scoring:
                return self.initial_score
            
            score_str = self.tokenizer.decode(
                request.output_tokens[:self.num_tokens_before_scoring]
            )

            if "\n" not in score_str:
                return request.max_gen_len
            
            length_estimate = score_str.split("\n")[0].strip()

            # Finds the first contiguous substring of digits
            match = re.search(r'\d+', length_estimate)
            if not match:
                return request.max_gen_len
            
            length_estimate = match.group()
            assert length_estimate

            if not length_estimate.isdigit():
                return request.max_gen_len
            
            # Now we know that length_estimate is for sure a number
            # NOTE: we take minimum with max_gen_len to take into account truncation.
            length_estimate = min(int(length_estimate) * self.tokens_per_word, request.max_gen_len)
            request.estimated_token_length = length_estimate
            return length_estimate
        
        return None

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

        # Score request and add to sorted request container.
        request_score = self.compute_score(request)
        self.request_scores[request_id] = request_score
        self.prioritized_request_ids.add(request_id)

        if self.scoring_method == "estimated_rpt":
            self.request_perceived_lengths[request_id] = request_score

        return request

    def update_scores(self, requests):
        """Updates scores of requests in the input list."""
        for request in requests:
            request_id = request.request_id

            # Rescore request. If score is same, no change needed to MLFQ
            # position.
            old_score = self.request_scores[request_id]

            # For ESTIMATED_RPT check if the the estimate is more the FCR_RATIO off, if so drop the score to the max score
            if (self.scoring_method == "estimated_rpt" 
                  and len(request.output_tokens) > self.request_perceived_lengths[request_id] * self.fcr_ratio):
                new_score = request.max_gen_len
                self.request_perceived_lengths[request_id] = new_score
            else:
                new_score = self.compute_score(request)

            if new_score == old_score:
                continue

            # Score changed, so set the request's new score and move it in the
            # sorted requests container appropriately.
            self.prioritized_request_ids.remove(request_id)
            self.request_scores[request_id] = new_score
            self.prioritized_request_ids.add(request_id)

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

        return promotion_eligible

    def promote_requests(self, batch):
        """Adds as many promotion eligible requests as possible to the batch."""
        curr_batch_request_ids = set([request.request_id for request in batch])

        # HACK: Sometimes requests are still in prioritized_request_ids that 
        # aren't in self.id_to_request. Find these and delete them at the end.
        request_ids_to_delete = []

        # TODO: Make this sublinear in number of requests.
        for request_id in self.prioritized_request_ids:

            # Add missing requests to list of requests to remove from scheduler.
            if request_id not in self.id_to_request:
                request_ids_to_delete.append(request_id)

                # Remove the missing request from the batch if needed.
                if request_id in curr_batch_request_ids:
                    curr_batch_request_ids.remove(request_id)
                
                batch = [r for r in batch if r.request_id != request_id]

                continue

            request = self.id_to_request[request_id]

            # If the batch is already full, don't try to promote more requests.
            if len(batch) == self.batch_size:
                break

            # Don't try to add to batch if request is not promotion eligible.
            if not self.check_promotion_eligible(request):
                continue

            # Only add to batch if the promotion candidate is not already in it.
            if request_id not in curr_batch_request_ids:
                batch.append(request)
                curr_batch_request_ids.add(request_id)
        
        # Delete missing requests.
        for request_id in request_ids_to_delete:
            self.prioritized_request_ids.remove(request_id)

            if request_id in self.request_scores:
                del self.request_scores[request_id]
            
            if self.scoring_method == "estimated_rpt" and request_id in self.request_perceived_lengths:
                del self.request_perceived_lengths[request_id]

            if request_id in self.last_timestep_scheduled:
                del self.last_timestep_scheduled[request_id]

            if request_id in self.first_timestep_scheduled:
                del self.first_timestep_scheduled[request_id]

        self.prev_batch = [
            request
            for request in self.prev_batch
            if request.request_id not in request_ids_to_delete
        ]


    def add_high_priority_requests(self, batch):
        curr_batch_request_ids = set([request.request_id for request in batch])

        for request_id in self.prioritized_request_ids:
            if len(batch) == self.batch_size:
                break

            if request_id in curr_batch_request_ids:
                continue

            batch.append(self.id_to_request[request_id])
            curr_batch_request_ids.add(request_id)

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

    def schedule(self, stage: RequestStage) -> List[Request]:
        """Schedules a batch to run prefill / decode iterations on. The stage
        supplied is currently ignored; it is assumed that different instances
        of the skip join scheduler will be created for prefill and decode.

        - Update scores and MLFQ positions for previously scheduled requests.
        - Preempt eligible requests.
        - Promote starving requests.
        - Schedule high priority / low scoring requests.
        - Update metadata.
        """

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

    def remove_request(self, request: Request):
        finished_request_id = request.request_id

        if finished_request_id not in self.id_to_request:
            raise ValueError(
                f"Request with id {finished_request_id} not in scheduler."
            )

        self.prioritized_request_ids.remove(finished_request_id)
        del self.request_scores[finished_request_id]
        if self.scoring_method == "estimated_rpt":
            del self.request_perceived_lengths[finished_request_id]

        del self.last_timestep_scheduled[finished_request_id]

        if finished_request_id in self.first_timestep_scheduled:
            del self.first_timestep_scheduled[finished_request_id]

        del self.id_to_request[finished_request_id]

        self.prev_batch = [
            request
            for request in self.prev_batch
            if request.request_id != finished_request_id
        ]
