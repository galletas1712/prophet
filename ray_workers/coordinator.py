import ray

@ray.remote
class Coordinator:
    def __init__(self, rank_mapping):
        self.rank_mapping = rank_mapping

    def send_tensor(self, sender_name, receiver_name, request_id, prompt_len):
        sender = ray.get_actor(sender_name)
        receiver = ray.get_actor(receiver_name)
        ray.get([
            sender.send_k.remote(request_id=request_id, target_rank=self.rank_mapping[receiver_name]),
            receiver.receive_k.remote(src_rank=self.rank_mapping[sender_name], prompt_len=prompt_len)])
        ray.get([
            sender.send_v.remote(request_id=request_id, target_rank=self.rank_mapping[receiver_name]),
            receiver.receive_v.remote(src_rank=self.rank_mapping[sender_name], prompt_len=prompt_len)])