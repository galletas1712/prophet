from models.llama3 import Llama

def get_model_constructor(model_config):
    return Llama.build(
        ckpt_dir=model_config.ckpt_dir,
        tokenizer_path=model_config.tokenizer_path,
        max_seq_len=model_config.max_seq_len,
        max_batch_size=model_config.max_batch_size,
    )