MODELS = {}


def register_model(name):
    def register_curr_model(model_fn):
        MODELS[name] = model_fn
        return model_fn

    return register_curr_model


@register_model("dummy")
def build_dummy(model_config):
    from models.dummy_model import DummyModel

    return DummyModel(**model_config)


@register_model("llama_3")
def build_llama_3(model_config):
    from models.llama3 import Llama, GlobalGenerationParams

    glob_params = GlobalGenerationParams(
        **model_config.global_generation_params
    )

    return Llama.build(
        ckpt_dir=model_config.ckpt_dir,
        tokenizer_path=model_config.tokenizer_path,
        max_seq_len=model_config.max_seq_len,
        max_batch_size=model_config.max_batch_size,
        glob_params=glob_params,
    )


def build_model(model_config):
    return MODELS[model_config.name](model_config)
