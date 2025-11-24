from contextlib import contextmanager
from megatron.core.utils import unwrap_model


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    assert not hasattr(model_config, "share_embeddings_and_output_weights")
    setattr(model_config, "share_embeddings_and_output_weights", unwrapped_model.share_embeddings_and_output_weights)

    try:
        yield
    finally:
        delattr(model_config, "share_embeddings_and_output_weights")
