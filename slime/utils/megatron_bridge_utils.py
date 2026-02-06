from contextlib import contextmanager
from functools import lru_cache

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@lru_cache(maxsize=1)
def get_bridge(hf_checkpoint: str):
    """Create or return cached AutoBridge instance. Bridge is stateless (only holds
    architecture metadata), so a single cached instance per hf_checkpoint is safe."""
    from megatron.bridge import AutoBridge

    return AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
