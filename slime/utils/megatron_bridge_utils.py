from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


def patch_hf_config_for_megatron_bridge(hf_config):
    configs = []
    seen_config_ids = set()

    def add_config(config):
        if config is None or id(config) in seen_config_ids:
            return
        seen_config_ids.add(id(config))
        configs.append(config)

    add_config(hf_config)
    add_config(getattr(hf_config, "config", None))

    for config in list(configs):
        add_config(getattr(config, "text_config", None))

    for config in configs:
        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params and not hasattr(config, "rope_theta"):
            config.rope_theta = rope_params["rope_theta"]

    return hf_config


def patch_auto_bridge_hf_config(bridge):
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        patch_hf_config_for_megatron_bridge(hf_pretrained)

    return bridge


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
