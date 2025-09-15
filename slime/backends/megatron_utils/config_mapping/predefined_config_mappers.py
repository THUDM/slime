from collections import namedtuple
import torch.nn.functional as F
from transformers import PretrainedConfig
from .registry import register_mapper


MegatronModelConfig = namedtuple("MegatronModelConfig", ["transformer_config", "gpt_model_args"])


def _get_activation_func(name: str):
    if name == "silu":
        return F.silu
    elif name == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def _to_n_args(value):
    if isinstance(value, list):
        return value
    return [value]


def _map_common_configs(hf_config: PretrainedConfig) -> MegatronModelConfig:
    rope_scaling_args = {}
    if "rope_scaling" in hf_config and hf_config.rope_scaling is not None:
        rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling["factor"]
    return MegatronModelConfig(
        transformer_config={
            # Model architecture parameters
            "num_layers": hf_config.num_hidden_layers,
            "hidden_size": hf_config.hidden_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_query_groups": hf_config.num_key_value_heads,
            "ffn_hidden_size": hf_config.intermediate_size,
            "kv_channels": getattr(hf_config, "head_dim", None),
            "layernorm_epsilon": hf_config.rms_norm_eps,
            # Activation and normalization
            "activation_func": _get_activation_func(hf_config.hidden_act),
            "normalization": "RMSNorm",
            "gated_linear_unit": True,
        },
        gpt_model_args={
            "vocab_size": hf_config.vocab_size,
            "rotary_base": hf_config.rope_theta,
            "position_embedding_type": "rope",
            "untie_embeddings_and_output_weights": not hf_config.tie_word_embeddings,
        },
    )


@register_mapper("qwen2")
def qwen2_config_mapper(hf_config: PretrainedConfig) -> MegatronModelConfig:
    mapped_config = _map_common_configs(hf_config)
    mapped_config.transformer_config.update(
        {
            "add_bias_linear": False,
            "add_qkv_bias": hf_config.attention_bias,
        }
    )

    return mapped_config


@register_mapper("qwen3")
def qwen3_config_mapper(hf_config: PretrainedConfig) -> MegatronModelConfig:
    mapped_config = _map_common_configs(hf_config)
    mapped_config.transformer_config.update(
        {
            "add_bias_linear": False,
            "add_qkv_bias": hf_config.attention_bias,
            "qk_layernorm": True,
        }
    )

    return mapped_config


@register_mapper("qwen3_moe")
def qwen3_moe_config_mapper(hf_config: PretrainedConfig) -> MegatronModelConfig:
    mapped_config = _map_common_configs(hf_config)
    mapped_config.transformer_config.update(
        {
            "add_bias_linear": False,
            "add_qkv_bias": hf_config.attention_bias,
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "num_moe_experts": hf_config.num_experts,
            "moe_aux_loss_coeff": _to_n_args(hf_config.router_aux_loss_coef),
            "moe_router_load_balancing_type": _to_n_args("none"),  # turn off aux_loss as it hurts perf in RL
            "moe_router_score_function": "softmax",
            "moe_router_pre_softmax": False,
            "qk_layernorm": True,
        }
    )

    return mapped_config


@register_mapper("glm4_moe")
def glm4_moe_config_mapper(hf_config: PretrainedConfig) -> MegatronModelConfig:
    moe_layer_freq = [1] * hf_config.num_hidden_layers
    for i in range(min(hf_config.first_k_dense_replace, hf_config.num_hidden_layers)):
        moe_layer_freq[i] = 0

    mapped_config = _map_common_configs(hf_config)
    mapped_config.transformer_config.update(
        {
            "add_bias_linear": False,
            "qk_layernorm": hf_config.use_qk_norm,
            "add_qkv_bias": hf_config.attention_bias,
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "moe_router_topk_scaling_factor": hf_config.routed_scaling_factor,
            "moe_router_dtype": "fp32",
            "num_moe_experts": hf_config.num_experts,
            "moe_router_enable_expert_bias": True,
            "moe_layer_freq": moe_layer_freq,
            "moe_router_bias_update_rate": 0.0,
            "moe_aux_loss_coeff": _to_n_args(hf_config.router_aux_loss_coef),
            "moe_router_load_balancing_type": _to_n_args("seq_aux_loss"),
            "moe_router_score_function": "sigmoid",
            "rotary_percent": hf_config.partial_rotary_factor,
        }
    )

    return mapped_config
