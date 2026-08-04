from __future__ import annotations

import re

import torch

from .common import SafetensorReader, strip_mcore_wrappers


def _merge_qkv(reader: SafetensorReader, prefix: str, text_config, suffix: str) -> torch.Tensor:
    q = reader.get_tensor(f"{prefix}.q_proj.{suffix}")
    k = reader.get_tensor(f"{prefix}.k_proj.{suffix}")
    v = reader.get_tensor(f"{prefix}.v_proj.{suffix}")
    num_groups = text_config.num_key_value_heads
    queries_per_group = text_config.num_attention_heads // num_groups
    head_dim = text_config.head_dim

    trailing_shape = q.shape[1:]
    q = q.reshape(num_groups, queries_per_group, 2, head_dim, *trailing_shape).transpose(1, 2)
    q = q.flatten(1, 3)
    k = k.reshape(num_groups, head_dim, *trailing_shape)
    v = v.reshape(num_groups, head_dim, *trailing_shape)
    return torch.cat((q, k, v), dim=1).reshape(-1, *trailing_shape).contiguous()


def qwen3_5_hf_tensor(name: str, reader: SafetensorReader, hf_config) -> torch.Tensor:
    """Return the full, unsharded MCore tensor for a Qwen3.5 parameter name."""

    name = strip_mcore_wrappers(name)
    if name.startswith("model.visual."):
        return reader.get_tensor(name)
    name = name.removeprefix("language_model.")

    text_config = getattr(hf_config, "text_config", hf_config)
    direct_mapping = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": (
            "model.language_model.embed_tokens.weight"
            if getattr(hf_config, "tie_word_embeddings", False) or getattr(text_config, "tie_word_embeddings", False)
            else "lm_head.weight"
        ),
    }
    if name in direct_mapping:
        return reader.get_tensor(direct_mapping[name])

    mtp_match = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", name)
    if mtp_match:
        is_mtp = True
        mtp_layer, rest = mtp_match.groups()
        direct_mtp = {
            "eh_proj.weight": "mtp.fc.weight",
            "enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "final_layernorm.weight": "mtp.norm.weight",
        }
        if rest in direct_mtp:
            return reader.get_tensor(direct_mtp[rest])
        rest = rest.removeprefix("transformer_layer.")
        name = f"decoder.layers.{mtp_layer}." + rest
        hf_layer_prefix = f"mtp.layers.{mtp_layer}"
    else:
        is_mtp = False
        layer_match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
        if not layer_match:
            raise KeyError(f"Unsupported Qwen3.5 Megatron parameter {name!r}")
        layer_idx, rest = layer_match.groups()
        hf_layer_prefix = f"model.language_model.layers.{layer_idx}"

    if rest.startswith("self_attention.linear_attn."):
        suffix = rest.removeprefix("self_attention.")
        return reader.get_tensor(f"{hf_layer_prefix}.{suffix}")
    if rest == "self_attention.input_layernorm.weight":
        return reader.get_tensor(f"{hf_layer_prefix}.input_layernorm.weight")
    if rest == "self_attention.linear_proj.weight":
        return reader.get_tensor(f"{hf_layer_prefix}.self_attn.o_proj.weight")
    if rest == "self_attention.linear_qkv.weight":
        return _merge_qkv(reader, f"{hf_layer_prefix}.self_attn", text_config, "weight")
    if rest == "self_attention.linear_qkv.bias":
        return _merge_qkv(reader, f"{hf_layer_prefix}.self_attn", text_config, "bias")
    if rest == "self_attention.linear_qkv.layer_norm_weight":
        return reader.get_tensor(f"{hf_layer_prefix}.input_layernorm.weight")
    if rest == "self_attention.q_layernorm.weight":
        return reader.get_tensor(f"{hf_layer_prefix}.self_attn.q_norm.weight")
    if rest == "self_attention.k_layernorm.weight":
        return reader.get_tensor(f"{hf_layer_prefix}.self_attn.k_norm.weight")

    if rest in {"mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight"}:
        return reader.get_tensor(f"{hf_layer_prefix}.post_attention_layernorm.weight")
    if rest == "mlp.linear_fc1.weight":
        gate = reader.get_tensor(f"{hf_layer_prefix}.mlp.gate_proj.weight")
        up = reader.get_tensor(f"{hf_layer_prefix}.mlp.up_proj.weight")
        return torch.cat((gate, up), dim=0)
    if rest == "mlp.linear_fc2.weight":
        return reader.get_tensor(f"{hf_layer_prefix}.mlp.down_proj.weight")
    if rest == "mlp.router.weight":
        return reader.get_tensor(f"{hf_layer_prefix}.mlp.gate.weight")
    if rest == "mlp.router.expert_bias":
        return reader.get_tensor(f"{hf_layer_prefix}.mlp.gate.e_score_correction_bias")

    expert_match = re.fullmatch(r"mlp\.experts\.linear_fc([12])(?:\.weight)?(\d+)?", rest)
    if expert_match:
        projection, expert_idx = expert_match.groups()
        if is_mtp and expert_idx is not None:
            prefix = f"{hf_layer_prefix}.mlp.experts.{expert_idx}"
            if projection == "1":
                return torch.cat(
                    (
                        reader.get_tensor(f"{prefix}.gate_proj.weight"),
                        reader.get_tensor(f"{prefix}.up_proj.weight"),
                    ),
                    dim=0,
                )
            return reader.get_tensor(f"{prefix}.down_proj.weight")
        suffix = "gate_up_proj" if projection == "1" else "down_proj"
        tensor = reader.get_tensor(f"{hf_layer_prefix}.mlp.experts.{suffix}")
        return tensor if expert_idx is None else tensor[int(expert_idx)].contiguous()

    shared_mapping = {
        "mlp.shared_experts.linear_fc1.weight": ("gate_proj.weight", "up_proj.weight"),
        "mlp.shared_experts.linear_fc2.weight": ("down_proj.weight",),
        "mlp.shared_experts.gate_weight": ("../shared_expert_gate.weight",),
    }
    if rest in shared_mapping:
        tensors = []
        for suffix in shared_mapping[rest]:
            if suffix.startswith("../"):
                key = f"{hf_layer_prefix}.mlp.{suffix.removeprefix('../')}"
            else:
                key = f"{hf_layer_prefix}.mlp.shared_expert.{suffix}"
            tensors.append(reader.get_tensor(key))
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)

    raise KeyError(f"Unsupported Qwen3.5 Megatron parameter {name!r}")
