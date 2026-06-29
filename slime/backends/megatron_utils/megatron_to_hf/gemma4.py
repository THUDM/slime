import re
import torch

_config_cache: dict[str, dict] = {}

# Per-layer buffers for stacked expert tensors. sglang's Gemma4 loader expects
# `experts.gate_up_proj` as a single 3D tensor of shape [E, 2I, H] and
# `experts.down_proj` as [E, H, I] - it walks all experts inside the loader
# and would silently drop per-expert 2D inputs. We accumulate expert tensors
# as they stream through and emit the stacked form once all num_experts arrive.
_expert_buffers: dict = {}


def reset_expert_buffers() -> None:
    """Drop any partial expert buckets. Callers that drive the converter from a
    long-lived process (tests, repeated conversions) should invoke this between
    runs so an interrupted prior conversion doesn't leak its partial state."""
    _expert_buffers.clear()


def _get_config(args):
    checkpoint = args.hf_checkpoint
    if checkpoint not in _config_cache:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        hf_text = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        _config_cache[checkpoint] = {
            "global_attn_layers": {i for i, t in enumerate(hf_text.layer_types) if t == "full_attention"},
            "local_head_dim": hf_text.head_dim,
            "global_head_dim": hf_text.global_head_dim,
            "num_attention_heads": hf_text.num_attention_heads,
            "local_num_kv_heads": hf_text.num_key_value_heads,
            "global_num_kv_heads": hf_text.num_global_key_value_heads,
            "hidden_size": hf_text.hidden_size,
            "num_experts": getattr(hf_text, "num_experts", 0),
        }
    return _config_cache[checkpoint]


def convert_gemma4_to_hf(args, name, param):
    cfg = _get_config(args)
    prefix = "model.language_model."

    if name == "module.module.embedding.word_embeddings.weight":
        return [(f"{prefix}embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [(f"{prefix}embed_tokens.weight", param)]  # tied embeddings
    if name == "module.module.decoder.final_layernorm.weight":
        return [(f"{prefix}norm.weight", param)]

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if match:
        layer_idx = int(match.group(1))
        rest = match.group(2)
        L = f"{prefix}layers.{layer_idx}"
        is_global = layer_idx in cfg["global_attn_layers"]

        if rest == "self_attention.linear_proj.weight":
            return [(f"{L}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            if is_global:
                head_dim = cfg["global_head_dim"]
                num_kv_heads = cfg["global_num_kv_heads"]
            else:
                head_dim = cfg["local_head_dim"]
                num_kv_heads = cfg["local_num_kv_heads"]

            q_heads_per_kv = cfg["num_attention_heads"] // num_kv_heads
            hidden_size = cfg["hidden_size"]
            param = param.view(num_kv_heads, (q_heads_per_kv + 2) * head_dim, hidden_size)
            q_dim = q_heads_per_kv * head_dim
            q_param = param[:, :q_dim, :].reshape(-1, hidden_size)
            k_param = param[:, q_dim : q_dim + head_dim, :].reshape(-1, hidden_size)

            if is_global:
                return [
                    (f"{L}.self_attn.q_proj.weight", q_param),
                    (f"{L}.self_attn.k_proj.weight", k_param),
                ]
            else:
                v_param = param[:, q_dim + head_dim :, :].reshape(-1, hidden_size)
                return [
                    (f"{L}.self_attn.q_proj.weight", q_param),
                    (f"{L}.self_attn.k_proj.weight", k_param),
                    (f"{L}.self_attn.v_proj.weight", v_param),
                ]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{L}.input_layernorm.weight", param)]
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{L}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{L}.self_attn.k_norm.weight", param)]
        elif rest in ("mlp.linear_fc1.weight", "dense_mlp.linear_fc1.weight"):
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{L}.mlp.gate_proj.weight", gate_weight),
                (f"{L}.mlp.up_proj.weight", up_weight),
            ]
        elif rest in ("mlp.linear_fc2.weight", "dense_mlp.linear_fc2.weight"):
            return [(f"{L}.mlp.down_proj.weight", param)]
        elif rest in ("mlp.linear_fc1.layer_norm_weight", "dense_mlp.linear_fc1.layer_norm_weight"):
            return [(f"{L}.pre_feedforward_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"{L}.pre_feedforward_layernorm.weight", param)]
        elif rest == "post_attention_layernorm.weight":
            return [(f"{L}.post_attention_layernorm.weight", param)]
        elif rest == "post_feedforward_layernorm.weight":
            return [(f"{L}.post_feedforward_layernorm.weight", param)]
        elif rest == "layer_scalar":
            return [(f"{L}.layer_scalar", param)]
        elif rest == "mlp.router.proj.weight":
            return [(f"{L}.router.proj.weight", param)]
        elif rest == "mlp.router.scale":
            return [(f"{L}.router.scale", param)]
        elif rest == "mlp.router.per_expert_scale":
            return [(f"{L}.router.per_expert_scale", param)]
        else:
            expert_match = re.match(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
            if expert_match:
                fc, expert_idx = expert_match.group(1), int(expert_match.group(2))
                return _buffer_expert_and_maybe_flush(
                    layer_idx,
                    fc,
                    expert_idx,
                    param,
                    L,
                    num_experts=cfg["num_experts"],
                )

        if rest == "pre_feedforward_layernorm_2.weight":
            return [(f"{L}.pre_feedforward_layernorm_2.weight", param)]
        elif rest == "mlp.pre_feedforward_layernorm_2.weight":
            return [(f"{L}.pre_feedforward_layernorm_2.weight", param)]
        elif rest == "post_feedforward_layernorm_2.weight":
            return [(f"{L}.post_feedforward_layernorm_2.weight", param)]
        elif rest == "post_feedforward_layernorm_1.weight":
            return [(f"{L}.post_feedforward_layernorm_1.weight", param)]

    raise ValueError(f"Unknown Gemma4 parameter name: {name}")


def _buffer_expert_and_maybe_flush(layer_idx, fc, expert_idx, param, L_prefix, num_experts):
    """Buffer per-expert tensor; emit stacked 3D `experts.gate_up_proj` / `experts.down_proj`
    once the bucket for (layer, fc) has all `num_experts` experts."""
    assert (
        num_experts and num_experts > 0
    ), f"num_experts must be known for MoE layer expert conversion, got {num_experts}"
    key = (layer_idx, fc)
    bucket = _expert_buffers.setdefault(key, {})
    bucket[expert_idx] = param

    if len(bucket) < num_experts:
        return []

    ordered = [bucket[i] for i in range(num_experts)]
    stacked = torch.stack(ordered, dim=0).contiguous()
    del _expert_buffers[key]

    if fc == "1":
        return [(f"{L_prefix}.experts.gate_up_proj", stacked)]
    else:
        return [(f"{L_prefix}.experts.down_proj", stacked)]
