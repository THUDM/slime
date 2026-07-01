import re

import torch

_expert_tensor_cache = {}


def _interleave_gate_up(param):
    gate, up = param.chunk(2, dim=0)
    return torch.stack([gate, up], dim=1).reshape(-1, *param.shape[1:]).contiguous()


def _collect_fused_expert_tensor(args, hf_name, expert_idx, param):
    num_experts = getattr(args, "num_experts", None)
    if num_experts is None:
        raise ValueError("args.num_experts is required to fuse GPT-OSS expert tensors")

    num_experts = int(num_experts)
    expert_idx = int(expert_idx)
    if expert_idx < 0 or expert_idx >= num_experts:
        raise ValueError(f"GPT-OSS expert index {expert_idx} is out of range for {num_experts} experts")

    bucket = _expert_tensor_cache.setdefault(hf_name, {})
    if expert_idx in bucket:
        raise ValueError(f"Duplicate GPT-OSS expert tensor for {hf_name} expert {expert_idx}")
    bucket[expert_idx] = param

    if len(bucket) != num_experts:
        return []

    missing = [i for i in range(num_experts) if i not in bucket]
    if missing:
        raise ValueError(f"Missing GPT-OSS expert tensors for {hf_name}: {missing}")

    fused = torch.stack([bucket[i] for i in range(num_experts)], dim=0).contiguous()
    del _expert_tensor_cache[hf_name]
    return [(hf_name, fused)]


def convert_gpt_oss_to_hf(args, name, param):
    """Convert Megatron GPT-OSS parameter names to HF format for weight update to SGLang."""

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # Expert weights
        expert_pattern = r"mlp\.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                param = _interleave_gate_up(param).transpose(0, 1).contiguous()
                return _collect_fused_expert_tensor(
                    args,
                    f"model.layers.{layer_idx}.mlp.experts.gate_up_proj",
                    expert_idx,
                    param,
                )
            elif rest == "linear_fc2":
                return _collect_fused_expert_tensor(
                    args,
                    f"model.layers.{layer_idx}.mlp.experts.down_proj",
                    expert_idx,
                    param.transpose(0, 1).contiguous(),
                )
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # Expert biases
        expert_bias_pattern = r"mlp\.experts\.(.+)\.bias(\d+)"
        match = re.match(expert_bias_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                return _collect_fused_expert_tensor(
                    args,
                    f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_bias",
                    expert_idx,
                    _interleave_gate_up(param),
                )
            elif rest == "linear_fc2":
                return _collect_fused_expert_tensor(
                    args,
                    f"model.layers.{layer_idx}.mlp.experts.down_proj_bias",
                    expert_idx,
                    param.contiguous(),
                )
            else:
                raise ValueError(f"Unknown expert bias parameter name: {name}")

        # Attention
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_proj.bias":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.bias", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        # Learnable softmax offset (sinks)
        elif rest == "self_attention.core_attention.softmax_offset":
            return [(f"model.layers.{layer_idx}.self_attn.sinks", param)]
        # Layer norms
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        # Router
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.router.weight", param)]
        elif rest == "mlp.router.bias":
            return [(f"model.layers.{layer_idx}.mlp.router.bias", param)]

    raise ValueError(f"Unknown parameter name: {name}")
