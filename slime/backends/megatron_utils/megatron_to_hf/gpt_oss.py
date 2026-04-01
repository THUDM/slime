import re

import torch


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
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # Expert biases
        expert_bias_pattern = r"mlp\.experts\.(.+)\.bias(\d+)"
        match = re.match(expert_bias_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_bias, up_bias = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.bias", gate_bias),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.bias", up_bias),
                ]
            elif rest == "linear_fc2":
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.bias", param),
                ]
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
