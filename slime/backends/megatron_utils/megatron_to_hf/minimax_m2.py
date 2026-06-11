import re

import torch


def convert_minimax_m2_to_hf(args, name, param):
    """Convert Megatron parameter names/tensors to HuggingFace format for MiniMax-M2.5.

    HF uses `block_sparse_moe` prefix with expert naming w1(gate)/w2(down)/w3(up).
    Custom SelfAttention uses `q_norm`/`k_norm` (not `q_layernorm`/`k_layernorm`).
    """
    # Direct mappings
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # MoE experts: linear_fc1 -> w1 (gate) + w3 (up), linear_fc2 -> w2 (down)
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight", gate_weight),
                    (f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [
                    (f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight", param),
                ]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # Attention: o_proj
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]

        # Attention: fused QKV -> split into Q/K/V (GQA: 48 heads, 8 kv heads)
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

        # Input layernorm
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]

        # QK Norm (custom attention uses q_norm/k_norm, NOT q_layernorm/k_layernorm)
        elif rest == "self_attention.q_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

        # Post-attention layernorm
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # Router
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.block_sparse_moe.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.block_sparse_moe.e_score_correction_bias", param)]

    raise ValueError(f"Unknown parameter name: {name}")
