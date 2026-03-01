import re
import torch


def convert_qwen3vl_to_hf(args, name, param):
    if name == "module.module.language_model.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.language_model.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.language_model.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups
    
    decoder_layers_pattern = r"module\.module\.language_model\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.language_model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        
        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight", param)]
        
    # patch embed / pos embed
    vision_prefix_table = {
        "module.module.vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
        "module.module.vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
        "module.module.vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
        "module.module.vision_model.merger.norm.weight": "model.visual.merger.norm.weight",
        "module.module.vision_model.merger.norm.bias": "model.visual.merger.norm.bias",
        "module.module.vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
        "module.module.vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
        "module.module.vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
        "module.module.vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
    }
    if name in vision_prefix_table:
        return [(vision_prefix_table[name], param)]

    # deepstack_merger_list
    deepstack_merger_pattern = r"module\.module\.vision_model\.deepstack_merger_list\.(\d+)\.(.+)"
    deepstack_match = re.match(deepstack_merger_pattern, name)
    if deepstack_match:
        idx, rest = deepstack_match.groups()
        return [(f"model.visual.deepstack_merger_list.{idx}.{rest}", param)]

    # vision transformer blocks
    vision_model_block_pattern = r"module\.module\.vision_model\.blocks\.(\d+)\.(.+)"
    vision_model_match = re.match(vision_model_block_pattern, name)
    if vision_model_match:
        block_idx, rest = vision_model_match.groups()
        return [(f"model.visual.blocks.{block_idx}.{rest}", param)]

    raise ValueError(f"Unknown parameter name: {name}")