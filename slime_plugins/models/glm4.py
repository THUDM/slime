from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


def get_glm_spec(args, config, vp_stage):
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        multi_latent_attention=args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
        post_self_attn_layernorm=args.post_self_attn_layernorm,
        post_mlp_layernorm=args.post_mlp_layernorm,
    )
    return transformer_layer_spec
