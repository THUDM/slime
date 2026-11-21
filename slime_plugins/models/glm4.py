import inspect

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_layer import TransformerLayer


class GLMTransformerLayer(TransformerLayer):
    def __init__(self, *args, post_self_attn_layernorm=False, post_mlp_layernorm=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Older Megatron patches provide these modules and apply the norms themselves.
        if hasattr(self, "post_self_attn_layernorm"):
            return
        if post_self_attn_layernorm:
            self.post_self_attn_layernorm = TENorm(
                config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
            )

            def attention_norm(module, inputs, output):
                return self.post_self_attn_layernorm(output[0]), output[1]

            self.self_attention.register_forward_hook(attention_norm)
        if post_mlp_layernorm:
            self.post_mlp_layernorm = TENorm(
                config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
            )

            def mlp_norm(module, inputs, output):
                return self.post_mlp_layernorm(output[0]), output[1]

            self.mlp.register_forward_hook(mlp_norm)


def add_post_layernorms(spec, args):
    if hasattr(spec, "layer_specs"):
        for layer_spec in spec.layer_specs:
            add_post_layernorms(layer_spec, args)
    elif spec.module is TransformerLayer:
        spec.module = GLMTransformerLayer
        spec.params.update(
            post_self_attn_layernorm=args.post_self_attn_layernorm,
            post_mlp_layernorm=args.post_mlp_layernorm,
        )


def get_glm_spec(args, config, vp_stage):
    kwargs = dict(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        multi_latent_attention=args.multi_latent_attention,
    )
    parameters = inspect.signature(get_gpt_layer_with_transformer_engine_spec).parameters
    for name in ("moe_use_legacy_grouped_gemm", "post_self_attn_layernorm", "post_mlp_layernorm"):
        if name in parameters:
            kwargs[name] = getattr(args, name)
    return get_gpt_layer_with_transformer_engine_spec(**kwargs)
