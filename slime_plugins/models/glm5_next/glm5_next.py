"""GLM-5.3 Flash spec for upstream Megatron, following the GLM5 plugin API."""

import copy

import torch
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_mlp_module_spec_for_backend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, get_transformer_layer_offset

from slime.utils.model_config import load_hf_config

from .attention import KimiDeltaAttention, NopeDSAttention
from .ops import ClampedSwiGLU, mhc_post, mhc_pre


class Glm5NextSpecProvider(TESpecProvider):
    def activation_func(self):
        return ClampedSwiGLU


class Glm5NextLayer(TransformerLayer):
    def __init__(self, *args, hf_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_config = hf_config
        n = hf_config.hc_mult
        for kind in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{kind}_fn", torch.nn.Parameter(torch.zeros(n * (n + 2), n * self.config.hidden_size))
            )
            self.register_parameter(
                f"hc_{kind}_scale", torch.nn.Parameter(torch.full((3,), 0.01, dtype=torch.float32))
            )
            self.register_parameter(
                f"hc_{kind}_base", torch.nn.Parameter(torch.zeros(n * (n + 2), dtype=torch.float32))
            )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for kind in ("attn", "ffn"):
            for suffix in ("scale", "base"):
                parameter = getattr(self, f"hc_{kind}_{suffix}")
                parameter.data = parameter.data.float()
        return self

    def forward(
        self, hidden_states, attention_mask=None, context=None, packed_seq_params=None, padding_mask=None, **kwargs
    ):
        config = self.hf_config
        if self.layer_number == 1:
            hidden_states = (
                hidden_states.unsqueeze(-2)
                .expand(*hidden_states.shape[:-1], config.hc_mult, hidden_states.shape[-1])
                .flatten(-2)
                .contiguous()
            )
        for kind, norm, module in (
            ("attn", self.input_layernorm, self.self_attention),
            ("ffn", self.pre_mlp_layernorm, self.mlp),
        ):
            x, post, residual = mhc_pre(
                hidden_states,
                getattr(self, f"hc_{kind}_fn"),
                getattr(self, f"hc_{kind}_scale"),
                getattr(self, f"hc_{kind}_base"),
                config.hc_mult,
                config.rms_norm_eps,
                config.hc_eps,
                config.hc_sinkhorn_iters,
            )
            x = norm(x)
            if kind == "attn":
                output, bias = module(x, packed_seq_params=packed_seq_params)
            else:
                output, bias = module(x, padding_mask=padding_mask)
            if bias is not None:
                output = output + bias
            hidden_states = mhc_post(output, hidden_states, post, residual)
        if self.layer_number == self.config.num_layers:
            hidden_states = hidden_states.unflatten(-1, (config.hc_mult, -1)).float().mean(-2).to(hidden_states.dtype)
        return hidden_states, context


def get_glm5_next_spec(args, config, vp_stage):
    hf_config = load_hf_config(args.hf_checkpoint, trust_remote_code=True).text_config
    if config.tensor_model_parallel_size != 1:
        raise ValueError("GLM5-next native attention requires TP=1; use CP, PP and EP to shard the 300B model")
    if config.context_parallel_size > 1 and not args.allgather_cp:
        raise ValueError("GLM5-next CP>1 requires --allgather-cp for contiguous KDA sequence partitions")
    config.activation_func_clamp_value = hf_config.swiglu_limit
    config.bias_activation_fusion = False
    config.use_te_activation_func = True
    config.variable_seq_lengths = True  # Pipeline payloads carry all mHC streams.
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    kda_layers = set(hf_config.linear_attn_config["kda_layers"])
    backend = Glm5NextSpecProvider()
    for index, layer in enumerate(spec.layer_specs):
        layer = copy.deepcopy(layer)
        layer.module = Glm5NextLayer
        layer.params = {**(layer.params or {}), "hf_config": hf_config}
        layer.submodules.input_layernorm = backend.layer_norm()
        layer.submodules.self_attention = ModuleSpec(
            module=KimiDeltaAttention if index + offset in kda_layers else NopeDSAttention,
            params={"hf_config": hf_config},
        )
        layer.submodules.mlp = get_mlp_module_spec_for_backend(
            backend,
            num_experts=None if index + offset < hf_config.first_k_dense_replace else config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            use_te_activation_func=True,
        )
        spec.layer_specs[index] = layer
    return spec
