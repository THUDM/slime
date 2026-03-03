from mbridge.core import register_model
from mbridge.models import Qwen2Bridge, Qwen2MoEBridge


@register_model("gpt_oss")
class GptOssBridge(Qwen2MoEBridge):
    """
    Bridge implementation for GPT-OSS models.

    Handles weight conversion between preprocessed GPT-OSS HF format
    (BF16 per-expert) and Megatron-Core.

    Key differences from Qwen2MoE:
    - All layers are MoE (no dense layers, no shared expert)
    - Has learnable softmax offset (sinks)
    - Has attention bias (q/k/v/o_proj.bias)
    - Has router bias
    - Has expert bias (gate/up/down_proj.bias)
    """

    _ATTENTION_MAPPING = {
        **(Qwen2Bridge._ATTENTION_MAPPING),
        "self_attention.linear_proj.bias": ["model.layers.{layer_number}.self_attn.o_proj.bias"],
        "self_attention.core_attention.softmax_offset": ["model.layers.{layer_number}.self_attn.sinks"],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm.weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.router.weight"],
        "mlp.router.bias": ["model.layers.{layer_number}.mlp.router.bias"],
        # Expert biases (must be checked before weight patterns)
        "mlp.experts.linear_fc1.bias": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.bias",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.bias",
        ],
        "mlp.experts.linear_fc2.bias": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.bias",
        ],
        # Expert weights
        "mlp.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight",
        ],
    }

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        elif "pre_mlp_layernorm" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        """Override to handle expert bias names correctly.

        Base class extracts expert_id by splitting on 'weight', which fails
        for bias parameters. We extract expert_id from after 'bias' as well.
        """
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    # Extract expert_id from end of name (after weight/bias)
                    if "weight" in name.split(".")[-1]:
                        expert_id = name.split("weight")[-1]
                    elif "bias" in name.split(".")[-1]:
                        expert_id = name.split("bias")[-1]
                    else:
                        raise ValueError(f"Cannot extract expert_id from: {name}")
                    convert_names.extend(
                        [x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names]
                    )
                else:
                    convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported MLP parameter name: {name}")
        return convert_names

    def _build_config(self):
        return self._build_base_config(
            use_cpu_initialization=False,
            # MoE
            moe_ffn_hidden_size=self.hf_config.intermediate_size,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_local_experts,
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            moe_router_pre_softmax=False,
            # GPT-OSS specific
            add_qkv_bias=True,
            add_bias_linear=True,
            qk_layernorm=False,
            persist_layer_norm=True,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            # SWA
            window_size=(self.hf_config.sliding_window, 0),
            window_attn_skip_freq=2,
            # Learnable softmax
            softmax_type="learnable",
            # Quick GeGLU
            glu_linear_offset=1.0,
            activation_func_clamp_value=getattr(self.hf_config, "swiglu_limit", 7.0),
            # RoPE
            rotary_interleaved=False,
        )

    def _get_transformer_layer_spec(self):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

        return get_gpt_layer_with_transformer_engine_spec()
