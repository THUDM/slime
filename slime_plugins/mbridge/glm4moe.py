import re

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge, Qwen2Bridge


@register_model("glm4_moe")
class GLM4MoEBridge(Qwen2MoEBridge):
    """
    Bridge implementation for Qwen2 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for Qwen2 models, handling the conversion between
    Hugging Face Qwen2 format and Megatron-Core.
    """

    _MLP_MAPPING = {
        **(Qwen2MoEBridge._MLP_MAPPING),
        **(Qwen2Bridge._MLP_MAPPING),
        "mlp.router.expert_bias": ["model.layers.{layer_number}.mlp.gate.e_score_correction_bias"],
        "shared_experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight",
            "model.layers.{layer_number}.mlp.shared_experts.up_proj.weight",
        ],
        "shared_experts.linear_fc2.weight": ["model.layers.{layer_number}.mlp.shared_experts.down_proj.weight"],
    }

    _MTP_MAPPING = {
        "enorm.weight": ["model.layers.{layer_number}.enorm.weight"],
        "hnorm.weight": ["model.layers.{layer_number}.hnorm.weight"],
        "eh_proj.weight": ["model.layers.{layer_number}.eh_proj.weight"],
        "final_layernorm.weight": ["model.layers.{layer_number}.shared_head.norm.weight"],
    }

    def _weight_name_mapping_mtp(self, name: str, num_layers: int) -> str:
        convert_names = []
        for keyword, mapping_names in self._MTP_MAPPING.items():
            if keyword in name:
                convert_names.extend(
                    [x.format(layer_number=num_layers) for x in mapping_names]
                )
                break
            elif "mlp" in name:
                mtp_layer_index = int(re.findall(r"mtp\.layers\.(\d+)\.", name)[0])
                name_ = re.sub(r"^mtp\.layers.\d+.transformer_layer", f"model.layers.{num_layers+mtp_layer_index}", name)
                convert_names = self._weight_name_mapping_mlp(name_)
                break
            elif "self_attention" in name:
                mtp_layer_index = int(re.findall(r"mtp\.layers.(\d+)\.", name)[0])
                name_ = re.sub(r"^mtp\.layers.\d+.transformer_layer", f"model.layers.{num_layers+mtp_layer_index}", name)
                convert_names = self._weight_name_mapping_attention(name_)
                break

        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert (
            "_extra_state" not in mcore_weights_name
        ), "extra_state should not be loaded"
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if mcore_weights_name in direct_name_mapping:
            return [direct_name_mapping[mcore_weights_name]]

        if "mtp" in mcore_weights_name:  # first check mtp
            return self._weight_name_mapping_mtp(mcore_weights_name, self.config.num_layers)
        elif "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"Unsupported parameter name: {mcore_weights_name}"
            )

    def _build_config(self):
        """
        Build the configuration for Qwen2 models.

        Configures Qwen2-specific parameters such as QKV bias settings and
        layer normalization options.

        Returns:
            TransformerConfig: Configuration object for Qwen2 models
        """
        return self._build_base_config(
            use_cpu_initialization=False,
            # MoE specific
            moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
            moe_router_bias_update_rate=0.001,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.n_routed_experts,
            # moe_router_load_balancing_type="aux_loss",
            moe_router_load_balancing_type="none",  # default None for RL
            moe_grouped_gemm=True,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=True,
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # GLM specific
            qk_layernorm=self.hf_config.use_qk_norm,
            add_qkv_bias=True,
            add_bias_linear=False,
            # post_mlp_layernorm=True,
            # post_self_attn_layernorm=True,
            rotary_interleaved=True,
        )
