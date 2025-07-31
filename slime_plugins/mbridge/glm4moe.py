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
            # post_mlp_layernorm=True,
            # post_self_attn_layernorm=True,
            rotary_interleaved=True,
        )
