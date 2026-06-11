from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


@register_model("minimax_m2")
class MiniMaxM2Bridge(Qwen2MoEBridge):
    """
    Bridge for MiniMax-M2.5 (229B MoE).

    Key differences from standard Qwen2MoE:
    - HF uses `block_sparse_moe` prefix (not `mlp`) with expert naming w1/w2/w3
    - Full-dimension QK Norm: custom SelfAttention uses `q_norm`/`k_norm` fields
      (NOT the default `q_layernorm`/`k_layernorm`), so state_dict key is
      `self_attention.q_norm.weight` / `self_attention.k_norm.weight`
    - Sigmoid router with e_score_correction_bias
    - Partial RoPE (rotary_percent=0.5)
    - No shared experts
    """

    _ATTENTION_MAPPING = {
        **Qwen2MoEBridge._ATTENTION_MAPPING,
        # Override QK norm: custom MiniMaxM2SelfAttention uses self.q_norm / self.k_norm
        # instead of the default self.q_layernorm / self.k_layernorm
        "self_attention.q_norm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
        "self_attention.k_norm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.router.weight": ["model.layers.{layer_number}.block_sparse_moe.gate.weight"],
        "mlp.router.expert_bias": ["model.layers.{layer_number}.block_sparse_moe.e_score_correction_bias"],
        "mlp.experts.linear_fc1": [
            "model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w1.weight",  # gate_proj
            "model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w3.weight",  # up_proj
        ],
        "mlp.experts.linear_fc2": [
            "model.layers.{layer_number}.block_sparse_moe.experts.{expert_id}.w2.weight",  # down_proj
        ],
    }

    def _build_config(self):
        return self._build_base_config(
            use_cpu_initialization=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # MoE config
            moe_ffn_hidden_size=self.hf_config.intermediate_size,
            moe_router_topk=self.hf_config.num_experts_per_tok,
            num_moe_experts=self.hf_config.num_local_experts,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=True,
            moe_router_dtype="fp32",
            moe_grouped_gemm=True,
            moe_router_load_balancing_type="none",
            # Attention config
            qk_layernorm=True,
            rotary_percent=0.5,
            add_qkv_bias=False,
            add_bias_linear=False,
            rotary_interleaved=False,
        )
