from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module


class MiniMaxM2SelfAttention(SelfAttention):
    """Custom SelfAttention for MiniMax-M2.5 with full-dimension QK Norm.

    MiniMax-M2.5 applies RMSNorm over all heads concatenated:
      Q: num_attention_heads * head_dim = 48 * 128 = 6144
      K: num_kv_heads * head_dim       =  8 * 128 = 1024
    instead of Megatron Core's default per-head norm (head_dim=128).

    With tensor parallelism, the norm requires TP gather -> norm -> TP scatter.
    """

    def __init__(self, config, submodules, *args, **kwargs):
        # Save real layernorm specs, replace with IdentityOp so Megatron
        # does not create per-head norms
        q_layernorm = submodules.q_layernorm
        k_layernorm = submodules.k_layernorm
        submodules.q_layernorm = IdentityOp
        submodules.k_layernorm = IdentityOp

        super().__init__(config, submodules, *args, **kwargs)

        # Restore submodules in case they are reused elsewhere
        submodules.q_layernorm = q_layernorm
        submodules.k_layernorm = k_layernorm

        # Create full-dimension norms
        self.q_norm = build_module(
            q_layernorm,
            hidden_size=self.hidden_size_per_attention_head * config.num_attention_heads,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.k_norm = build_module(
            k_layernorm,
            hidden_size=self.hidden_size_per_attention_head * config.num_query_groups,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, *args, **kwargs):
        query, key, value = super().get_query_key_value_tensors(hidden_states, key_value_states, *args, **kwargs)
        # query: [sq, b, num_heads_local, head_dim]
        # key:   [sq, b, num_kv_heads_local, head_dim]

        # Merge head dims: [sq, b, num_heads_local * head_dim]
        query = query.reshape(*query.shape[:-2], -1)
        key = key.reshape(*key.shape[:-2], -1)

        # TP gather -> full-dimension norm -> TP scatter
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            query = gather_from_tensor_model_parallel_region(query)
            key = gather_from_tensor_model_parallel_region(key)

        query = self.q_norm(query)
        key = self.k_norm(key)

        if tp_size > 1:
            query = scatter_to_tensor_model_parallel_region(query)
            key = scatter_to_tensor_model_parallel_region(key)

        # Reshape back: [sq, b, num_heads_local, head_dim]
        query = query.view(*query.shape[:2], -1, self.hidden_size_per_attention_head)
        key = key.view(*key.shape[:2], -1, self.hidden_size_per_attention_head)

        return query, key, value


def get_minimax_m2_layer_spec(args, config, vp_stage=None):
    """Build layer spec for MiniMax-M2.5, replacing SelfAttention with the custom version.

    Used via: --spec "slime_plugins.models.minimax_m2" "get_minimax_m2_layer_spec"
    """
    kwargs = {"use_transformer_engine": args.transformer_impl == "transformer_engine"}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    spec = get_gpt_decoder_block_spec(config, **kwargs)

    for layer_spec in spec.layer_specs:
        layer_spec.submodules.self_attention.module = MiniMaxM2SelfAttention

    return spec
