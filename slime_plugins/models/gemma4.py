"""Native Megatron Gemma4 transformer layer and config.

Extends the Gemma3 implementation from mbridge with Gemma4-specific features:
- Heterogeneous attention: global layers use head_dim=512, num_kv_heads=4;
  sliding layers use head_dim=256, num_kv_heads=16.
- attention_k_eq_v: global layers reuse K output as V (no v_proj).
- v_norm: RMSNorm without learnable scale applied to V states.
- layer_scalar: buffer multiplied after residual (not learned).
- final_logit_softcapping: applied to output logits in the model wrapper.
- MoE block (26B-A4B): Gemma4's custom router (with per-expert scale) plugged
  into Megatron's MoE infrastructure for proper expert-parallel sharding.
  The router is still custom (see Gemma4Router); dispatching + grouped-GEMM
  come from Megatron's MoELayer + TEGroupedMLP.
"""

import functools
import logging
from dataclasses import dataclass, replace as dc_replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, BaseMoELayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from mbridge.models.gemma3.transformer_config import Gemma3TransformerConfig


# Gemma uses GeGLU, not SwiGLU.
_gelu_tanh = functools.partial(F.gelu, approximate="tanh")


@dataclass
class Gemma4TransformerConfig(Gemma3TransformerConfig):
    """Gemma4-specific config extending Gemma3."""
    # Heterogeneous attention: global layers use different head_dim and num_kv_heads
    global_kv_channels: int = 512
    global_num_query_groups: int = 4
    global_partial_rotary_factor: float = 0.25  # fraction of global head_dim that gets RoPE
    attention_k_eq_v: bool = True  # global layers: V = K (no v_proj)
    enable_moe_block: bool = False  # 26B-A4B MoE variant


class VNorm(nn.Module):
    """RMSNorm without learnable scale, matching Gemma4's v_norm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        return (x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)).to(dtype)


@dataclass
class Gemma4TransformerLayerSubmodules(TransformerLayerSubmodules):
    post_attention_layernorm: ModuleSpec | type = IdentityOp
    post_feedforward_layernorm: ModuleSpec | type = IdentityOp
    # For MoE-enabled variants (26B-A4B), the primary `mlp` submodule is swapped
    # to a Gemma4MoELayer and the original dense MLP moves to `dense_mlp`. This
    # keeps the `.mlp.experts.linear_fc...` naming that mbridge's EP auto-handling
    # expects while preserving Gemma4's dense+MoE-in-parallel structure.
    dense_mlp: ModuleSpec | type = IdentityOp


class Gemma4Router(nn.Module):
    """Gemma4 MoE router.

    The router equation (mirroring HF ``Gemma4TextTopkRouter``) is:

        h_norm   = RMSNorm_no_scale(h)              # VNorm: no learnable scale
        h_scaled = h_norm * scale / sqrt(H)         # learnable per-hidden scale
        logits   = proj(h_scaled)                   # [T, E]
        probs    = softmax(logits, dim=-1)
        top_w, top_i = topk(probs, k=top_k)
        top_w    = top_w / top_w.sum(dim=-1, keepdim=True)   # renormalize
        top_w    = top_w * per_expert_scale[top_i]           # per-expert scale

    The renormalise-then-scale order is load-bearing and must match HF: it
    produces ``top_w.sum() == per_expert_scale.mean_over_selected`` rather
    than a renormalised-back-to-1 distribution. Reversing the order (scale
    first, then renormalise) would cancel ``per_expert_scale``.
    ``test_router_matches_hf_reference_equation`` guards this.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_moe_experts
        self.top_k = config.moe_router_topk
        self.scalar_root_size = self.hidden_size ** -0.5
        self.norm = VNorm(self.hidden_size, eps=config.layernorm_epsilon)
        self.proj = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts))

    def forward(self, hidden_states):
        # hidden_states: [tokens, hidden_size]
        h = self.norm(hidden_states)
        h = h * self.scale * self.scalar_root_size
        logits = self.proj(h)
        probs = torch.softmax(logits, dim=-1)
        top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1)
        # Order matters: renormalize FIRST, then apply per-expert scale. See
        # class docstring.
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_weights, top_k_index

    def set_layer_number(self, layer_number):
        # Megatron's ``MoELayer.set_layer_number`` delegates to
        # ``self.router.set_layer_number``. ``TopKRouter`` uses it for
        # aux-loss scoping / logging; Gemma4Router has no aux loss and
        # doesn't log, so this is a no-op — but the method MUST exist or
        # ``MoELayer.set_layer_number`` raises AttributeError on first call.
        pass


class Gemma4MoELayer(MoELayer):
    """Gemma4 MoE block: Megatron's MoELayer with Gemma4's custom router.

    Megatron's MoELayer hardcodes its own ``TopKRouter`` which uses a
    softmax-with-expert-bias scheme. Gemma4 has its own router semantics
    (no-scale RMSNorm → learnable per-hidden scale → proj → softmax → topk →
    per-expert scale multiplier). We reuse all of Megatron's infrastructure
    for dispatching (alltoall), expert parallelism, and grouped-GEMM expert
    computation — but swap in our ``Gemma4Router`` and convert its compact
    (top_k_weights [T, K], top_k_index [T, K]) output into Megatron's
    expected (probs [T, E], routing_map [T, E]) format inside ``route()``.
    """

    def __init__(self, config, submodules=None, layer_number=None, pg_collection=None):
        # Bypass MoELayer.__init__ so we can avoid building Megatron's TopKRouter,
        # then run the rest of MoELayer's setup ourselves. The parts we need:
        # - self.ep_group / num_local_experts / local_expert_indices (from BaseMoELayer)
        # - token_dispatcher, experts (alltoall path, GroupedMLP)
        # Anything shared-expert related is disabled: Gemma4 has no shared experts in
        # this sense — its "dense MLP" lives outside the MoE block in the parent layer.
        #
        # Fall back to Megatron's global parallel_state when pg_collection isn't
        # explicitly passed. TransformerLayer only forwards pg_collection when
        # submodules.mlp.module is *exactly* one of
        # (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP) — an identity check
        # via `in`, so Gemma4MoELayer (a MoELayer subclass) slips through and
        # receives None. BaseMoELayer.__init__ then crashes on `pg_collection.ep`.
        # Same fallback MoELayer.__init__ uses when invoked directly.
        # TODO(gemma4): remove this fallback and the matching
        # _Gemma4MoELayerWarningFilter when Megatron widens the MLP-type check
        # to use issubclass() (search TransformerLayer.__init__ for
        # "Unknown MLP type"); both the fallback and the filter will become
        # dead code at that point.
        if pg_collection is None:
            from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
            pg_collection = get_default_pg_collection()
        BaseMoELayer.__init__(
            self, config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        # Disable Megatron-checkpoint paths that don't apply here.
        self.moe_layer_recompute = False
        self.shared_experts_recompute = False
        self.submodules = submodules

        # --- Router: Gemma4's custom router, not TopKRouter. ---
        self.router = Gemma4Router(config)

        # --- Token dispatcher (identical to MoELayer.__init__). ---
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAllGatherTokenDispatcher,
            MoEAlltoAllTokenDispatcher,
            MoEFlexTokenDispatcher,
        )
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices,
                config=self.config, pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices,
                config=self.config, pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices,
                config=self.config, pg_collection=pg_collection,
            )
        else:
            raise ValueError(f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}")

        # --- Experts: Megatron's GroupedMLP / TEGroupedMLP. ---
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        # Gemma4 doesn't use shared experts in Megatron's sense.
        self.shared_experts = None

        # cudagraph tensor store (required by MoELayer.forward decorators)
        from megatron.core.transformer.moe.moe_utils import MoECudaGraphTensorStore
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()

        # pre_feedforward_layernorm_2: applied to experts' input ONLY (router
        # input stays un-normed). Matches HF Gemma4TextDecoderLayer:
        #   hidden_states_flat = residual            # router input (un-normed)
        #   hidden_states_2 = pre_feedforward_layernorm_2(hidden_states_flat)
        #   hidden_states_2 = experts(hidden_states_2, top_k_index, top_k_weights)
        self.pre_feedforward_layernorm_2 = TENorm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon,
        )

    def route(self, hidden_states: torch.Tensor):
        """Call ``Gemma4Router`` and pack its output into Megatron's
        ``(probs, routing_map)`` format.

        ``Gemma4Router`` emits compact top-k tensors:
            top_k_weights: [T, K] — routing weights (already scaled by per_expert_scale)
            top_k_index:   [T, K] — which experts each token routes to
        Megatron's dispatcher wants:
            probs:       [T, E] — weight per (token, expert), 0 where not routed
            routing_map: [T, E] — boolean mask
        """
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        top_k_weights, top_k_index = self.router(flat)

        num_tokens = flat.shape[0]
        num_experts = self.config.num_moe_experts
        probs = torch.zeros(
            num_tokens, num_experts,
            dtype=top_k_weights.dtype, device=top_k_weights.device,
        )
        probs.scatter_(1, top_k_index, top_k_weights)
        routing_map = probs != 0
        return probs, routing_map

    def forward(
        self, hidden_states: torch.Tensor, router_input: torch.Tensor | None = None,
    ):
        """Gemma4 MoE forward with split router / experts inputs.

        HF's ``Gemma4TextDecoderLayer`` routes based on the *un-normed* residual
        but feeds the experts the *pre-ff-norm-2'd* residual:

            hidden_states_flat = residual                       # un-normed
            _, tk_w, tk_i = self.router(hidden_states_flat)
            experts_input = self.pre_feedforward_layernorm_2(hidden_states_flat)
            output        = self.experts(experts_input, tk_i, tk_w)

        We take the un-normed residual in ``hidden_states`` and apply
        ``pre_feedforward_layernorm_2`` internally to obtain the experts
        input. The router path uses the un-normed residual directly. Callers
        may pass a different ``router_input`` for tests or ablations; when
        ``router_input is None`` (the normal case) the router sees the same
        un-normed residual the layer was called with.

        We inline the Megatron parent's ``forward`` body here — rather than
        calling ``super().forward`` with a side-channel stash — so the
        router input is passed explicitly end-to-end and the code is safe
        under activation checkpointing / recomputation.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor "
                "parallelism are enabled without also enabling sequence parallelism."
            )

        router_in = router_input if router_input is not None else hidden_states
        experts_in = self.pre_feedforward_layernorm_2(hidden_states)

        def custom_forward(experts_in, router_in):
            # Gemma4 has no shared experts; shared_experts_compute returns None.
            shared_expert_output = self.shared_experts_compute(experts_in)
            probs, routing_map = self.route(router_in)
            experts_in2, probs, residual = self.preprocess(experts_in, probs, routing_map)
            dispatched_input, probs = self.dispatch(experts_in2, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs, residual)
            output = self.combine(output, shared_expert_output)
            return output, mlp_bias

        # moe_layer_recompute is forced to False in __init__; call directly.
        return custom_forward(experts_in, router_in)


class Gemma4TransformerLayer(TransformerLayer):
    """Gemma4 transformer layer with heterogeneous attention and layer_scalar."""

    def __init__(
        self,
        config: Gemma4TransformerConfig,
        submodules: Gemma4TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        **kwargs,
    ):
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
        global_layer_number = layer_number + get_transformer_layer_offset(config)
        # Megatron passes `layer_number` as 1-indexed (default 1), so in 0-indexed
        # HF space a global layer is `(i+1) % pattern == 0` → `i % pattern == pattern-1`.
        # Equivalently: `is_sliding` when `global_layer_number % pattern != 0`.
        self.is_sliding = bool(global_layer_number % config.sliding_window_pattern)
        self._is_global = not self.is_sliding

        # Global layers have different head_dim (kv_channels) and num_kv_heads
        # (num_query_groups). Build the layer against a *cloned* config with
        # those overrides so we never mutate the shared transformer config.
        # Mutation would be reentrant-unsafe under concurrent layer
        # construction and leak global-layer shapes into sibling sliding
        # layers if an exception were raised during super().__init__.
        layer_config = (
            dc_replace(
                config,
                kv_channels=config.global_kv_channels,
                num_query_groups=config.global_num_query_groups,
            )
            if self._is_global
            else config
        )
        super().__init__(
            config=layer_config, submodules=submodules,
            layer_number=layer_number, hidden_dropout=hidden_dropout,
            **kwargs,
        )

        # Tell the attention module whether this is a global layer
        self.self_attention._is_global = self._is_global

        # Replace TE core attention with PyTorch SDPA for all layers.
        # Global layers require this because head_dim=512 exceeds flash attention's limit (256).
        # Local layers also use SDPA for consistency.
        self.self_attention.core_attention = SDPACoreAttention(
            config=config,
            layer_number=self.layer_number,
            attn_mask_type=AttnMaskType.causal,
            softmax_scale=config.softmax_scale,
        )
        self.self_attention.core_attention._is_sliding = self.is_sliding

        # Post-attention and post-feedforward layernorms (Gemma-specific)
        self.post_attention_layernorm = build_module(
            submodules.post_attention_layernorm,
            config=self.config, hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_feedforward_layernorm = build_module(
            submodules.post_feedforward_layernorm,
            config=self.config, hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # Layer scalar (buffer, not learned). Kept in fp32 intentionally —
        # HF stores this scalar in fp32 and relies on the implicit upcast of
        # ``bf16_hidden * fp32_scalar`` at multiply time (see HF Gemma4
        # ``Gemma4TextDecoderLayer.__init__`` at modeling_gemma4.py:1331).
        # Don't switch to ``dtype=self.config.params_dtype``; that would
        # silently change the arithmetic.
        self.register_buffer("layer_scalar", torch.ones(1))

        # MoE block (26B-A4B): super().__init__ already built self.mlp from the
        # layer spec, which when enable_moe_block=True is a Gemma4MoELayer (not
        # a dense MLP). We also build a parallel `dense_mlp` for Gemma4's
        # dense + MoE combined-FFN pattern. The two outputs are summed in
        # forward().
        self.enable_moe_block = getattr(config, 'enable_moe_block', False)
        if self.enable_moe_block:
            # Parallel dense MLP branch (sibling to self.mlp, which is the MoE).
            self.dense_mlp = build_module(
                submodules.dense_mlp,
                config=config,
            )
            # Gemma4 pre/post layernorms that wrap the two FFN paths.
            self.post_feedforward_layernorm_1 = TENorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon,
            )
            # pre_feedforward_layernorm_2 now lives INSIDE Gemma4MoELayer
            # (matching HF Gemma4TextDecoderLayer semantics: router sees un-normed
            # residual, experts see pre_feedforward_layernorm_2(residual)). This
            # attribute is kept on the MoE block so mbridge/state-dict paths
            # don't change.
            self.post_feedforward_layernorm_2 = TENorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon,
            )

    def _forward_dense_ffn(self, pre_mlp_ln):
        """Run the dense MLP. ``self.mlp`` is the dense MLP directly for the
        31B variant."""
        out, bias = self.mlp(pre_mlp_ln)
        return out + bias if bias is not None else out

    def _forward_moe_ffn(self, residual, pre_mlp_ln):
        """Run dense + MoE in parallel and sum (26B-A4B variant).

        Mirrors HF ``Gemma4TextDecoderLayer.forward`` (transformers
        modeling_gemma4.py:1376-1391): dense branch goes through
        ``post_feedforward_layernorm_1``, MoE branch through
        ``post_feedforward_layernorm_2``, the two are summed, and the outer
        ``Gemma4TransformerLayer.forward`` applies ``post_feedforward_layernorm``
        to the sum — 3 post-FFN LNs total for MoE layers is correct.

        HF routes on the un-normed residual but feeds experts the
        ``pre_feedforward_layernorm_2``'d residual; Gemma4MoELayer applies
        that norm internally, so we pass the un-normed residual directly.
        """
        dense_out, dense_bias = self.dense_mlp(pre_mlp_ln)
        if dense_bias is not None:
            dense_out = dense_out + dense_bias
        mlp_output = self.post_feedforward_layernorm_1(dense_out)

        moe_output, _ = self.mlp(residual)
        moe_output = self.post_feedforward_layernorm_2(moe_output)

        return mlp_output + moe_output

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_context=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        # Select per-layer rotary embeddings and attention mask
        # DualRotaryEmbedding returns concatenated [seq, 1, global_dim + local_dim] tensor.
        # Split and select based on layer type.
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            global_dim = getattr(self.config, 'dual_rope_global_dim', 0)
            if global_dim > 0 and rotary_pos_emb.shape[-1] > global_dim:
                if self.is_sliding:
                    rotary_pos_emb = rotary_pos_emb[..., global_dim:]  # local part
                else:
                    rotary_pos_emb = rotary_pos_emb[..., :global_dim]  # global part
        elif isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb[1] if self.is_sliding else rotary_pos_emb[0]
        if isinstance(attention_mask, tuple):
            attention_mask = attention_mask[1] if self.is_sliding else attention_mask[0]

        # Global layers use partial RoPE (25% of head_dim=512 = 128 dims)
        # Local layers use full RoPE (100% of head_dim=256 = 256 dims)
        # With DualRotaryEmbedding, global RoPE is full-size (512 dims) with zero-padded
        # non-rotated dims, so no truncation needed.
        # With single RoPE (local only, 256 dims), truncate for global layers.
        if not self.is_sliding and rotary_pos_emb is not None:
            global_rope_dim = int(self.config.global_kv_channels * self.config.global_partial_rotary_factor)
            if rotary_pos_emb.shape[-1] != self.config.global_kv_channels and rotary_pos_emb.shape[-1] > global_rope_dim:
                rotary_pos_emb = rotary_pos_emb[..., :global_rope_dim]

        residual = hidden_states

        extra_kwargs = {}
        if inference_context is not None:
            extra_kwargs["inference_context"] = inference_context
        elif inference_params is not None:
            extra_kwargs["inference_params"] = inference_params

        # Input layernorm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, hidden_states_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **extra_kwargs,
        )

        if hidden_states_bias is not None:
            hidden_states = hidden_states + hidden_states_bias
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # FFN path: dense-only (31B) vs dense + MoE (26B-A4B).
        residual = hidden_states
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        if self.enable_moe_block:
            hidden_states = self._forward_moe_ffn(residual, pre_mlp_layernorm_output)
        else:
            hidden_states = self._forward_dense_ffn(pre_mlp_layernorm_output)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Layer scalar
        hidden_states = hidden_states * self.layer_scalar

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True,
        )

        if self.config.external_cuda_graph and self.training:
            return output
        return output, context


class SDPACoreAttention(nn.Module):
    """Gemma4 core attention.

    Replaces TE's DotProductAttention because:
    - Global layers have head_dim=512, which flash-attn 2.x doesn't support.
    - Sliding-window layers need an explicit left-window mask (HF behavior).
    - Context-parallelism on the global layers needs an all-gather+full-attn
      path with a differentiable K/V gather.

    Dispatch at call time (packed / thd shape):
      - CP > 1 (any layer) : all-gather K/V, apply causal + optional
        sliding-window mask computed from slime zig-zag global indices.
      - global  + CP == 1  : sub-sequence causal SDPA (no O(T²) mask alloc).
      - sliding + CP == 1  : flash_attn_varlen_func with (sw-1, 0) window.
    """

    def __init__(self, config, layer_number, attn_mask_type, attention_type="self",
                 attention_dropout=None, softmax_scale=None, **kwargs):
        super().__init__()
        # Megatron's SelfAttention.__init__ passes a few kwargs (e.g. cp_comm_type,
        # model_comm_pgs) intended for TE's DotProductAttention. We accept-and-ignore
        # by name rather than asserting empty; a strict assert breaks whenever
        # Megatron/TE add a new kwarg. If a kwarg shows up here that we *should*
        # honor (e.g. a new softmax dtype), it will surface as a behavioral bug
        # in parity, which is what the test suite covers.
        del kwargs
        self.config = config
        self.softmax_scale = softmax_scale
        self.dropout_p = config.attention_dropout if attention_dropout is None else attention_dropout
        self._is_sliding = False  # set by Gemma4TransformerLayer

    def _resolve_scale(self, hn: int) -> float:
        # `0.0 or fallback` would silently mask a misconfigured scale; be explicit.
        return self.softmax_scale if self.softmax_scale is not None else (hn ** -0.5)

    @staticmethod
    def _zigzag_global_indices(local_len, cp_rank, cp_size, device):
        """Global positions of this rank's local Q tokens under slime's
        zig-zag CP layout (matches cp_utils.slice_with_cp).

        Local tokens on rank r occupy two global sub-ranges:
          [r*cs, (r+1)*cs) and [(2*cp-r-1)*cs, (2*cp-r)*cs)
        where cs = local_len / 2 = seq_len / (2*cp_size).
        """
        cs = local_len // 2
        first = torch.arange(cp_rank * cs, (cp_rank + 1) * cs, device=device)
        second = torch.arange(
            (2 * cp_size - cp_rank - 1) * cs,
            (2 * cp_size - cp_rank) * cs,
            device=device,
        )
        return torch.cat([first, second])

    def _forward_cp_subseq_mask(self, query, key, value, packed_seq_params,
                                 sliding_window=None):
        """CP>1 path for any layer: all-gather K/V, then loop over sub-seqs
        and apply a per-sub-seq attention mask built from zig-zag global
        positions. Supports causal-only (global layers) and causal +
        sliding-window (sliding layers).

        Under slime's CP convention, ``packed_seq_params.cu_seqlens_q`` holds
        GLOBAL boundaries: each packed sub-sequence on this rank represents
        ``(cu[i+1] - cu[i])`` tokens globally but only ``(cu[i+1] - cu[i]) //
        cp_size`` tokens locally (the zig-zag slice of this rank's two
        chunks, concatenated as [first, second]).
        """
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        cp_group = parallel_state.get_context_parallel_group()
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()

        t_local = query.shape[0]
        np_q, hn = query.shape[1], query.shape[2]
        nk = key.shape[1]
        scale = self._resolve_scale(hn)

        # Differentiable all-gather along the token dim. forward: AG,
        # backward: RS — so K/V grads on non-owning ranks flow back to the
        # originating rank. The raw `dist.all_gather_into_tensor` has no
        # autograd rule and PyTorch prints a "silently incorrect behavior"
        # warning + drops those grads.
        k_full = gather_from_sequence_parallel_region(key.contiguous(), group=cp_group)
        v_full = gather_from_sequence_parallel_region(value.contiguous(), group=cp_group)
        # gather_from_sequence_parallel_region stacks each rank's chunk
        # consecutively in rank order. Under zig-zag, each rank's [2*cs]
        # local tokens are [chunk_r_first, chunk_r_second]. So the gathered
        # tensor layout is [r0_first, r0_second, r1_first, r1_second, ...].
        # We need to un-zig-zag into pure global order so mask indices line
        # up. Build a permutation that maps gathered index -> global index.
        device = query.device
        dtype = query.dtype
        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None

        # Sanity: for each packed sub-seq, the GLOBAL length must be
        # divisible by 2*cp_size so chunk_size is integer. With cp_size=1 this
        # reduces to even-length, which the CP=1 parity-test harness may
        # violate (no zig-zag pre-slicing). Skip the check there; permutation
        # is identity under cp_size=1 so odd length is harmless.
        if cu_seqlens is not None and cp_size > 1:
            expected_t_local = 0
            for s_idx in range(len(cu_seqlens) - 1):
                s_len = (cu_seqlens[s_idx + 1] - cu_seqlens[s_idx]).item()
                assert s_len % (2 * cp_size) == 0, (
                    f"sub-sequence {s_idx} global length ({s_len}) is not "
                    f"divisible by 2*cp_size ({2 * cp_size}); `slice_with_cp` "
                    "should pad before packing"
                )
                expected_t_local += s_len // cp_size
            assert expected_t_local == t_local, (
                f"packed-seq local length mismatch: sum(seq_len // cp_size) = "
                f"{expected_t_local}, but query.shape[0] = {t_local}"
            )

        # Un-zig-zag: build a permutation over k_full of length `t_full` so
        # that `k_full[perm]` is in pure global order. Each sub-sequence
        # contributes its own block of size `s_len` to the permutation.
        #
        # For each sub-sequence, the gathered block layout is:
        #   [r=0: first_cs, second_cs, r=1: first_cs, second_cs, ...]
        # where first_r is global [r*cs, (r+1)*cs) and
        #       second_r is global [(2*cp-r-1)*cs, (2*cp-r)*cs).
        # We build `perm[global_pos] = gathered_pos` within each sub-seq.
        if cu_seqlens is None:
            # Single-sequence fallback: treat the full t_full as one sub-seq.
            t_full_total = k_full.shape[0]
            cu_seqlens_list = [0, t_full_total]
        else:
            cu_seqlens_list = cu_seqlens.tolist()

        # With cp_size=1 the zigzag degenerates to identity and all-gather is
        # a no-op; skip the permutation (and the floor-div that would drop the
        # trailing odd token for seq_len_global % 2 == 1).
        if cp_size > 1:
            perm_parts = []
            for s_idx in range(len(cu_seqlens_list) - 1):
                seq_start = cu_seqlens_list[s_idx]
                seq_len_global = cu_seqlens_list[s_idx + 1] - seq_start
                cs = seq_len_global // (2 * cp_size)
                # For this sub-seq, build inverse permutation: for each global
                # position g in [0, seq_len_global), find its gathered offset.
                # gathered_offset(g) = (rank of g) * 2*cs (offset to start of
                # that rank's block within the sub-seq) + local_offset_in_rank(g).
                # Rank ownership of global position g in [0, 2*cp*cs):
                #   r = g // cs (if g < cp*cs  -> "first half", r = g//cs,
                #                                  local_offset = g - r*cs)
                #   r = 2*cp-1 - g//cs (if g >= cp*cs -> "second half",
                #                       local_offset = cs + g - (2*cp-r-1)*cs)
                g = torch.arange(seq_len_global, device=device)
                q_idx = g // cs   # chunk index 0..2*cp-1
                # rank for chunk q_idx: first half (q_idx < cp) owned by q_idx,
                # second half (q_idx >= cp) owned by 2*cp - 1 - q_idx.
                owner = torch.where(q_idx < cp_size, q_idx, 2 * cp_size - 1 - q_idx)
                # local offset within that rank's 2*cs block:
                #   first half (q_idx < cp): local = g - owner*cs
                #   second half (q_idx >= cp): local = cs + (g - (2*cp-1-owner)*cs)
                local_in_rank = torch.where(
                    q_idx < cp_size,
                    g - owner * cs,
                    cs + (g - (2 * cp_size - 1 - owner) * cs),
                )
                gathered_offset = owner * (2 * cs) + local_in_rank
                perm_parts.append(gathered_offset + seq_start)  # add sub-seq base
            perm = torch.cat(perm_parts)
            k_full = k_full.index_select(0, perm)
            v_full = v_full.index_select(0, perm)

        out = torch.empty(t_local, np_q * hn, dtype=dtype, device=device)

        local_offset = 0
        for s_idx in range(len(cu_seqlens_list) - 1):
            seq_start = cu_seqlens_list[s_idx]
            seq_len_global = cu_seqlens_list[s_idx + 1] - seq_start
            local_len = seq_len_global // cp_size  # this sub-seq's local Q count

            q_seq = query[local_offset:local_offset + local_len]
            k_seq = k_full[seq_start:seq_start + seq_len_global]
            v_seq = v_full[seq_start:seq_start + seq_len_global]

            q4 = q_seq.unsqueeze(0).transpose(1, 2)   # [1, np, local_len, hn]
            k4 = k_seq.unsqueeze(0).transpose(1, 2)   # [1, nk, seq_len, hn]
            v4 = v_seq.unsqueeze(0).transpose(1, 2)

            # Global positions of local Q tokens. cp_size=1 degenerates to
            # identity; use arange to preserve odd-length seqs (zigzag helper
            # floor-divides, dropping the trailing token).
            if cp_size > 1:
                row_idx = self._zigzag_global_indices(local_len, cp_rank, cp_size, device)
            else:
                row_idx = torch.arange(local_len, device=device)
            col_idx = torch.arange(seq_len_global, device=device)
            # Causal: K pos <= Q pos. Sliding: also K pos > Q pos - sw.
            forbid_future = col_idx[None, :] > row_idx[:, None]
            if sliding_window is not None and sliding_window > 0:
                forbid_past = col_idx[None, :] < (row_idx[:, None] - (sliding_window - 1))
                forbid = forbid_future | forbid_past
            else:
                forbid = forbid_future
            mask = torch.where(
                forbid, torch.finfo(dtype).min, 0.0,
            ).to(dtype=dtype)

            o = F.scaled_dot_product_attention(
                q4, k4, v4, attn_mask=mask[None, None, :, :],
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=scale, enable_gqa=(np_q != nk),
            )
            out[local_offset:local_offset + local_len] = o.transpose(1, 2).reshape(local_len, -1)
            local_offset += local_len

        return out

    def _forward_thd_flash(self, query, key, value, cu_seqlens):
        """Sliding-window or head_dim<=256 path via flash_attn_varlen_func.

        CP==1 only. For CP>1, `_forward_cp_subseq_mask` handles zig-zag.

        Sliding-window layers must pass `window_size=(sliding_window-1, 0)` so
        only tokens within `sliding_window` positions back are attended to —
        this matches HF's `sliding_window_mask_function`. Global layers and
        dense-attention sliding layers use the default full-causal window.
        """
        from flash_attn import flash_attn_varlen_func

        window_size = (-1, -1)  # full causal when causal=True
        if self._is_sliding:
            sw = getattr(self.config, "sliding_window", None)
            if sw and sw > 0:
                window_size = (int(sw) - 1, 0)

        cu = cu_seqlens.to(torch.int32)
        max_seqlen = (cu[1:] - cu[:-1]).max().item()
        out = flash_attn_varlen_func(
            query.contiguous(), key.contiguous(), value.contiguous(),
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self._resolve_scale(query.shape[2]),
            causal=True,
            window_size=window_size,
        )
        return out.reshape(query.shape[0], -1)

    def _forward_thd_sdpa_per_subseq(self, query, key, value, cu_seqlens):
        """Per-sub-sequence causal SDPA — used when flash-attn can't handle
        head_dim (global layer w/o CP). Avoids materializing a [T, T] mask.
        """
        np_q, hn = query.shape[1], query.shape[2]
        nk = key.shape[1]
        scale = self._resolve_scale(hn)
        out = torch.empty(query.shape[0], np_q * hn, dtype=query.dtype, device=query.device)
        for i in range(len(cu_seqlens) - 1):
            s = cu_seqlens[i].item()
            e = cu_seqlens[i + 1].item()
            q4 = query[s:e].unsqueeze(0).transpose(1, 2)  # [1, np, L, hn]
            k4 = key[s:e].unsqueeze(0).transpose(1, 2)
            v4 = value[s:e].unsqueeze(0).transpose(1, 2)
            o = F.scaled_dot_product_attention(
                q4, k4, v4,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=scale, is_causal=True, enable_gqa=(np_q != nk),
            )
            out[s:e] = o.transpose(1, 2).reshape(e - s, -1)
        return out

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None,
                packed_seq_params=None, **kwargs):
        cp_size = getattr(self.config, "context_parallel_size", 1) or 1
        is_thd = query.dim() == 3

        # Parity-test hook: force CP=1 to take the same _forward_cp_subseq_mask
        # code path as CP>1 so parity comparisons don't confound kernel choice
        # (flash-attn vs SDPA) with CP-correctness. NEVER set in production —
        # flash-attn is faster than SDPA-with-mask.
        force_cp_path = getattr(self.config, "force_cp_subseq_mask", False)

        if is_thd:
            # CP>1 (any layer): all-gather KV, apply zig-zag-aware mask.
            # Sliding layers also need sliding_window masking in addition to
            # causal — `_forward_cp_subseq_mask` handles both.
            if cp_size > 1 or force_cp_path:
                sw = None
                if self._is_sliding:
                    sw_cfg = getattr(self.config, "sliding_window", None)
                    if sw_cfg and sw_cfg > 0:
                        sw = int(sw_cfg)
                return self._forward_cp_subseq_mask(
                    query, key, value, packed_seq_params, sliding_window=sw,
                )

            # CP==1: use the existing per-layer-type paths.
            cu_seqlens = None
            if packed_seq_params is not None:
                cu_seqlens = packed_seq_params.cu_seqlens_q

            hn = query.shape[2]
            if cu_seqlens is not None:
                if hn <= 256:
                    return self._forward_thd_flash(query, key, value, cu_seqlens)
                # Global layer, no CP, packed — flash-attn won't take head_dim>256.
                return self._forward_thd_sdpa_per_subseq(query, key, value, cu_seqlens)

            # Un-packed thd (rare: eval/smoke test with batch=1). Plain SDPA.
            q = query.unsqueeze(0).transpose(1, 2)
            k = key.unsqueeze(0).transpose(1, 2)
            v = value.unsqueeze(0).transpose(1, 2)
            nq, nk = q.shape[1], k.shape[1]
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=self._resolve_scale(hn), is_causal=True,
                enable_gqa=(nq != nk),
            )
            return out.transpose(1, 2).reshape(query.shape[0], -1)

        # bshd path: 4D input [seq, batch, np, hn]. Used by smoke tests and
        # potential eval with --qkv-format bshd.
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        nq, nk = q.shape[1], k.shape[1]
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self._resolve_scale(query.shape[3]), is_causal=True,
            enable_gqa=(nq != nk),
        )
        return out.permute(2, 0, 1, 3).reshape(out.size(2), out.size(0), -1)


class Gemma4SelfAttention(SelfAttention):
    """SelfAttention with Gemma4-specific modifications:
    - v_norm: RMSNorm without learnable scale applied to value states.
    - attention_k_eq_v: on global layers the linear_qkv projection emits
      ``[q, k]`` only (no v_proj) and V is derived from K — specifically
      ``V = v_norm(raw_k)`` while ``K = k_norm(raw_k)``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_global = False  # set by Gemma4TransformerLayer after construction
        self.v_norm = VNorm(self.hidden_size_per_attention_head, eps=self.config.layernorm_epsilon)

    def _split_qkv_global_k_eq_v(self, hidden_states):
        """Split linear_qkv output for global K=V layers.

        The Mcore linear_qkv weight for a K=V global layer is built with
        ``v_proj_weight == k_proj_weight`` (see Gemma4Bridge + convert_gemma4_to_hf),
        so ``linear_qkv(h)`` emits Q/K/V with ``raw_k == raw_v``. Gemma4's
        per-head norms then apply as ``key = k_norm(raw_k)`` and
        ``value = v_norm(raw_k)`` — *not* ``v_norm(k_norm(raw_k))``. We
        reimplement the split here rather than calling the parent so we
        don't have to mutate ``self.k_layernorm`` mid-forward.

        Returns (query[sq,b,np,hn], key[sq,b,ng,hn], value[sq,b,ng,hn]).
        """
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        new_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (num_query_heads_per_group + 2) * self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_shape)

        q_width = num_query_heads_per_group * self.hidden_size_per_attention_head
        hn = self.hidden_size_per_attention_head
        # _raw_value is bit-identical to raw_key (K=V linear_qkv layout);
        # use raw_key directly and skip the extra tensor.
        query, raw_key, _raw_value = torch.split(mixed_qkv, [q_width, hn, hn], dim=3)
        query = query.reshape(query.size(0), query.size(1), -1, hn)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        value = self.v_norm(raw_key)
        key = self.k_layernorm(raw_key) if self.k_layernorm is not None else raw_key
        return query, key, value

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True
    ):
        if self._is_global and self.config.attention_k_eq_v and split_qkv:
            if output_gate:
                raise NotImplementedError(
                    "output_gate is not supported together with attention_k_eq_v"
                )
            return self._split_qkv_global_k_eq_v(hidden_states)

        result = super().get_query_key_value_tensors(
            hidden_states, key_value_states, output_gate=output_gate, split_qkv=split_qkv
        )
        if not split_qkv:
            return result

        if output_gate:
            query, key, value, gate = result
            value = self.v_norm(value)
            return query, key, value, gate

        query, key, value = result
        value = self.v_norm(value)
        return query, key, value


def _build_moe_submodule_spec(config):
    """Build the MoE submodule spec (Gemma4MoELayer + TE GroupedMLP experts)."""
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    # Reuse Megatron's canonical TE-backed MoE spec factory to get
    # TEColumnParallelGroupedLinear / TERowParallelGroupedLinear etc. wired up
    # properly for GroupedMLP experts. Then swap the top-level module from
    # Megatron's MoELayer to our Gemma4MoELayer, which keeps all that wiring
    # but plugs in Gemma4Router.
    base_spec = get_moe_module_spec_for_backend(
        backend=TESpecProvider(),
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=False,  # use plain F.gelu(approximate='tanh') from config.activation_func
    )
    return ModuleSpec(
        module=Gemma4MoELayer,
        submodules=base_spec.submodules,
        metainfo=base_spec.metainfo,
    )


def get_gemma4_layer_spec_te(config=None) -> ModuleSpec:
    """Layer spec for Gemma4 using native Megatron attention with TE.

    If ``config.enable_moe_block`` is set, the main ``mlp`` submodule is a
    :class:`Gemma4MoELayer` (so that the state-dict path
    ``.mlp.experts.linear_fc*.weight*`` matches mbridge's EP auto-handling),
    and the original dense MLP moves to a sibling ``dense_mlp`` submodule that
    the layer forward sums with the MoE output. For the 31B dense variant,
    ``enable_moe_block=False`` and ``mlp`` stays as the normal Megatron MLP.
    """
    # dense_mlp: use a plain (non-fused-layernorm) linear_fc1 so our explicit
    # `pre_mlp_layernorm` in the layer forward is the sole norm applied to the
    # MLP input. Using TELayerNormColumnParallelLinear here would apply a
    # SECOND layernorm inside fc1, resulting in double-normalization and
    # ~8× inflated MLP outputs.
    dense_mlp_spec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
    if config is not None and getattr(config, "enable_moe_block", False):
        mlp_spec = _build_moe_submodule_spec(config)
        dense_spec = dense_mlp_spec
    else:
        mlp_spec = dense_mlp_spec
        dense_spec = IdentityOp

    submods = Gemma4TransformerLayerSubmodules(
        self_attention=ModuleSpec(
            module=Gemma4SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=TENorm,
                k_layernorm=TENorm,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=IdentityOp,
        mlp=mlp_spec,
        mlp_bda=get_bias_dropout_add,
        post_attention_layernorm=TENorm,
        post_feedforward_layernorm=TENorm,
        dense_mlp=dense_spec,
    )
    return ModuleSpec(module=Gemma4TransformerLayer, submodules=submods)


@functools.lru_cache(maxsize=4)
def _load_hf_text_config(hf_checkpoint):
    """Load HF config and unwrap `text_config` if it's a multimodal wrapper.

    Cached via lru_cache so repeated callers (model provider, mbridge, weight
    converter) all share the same parsed object.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


class _Gemma4MoELayerWarningFilter(logging.Filter):
    """Silence the once-per-layer Megatron warning:
        'Unknown MLP type: <class Gemma4MoELayer>. Using default kwargs.'
    Megatron's TransformerLayer.__init__ recognizes a hardcoded tuple of MLP
    classes via `==` (not issubclass), so Gemma4MoELayer (a MoELayer subclass)
    falls through to the default-kwargs branch. That branch is correct for us
    — Gemma4MoELayer.__init__ fetches its own pg_collection via
    get_default_pg_collection — but the warning spams 30 lines per layer at
    init and confuses log readers. See gemma4_provider.py install hook.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("Unknown MLP type" in msg and "Gemma4MoELayer" in msg)


def _install_moe_warning_filter():
    """Silence the per-layer "Unknown MLP type: Gemma4MoELayer" warning.

    Megatron's TransformerLayer compares MLP class identity via ``==``, so
    MoELayer subclasses hit the default-kwargs branch and log a warning.
    The default-kwargs branch is correct for us (Gemma4MoELayer fetches
    pg_collection itself); filter the noise.
    """
    tl_logger = logging.getLogger("megatron.core.transformer.transformer_layer")
    if getattr(tl_logger, "_gemma4_moe_filter_installed", False):
        return
    tl_logger.addFilter(_Gemma4MoELayerWarningFilter())
    tl_logger._gemma4_moe_filter_installed = True


def _assert_hf_features_supported(hf_text):
    """Fail loudly on Gemma4 HF features this plugin doesn't implement."""
    if getattr(hf_text, "hidden_size_per_layer_input", 0):
        raise NotImplementedError(
            "Gemma4 per-layer input mechanism "
            f"(hidden_size_per_layer_input={hf_text.hidden_size_per_layer_input}) "
            "is not implemented. See Gemma4TextDecoderLayer.per_layer_input_gate in HF."
        )
    if getattr(hf_text, "num_kv_shared_layers", 0):
        raise NotImplementedError(
            "Gemma4 KV-sharing across the last N layers "
            f"(num_kv_shared_layers={hf_text.num_kv_shared_layers}) is not implemented."
        )
    if getattr(hf_text, "use_double_wide_mlp", False):
        raise NotImplementedError("Gemma4 use_double_wide_mlp is not implemented.")
    # Text-only training assumes causal attention; HF's "all" mode disables it.
    if getattr(hf_text, "use_bidirectional_attention", "vision") == "all":
        raise NotImplementedError(
            "Gemma4 use_bidirectional_attention='all' disables causal masking; not supported."
        )


def _apply_core_config(config, hf_text):
    """Set Gemma4's non-MoE, non-RoPE config fields.

    Mutates ``config`` in place. Promotes its ``__class__`` to
    ``Gemma4TransformerConfig`` so the new dataclass fields are reachable
    from downstream Megatron code.
    """
    # Gemma uses GeGLU (gated gelu-tanh), not SwiGLU.
    config.gated_linear_unit = True
    config.activation_func = _gelu_tanh
    config.bias_activation_fusion = False

    # No MoE-vs-dense layer scheduling: every layer is our Gemma4TransformerLayer
    # and the MoE block lives inside its forward. An all-zero list keeps
    # transformer_block's non_homogeneous_layers=True branch active (correct for
    # 26B's differing global vs sliding head_dim / num_kv_heads).
    # Rationale for using moe_layer_freq as the flag: Megatron's
    # TransformerBlock.__init__ sets ``non_homogeneous_layers = True`` iff
    # ``config.moe_layer_freq is not None``. We only need that flag on —
    # the actual dense/MoE dispatch happens inside
    # Gemma4TransformerLayer.forward, so the list contents are never
    # consulted by TransformerBlock itself. If a future Megatron refactor
    # starts reading the list per-layer, we need a Gemma4-specific schedule
    # instead.
    config.moe_layer_freq = [0] * config.num_layers

    # Mirror Megatron's own misspelling (`hetereogenous_*`) — correcting it
    # would silently no-op on Megatron's read path.
    # TODO(gemma4): rename to ``heterogeneous_dist_checkpoint`` when Megatron
    # fixes the spelling upstream.
    config.hetereogenous_dist_checkpoint = True

    config.__class__ = Gemma4TransformerConfig
    config.global_kv_channels = hf_text.global_head_dim
    config.global_num_query_groups = hf_text.num_global_key_value_heads
    config.attention_k_eq_v = getattr(hf_text, "attention_k_eq_v", True)
    config.final_logit_softcapping = getattr(hf_text, "final_logit_softcapping", 30.0)
    config.sliding_window = hf_text.sliding_window

    # `sliding_window_pattern` isn't in Gemma4 HF configs — infer from
    # layer_types (first full_attention layer's 1-indexed position).
    layer_types = list(getattr(hf_text, "layer_types", []))
    try:
        config.sliding_window_pattern = layer_types.index("full_attention") + 1
    except ValueError:
        config.sliding_window_pattern = 6

    # Q/K norms handle softmax scaling; Megatron's default of 1/sqrt(hn) is wrong.
    config.softmax_scale = 1.0
    # Fused RoPE ignores zeroed inv_freq tails; we need unfused for partial-rotary.
    config.apply_rope_fusion = False


def _apply_moe_config(config, hf_text):
    """Set MoE fields if this is a MoE variant (26B-A4B)."""
    config.enable_moe_block = getattr(hf_text, "enable_moe_block", False)
    if not config.enable_moe_block:
        return

    config.num_moe_experts = hf_text.num_experts
    config.moe_router_topk = hf_text.top_k_experts
    config.moe_ffn_hidden_size = hf_text.moe_intermediate_size
    # Megatron MoE infrastructure reads these even though our custom router
    # bypasses its scoring logic; defaults mirror a working Qwen3.5-A3B config.
    config.moe_token_dispatcher_type = (
        getattr(config, "moe_token_dispatcher_type", None) or "alltoall"
    )
    config.moe_grouped_gemm = getattr(config, "moe_grouped_gemm", None) or True
    config.moe_aux_loss_coeff = 0.0  # Gemma4 router has no aux loss
    config.moe_router_load_balancing_type = (
        getattr(config, "moe_router_load_balancing_type", None) or "none"
    )
    config.moe_router_score_function = (
        getattr(config, "moe_router_score_function", None) or "softmax"
    )
    config.moe_router_topk_scaling_factor = (
        getattr(config, "moe_router_topk_scaling_factor", None) or 1.0
    )
    config.moe_router_pre_softmax = False


def get_rope_local_base_freq(hf_text) -> float:
    """Extract sliding-attention RoPE theta from an HF Gemma4 text config.

    Single source of truth for both the model provider and the mbridge
    config builder — otherwise the 10000.0 default would drift between
    call sites.
    """
    return (
        getattr(hf_text, "rope_parameters", {}) or {}
    ).get("sliding_attention", {}).get("rope_theta", 10000.0)


def _apply_rope_config(config, hf_text):
    rope_params = getattr(hf_text, "rope_parameters", {}) or {}
    config.rope_local_base_freq = get_rope_local_base_freq(hf_text)
    config.global_partial_rotary_factor = (
        rope_params.get("full_attention", {}).get("partial_rotary_factor", 0.25)
    )


def _guard_cp_sliding_window(args, config):
    """Fail if per-rank CP token cap is smaller than the sliding window.

    Strong signal of a miscounted CP sizing — we'd train on truncated
    attention windows otherwise.
    """
    cp_size = getattr(args, "context_parallel_size", 1) or 1
    if cp_size <= 1:
        return
    max_tokens = getattr(args, "max_tokens_per_gpu", None)
    if max_tokens is not None and max_tokens < config.sliding_window:
        raise ValueError(
            f"context_parallel_size={cp_size} with max_tokens_per_gpu={max_tokens} "
            f"< sliding_window={config.sliding_window}: per-rank CP chunk cap is "
            "smaller than the sliding window. Reduce CP or raise max_tokens_per_gpu."
        )


def get_gemma4_spec(args, config, vp_stage):
    """Return the native Gemma4 layer spec with proper config overrides."""
    hf_text = _load_hf_text_config(args.hf_checkpoint)

    _install_moe_warning_filter()
    _assert_hf_features_supported(hf_text)
    _apply_core_config(config, hf_text)
    _apply_moe_config(config, hf_text)
    _apply_rope_config(config, hf_text)
    _guard_cp_sliding_window(args, config)

    # Build layer spec AFTER MoE config (so the MoE submodule spec attaches
    # when enable_moe_block is True), then override MLP specs for HF numerics.
    # The linear_fc1 override applies only to the dense MLP — on the MoE path
    # the top-level `mlp` is Gemma4MoELayer and its `submodules` is an
    # MoESubmodules (experts / shared_experts), which has no `linear_fc1`.
    # The dense_mlp sibling already uses TEColumnParallelLinear (see
    # `dense_mlp_spec` above), so on MoE layers this block is a no-op by
    # design.
    spec = get_gemma4_layer_spec_te(config)
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    if not getattr(config, "enable_moe_block", False):
        spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
    spec.submodules.mlp.metainfo = {"fuse_pre_mlp_layernorm": False}
    spec.submodules.pre_mlp_layernorm = TESpecProvider().layer_norm()
    return spec
