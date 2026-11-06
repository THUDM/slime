# Qwen3-30B-A3B SGLang-aligned self-attention plugin.
#
# Purpose: bit-align Megatron train-side attention with SGLang rollout-side
# attention for Qwen3-30B-A3B (standard GQA, 32 Q / 4 KV heads, head_dim=128,
# q_norm+k_norm on head_dim, RoPE base=1e6).
#
# Context parallel (CP>1): Megatron THD zigzag splits each sequence across the
# CP group. This plugin all-gathers Q/K/V, unshuffles to original packed order,
# runs the same per-seq FA4 as CP=1, then slices the local rows. RoPE uses
# global in-sequence offsets, not arange(local_T).
#
# Strategy mirrors slime_plugins.models.glm5.glm5:
#   * Register a spec that REPLACES layer.self_attention with a custom module
#     while leaving mlp untouched (the DeepGEMM MoE forward hook already
#     aligns the MLP path with SGLang).
#   * Custom module uses plain Megatron ColumnParallelLinear / RowParallelLinear
#     (NOT TELayerNormColumnParallelLinear) so `linear_qkv` is a pure matmul
#     and cannot double-normalize its input.
#   * Custom module reads the FP32 residual sum written by transformer_layer
#     ("_use_sglang_fused_residual_rmsnorm" path), does the RMSNorm itself in
#     FP32, then runs qkv_proj -> split -> q_norm/k_norm -> SGLang RoPE ->
#     flash_attn.cute.flash_attn_varlen_func (FA4) -> o_proj. Every step
#     mirrors SGLang qwen3_moe.py Qwen3MoeAttention.forward_prepare_native +
#     forward_core.
#   * Fused QK-Norm+RoPE and FA4 have no kernel backward. qwen3_attn_ops wraps
#     them like GLM-5 SparseMLA: SGLang kernel forward, analytic/SDPA backward.

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import get_num_layers_to_build

_SGLANG_ROPE_CACHE: dict = {}


# ---------------------------------------------------------------------------
# SGLang-aligned RoPE (bit-identical to sglang.srt.layers.rotary_embedding when
# used with base=rope_theta, is_neox=True; matches GLM-5 plugin's approach).
# ---------------------------------------------------------------------------

@torch.no_grad()
def _apply_sglang_rope_forward(
    value: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace

    output = torch.empty_strided(
        value.size(), value.stride(), dtype=value.dtype, device=value.device
    )
    output.copy_(value)
    dummy_k = torch.empty(
        (value.shape[0], 1, value.shape[-1]),
        dtype=value.dtype,
        device=value.device,
    )
    apply_rope_with_cos_sin_cache_inplace(
        output,
        dummy_k,
        cos_sin_cache,
        positions,
        is_neox=True,
    )
    return output


class _SGLangRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, cos_sin_cache, positions):
        ctx.save_for_backward(cos_sin_cache, positions)
        ctx.is_neox = True
        return _apply_sglang_rope_forward(value, cos_sin_cache, positions)

    @staticmethod
    def backward(ctx, grad_output):
        cos_sin_cache, positions = ctx.saved_tensors
        half = grad_output.shape[-1] // 2
        broadcast_shape = (positions.numel(),) + (1,) * (grad_output.ndim - 2) + (half,)
        cos = cos_sin_cache[positions, :half].view(broadcast_shape).to(grad_output.dtype)
        sin = cos_sin_cache[positions, half:].view(broadcast_shape).to(grad_output.dtype)
        # is_neox=True: pairs are (x[:half], x[half:]) rotated as
        #   out[:half] = x[:half] * cos - x[half:] * sin
        #   out[half:] = x[half:] * cos + x[:half] * sin
        # Gradient is the transpose:
        grad_first = grad_output[..., :half]
        grad_second = grad_output[..., half:]
        grad_input = torch.cat(
            (
                grad_first * cos + grad_second * sin,
                grad_second * cos - grad_first * sin,
            ),
            dim=-1,
        )
        return grad_input, None, None


def _get_sglang_rope_cache(
    device: torch.device,
    rotary_dim: int,
    rotary_base: float,
    needed_positions: int,
) -> torch.Tensor:
    key = (device.type, device.index, rotary_dim, float(rotary_base))
    cache = _SGLANG_ROPE_CACHE.get(key)
    if cache is not None and cache.shape[0] >= needed_positions:
        return cache
    inv_freq = 1.0 / (
        rotary_base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    positions = torch.arange(needed_positions, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    _SGLANG_ROPE_CACHE[key] = cache
    return cache


def _apply_rope(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_dim: int,
    rotary_base: float,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SGLang-aligned RoPE to x with shape [T, H, D]. Positions are
    reconstructed from cu_seqlens so pack layout matches SGLang."""

    if positions is None:
        token_ids = torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
        seq_ids = torch.searchsorted(cu_seqlens[1:], token_ids, right=True)
        positions = token_ids - cu_seqlens[seq_ids]
    cache = _get_sglang_rope_cache(
        x.device,
        rotary_dim,
        rotary_base,
        int(positions.max().item()) + 1,
    )
    if torch.is_grad_enabled() and x.requires_grad:
        return _SGLangRoPE.apply(x, cache, positions)
    return _apply_sglang_rope_forward(x, cache, positions)



# ---------------------------------------------------------------------------
# SGLang-aligned per-token RMSNorm for input_layernorm.
#
# Rationale: on layer 0, transformer_layer.py's FP32-residual-sum branch is
# skipped (no residual sum yet), and it falls through to
# `self.input_layernorm(hidden_states)`. Baseline (TE-fused linear_qkv) routes
# through the DeepGEMM `_norm_forward` hook, which short-circuits to
# `sglang.srt.batch_invariant_ops.rms_norm_batch_invariant` when
# MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS=1. Our plain-linear_qkv plugin does
# not go through that hook; using TENorm here therefore diverges from SGLang.
# This wrapper is a drop-in replacement that calls rms_norm_batch_invariant
# directly, matching SGLang's Qwen3MoeDecoderLayer.forward on layer 0.
# ---------------------------------------------------------------------------


class _SGLangBatchInvariantRMSNorm(torch.nn.Module):
    """Weight-only RMSNorm whose forward is bit-identical to SGLang.

    Signature matches Megatron's ``TENorm(config, hidden_size, eps=1e-5)`` so
    it can be dropped into ``layer_specs.submodules.input_layernorm``.
    """

    def __init__(self, config, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        import os as _os
        if _os.environ.get("QWEN3_ALIGNED_PROBE", "0") == "1":
            print(f"[QWEN3-ALIGNED v12-PROBE] _SGLangBatchInvariantRMSNorm __init__ hidden={hidden_size} eps={eps}", flush=True)
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        # Mark weight as sequence-parallel-safe for Megatron gradient allreduce.
        setattr(self.weight, "sequence_parallel", bool(getattr(config, "sequence_parallel", False)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        try:
            from sglang.srt.batch_invariant_ops import rms_norm_batch_invariant
        except ImportError:
            rms_norm_batch_invariant = None
        if rms_norm_batch_invariant is not None:
            out_shape = hidden_states.shape
            x_flat = hidden_states.reshape(-1, self.hidden_size)
            out = rms_norm_batch_invariant(x_flat, self.weight, self.eps)
            return out.view(out_shape)
        # FP32-reduction fallback (matches _sglang_native_rmsnorm_from_fp32_sum).
        x_fp32 = hidden_states.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Submodule dataclass + custom self-attention module for Qwen3-30B-A3B.
# ---------------------------------------------------------------------------

@dataclass
class Qwen3MoeAlignedSelfAttentionSubmodules:
    """Submodules for the SGLang-aligned Qwen3 MoE self-attention layer."""

    linear_qkv: ModuleSpec | type = None
    linear_proj: ModuleSpec | type = None
    q_layernorm: ModuleSpec | type = None
    k_layernorm: ModuleSpec | type = None


class Qwen3MoeAlignedSelfAttention(MegatronModule):
    """SGLang-aligned Qwen3 MoE self-attention.

    Mirrors :class:`Qwen3MoeAttention` from
    ``sglang.srt.models.qwen3_moe`` at the arithmetic level:

    1. ``qkv_proj(hidden_states)``  (plain ColumnParallelLinear)
    2. split into Q / K / V of sizes ``[q_size, kv_size, kv_size]``
    3. reshape to ``[T, H, head_dim]`` and apply per-head RMSNorm
       (``q_norm`` / ``k_norm`` with weight of shape ``[head_dim]``)
    4. apply SGLang RoPE (``is_neox=True``, base = ``rotary_base``)
    5. ``flash_attn.cute.flash_attn_varlen_func`` (FA4) — the same kernel
       SGLang uses with ``--attention-backend fa4``
    6. ``o_proj`` (plain RowParallelLinear)

    The Megatron ``transformer_layer`` still owns the residual-add and the
    input-layernorm hop. This module accepts ``hidden_states`` that has
    already been normalized. Alignment with SGLang RMSNorm is achieved by
    setting the layer's ``input_layernorm`` to the local ``FusedLayerNorm``
    (which has a plain ``.weight``); with
    ``MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS=1`` the layer replaces its
    output with the FP32-residual-sum RMSNorm value, and because our
    ``linear_qkv`` is a *plain* ColumnParallelLinear (not a TE fused
    LayerNormColumnParallelLinear) there is no double normalization.
    """

    def __init__(
        self,
        config,
        submodules: Qwen3MoeAlignedSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,
        model_comm_pgs=None,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type

        tp_size = mpu.get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % tp_size == 0, (
            f"num_attention_heads={config.num_attention_heads} not divisible by TP={tp_size}"
        )

        total_num_heads = config.num_attention_heads
        total_num_kv_heads = getattr(config, "num_query_groups", None) or config.num_attention_heads
        assert total_num_heads % total_num_kv_heads == 0

        self.head_dim = getattr(config, "kv_channels", None) or (
            config.hidden_size // total_num_heads
        )
        # v34c: get rotary_base from TransformerConfig, but also check
        # the HF config via args if TransformerConfig has the default.
        _tc_rb = float(getattr(config, "rotary_base", 10000.0))
        if _tc_rb == 10000.0:
            # TransformerConfig may not have the correct value;
            # try to get it from the HF model config.
            import os as _os34c
            _hf_path = _os34c.environ.get("HF_CHECKPOINT", "")
            if not _hf_path:
                # fallback: try reading from megatron args
                try:
                    from megatron.training import get_args
                    _args34 = get_args()
                    _hf_path = getattr(_args34, "hf_checkpoint", "")
                    _tc_rb = float(getattr(_args34, "rotary_base", 10000.0))
                except Exception:
                    pass
        self.rotary_base = _tc_rb
        self.rms_norm_eps = float(config.layernorm_epsilon)

        if total_num_kv_heads >= tp_size:
            assert total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % total_num_kv_heads == 0

        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # QKV projection: plain ColumnParallelLinear over [q_size + 2*kv_size].
        # Bias controlled by attention_bias (Qwen3 = False).
        attn_bias = not getattr(config, "add_bias_linear", False) is False
        # Actually: --disable-bias-linear -> add_bias_linear=False. Qwen3 has no attn bias.
        # Force off:
        attn_bias = False

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            config.hidden_size,
            total_num_heads * self.head_dim + 2 * total_num_kv_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=attn_bias,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )

        # Output projection: plain RowParallelLinear.
        self.linear_proj = build_module(
            submodules.linear_proj,
            total_num_heads * self.head_dim,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=attn_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

        # Per-head RMSNorm on head_dim. Match SGLang RMSNorm(head_dim, eps).
        # We use torch.nn.RMSNorm (available in PyTorch >= 2.4) for a portable,
        # weight-only RMSNorm that matches SGLang bit-for-bit when both sides
        # run FP32-cast internal reduction (rms_norm_batch_invariant path).
        self.q_norm = torch.nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.k_norm = torch.nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        import os as _os
        if _os.environ.get("QWEN3_ALIGNED_PROBE", "0") == "1":
            print(f"[QWEN3-ALIGNED v12-PROBE] Qwen3MoeAlignedSelfAttention __init__ layer={layer_number} q_size={self.q_size} kv_size={self.kv_size}", flush=True)

    @staticmethod
    def _cp_meta() -> tuple[int, int, object | None]:
        try:
            cp_size = int(mpu.get_context_parallel_world_size())
            cp_rank = int(mpu.get_context_parallel_rank())
            cp_group = mpu.get_context_parallel_group() if cp_size > 1 else None
        except Exception:
            return 1, 0, None
        return cp_size, cp_rank, cp_group

    def _gather_unshuffle_thd(self, local: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """All-gather CP-local THD tokens and restore original packed order."""
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
        from slime_plugins.models.qwen3_attn_ops import unshuffle_cp_rank_concat

        cp_size, _, cp_group = self._cp_meta()
        if cp_size <= 1 or cp_group is None:
            return local
        packed = gather_from_sequence_parallel_region(local.contiguous(), group=cp_group)
        return unshuffle_cp_rank_concat(packed, cu_seqlens, cp_size=cp_size)

    def _local_rope_positions(self, packed_seq_params: PackedSeqParams, local_tokens: int) -> torch.Tensor:
        from slime_plugins.models.qwen3_attn_ops import thd_cp_local_positions

        cp_size, cp_rank, _ = self._cp_meta()
        cu_q = packed_seq_params.cu_seqlens_q
        device = cu_q.device
        if cp_size <= 1:
            token_ids = torch.arange(local_tokens, dtype=torch.int64, device=device)
            seq_ids = torch.searchsorted(cu_q[1:], token_ids, right=True)
            return (token_ids - cu_q[seq_ids]).to(torch.int32)
        positions = thd_cp_local_positions(cu_q, cp_size=cp_size, cp_rank=cp_rank, device=device)
        if positions.numel() != local_tokens:
            raise RuntimeError(
                "THD CP RoPE position count mismatch: "
                f"positions={positions.numel()} local_tokens={local_tokens} cp_size={cp_size}"
            )
        return positions

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head RMSNorm over the head_dim axis.

        Follows sglang.srt.models.utils.apply_qk_norm: reshape flat
        [..., num_heads * head_dim] into [..., num_heads, head_dim],
        RMSNorm on the last axis with a shared weight, then flatten.
        """

        if os.getenv("MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS", "0") == "1":
            try:
                from sglang.srt.batch_invariant_ops import rms_norm_batch_invariant
            except ImportError:
                rms_norm_batch_invariant = None
            # Triton RMS has no autograd. Keep it for the bit-exact no-grad
            # path; training backward uses native RMSNorm.
            if rms_norm_batch_invariant is not None and not (
                torch.is_grad_enabled() and (q.requires_grad or k.requires_grad)
            ):
                q_flat = q.reshape(-1, self.head_dim)
                k_flat = k.reshape(-1, self.head_dim)
                q_flat = rms_norm_batch_invariant(
                    q_flat, self.q_norm.weight, self.rms_norm_eps
                )
                k_flat = rms_norm_batch_invariant(
                    k_flat, self.k_norm.weight, self.rms_norm_eps
                )
                return q_flat.view_as(q), k_flat.view_as(k)

        # Fallback: torch.nn.functional.rms_norm with FP32 reduction.
        q_out = F.rms_norm(
            q.float(),
            normalized_shape=(self.head_dim,),
            weight=self.q_norm.weight.float(),
            eps=self.rms_norm_eps,
        ).to(q.dtype)
        k_out = F.rms_norm(
            k.float(),
            normalized_shape=(self.head_dim,),
            weight=self.k_norm.weight.float(),
            eps=self.rms_norm_eps,
        ).to(k.dtype)
        return q_out, k_out


    def _apply_fused_qk_norm_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """v18: SGLang fused QK-Norm + RoPE (bit-identical to
        sglang.srt.models.qwen3_moe.Qwen3MoeAttention.apply_qk_norm_rope
        when use_fused_qk_norm_rope=True).

        query: [T, num_heads, head_dim] bf16
        key:   [T, num_kv_heads, head_dim] bf16
        value: [T, num_kv_heads, head_dim] bf16 (unused here, only for shape)
        Returns (query, key) after QK-norm + RoPE. Value untouched.
        """
        from slime_plugins.models.qwen3_attn_ops import fused_qk_norm_rope_with_grad

        T = query.shape[0]
        positions = self._local_rope_positions(packed_seq_params, T)
        q_w = self.q_norm.weight
        k_w = self.k_norm.weight
        query, key, value_v = fused_qk_norm_rope_with_grad(
            query,
            key,
            value,
            q_w,
            k_w,
            positions,
            eps=self.rms_norm_eps,
            rotary_base=self.rotary_base,
        )
        return query, key, value_v

    def _fa4_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> torch.Tensor:
        """FA4 varlen forward. query/key/value are [T, H, D] (already squeezed)."""

        from slime_plugins.models.qwen3_attn_ops import fa4_varlen_with_grad

        cu_q = packed_seq_params.cu_seqlens_q.to(torch.int32)
        cu_kv = packed_seq_params.cu_seqlens_kv.to(torch.int32)
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        return fa4_varlen_with_grad(
            query,
            key,
            value,
            cu_q,
            cu_kv,
            softmax_scale=self.scaling,
        )

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb=None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        import os as _os
        import torch as _th
        assert packed_seq_params is not None, (
            "Qwen3MoeAlignedSelfAttention requires packed_seq_params (THD layout)"
        )
        assert inference_context is None and inference_params is None, (
            "Qwen3MoeAlignedSelfAttention is training-only (no KV-cache path)"
        )

        # hidden_states: [s, b, h]. We keep this layout throughout the module.
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        # mixed_qkv: [T, 1, num_groups * (heads_per_group + 2) * head_dim].
        # Megatron GQA layout (see hf_to_megatron/common.merge_qkv) interleaves
        # per KV group: [Q_g0, K_g0, V_g0, Q_g1, K_g1, V_g1, ...]. We MUST
        # respect that layout when splitting, otherwise Q/K/V get scrambled.
        num_groups = self.num_kv_heads
        heads_per_group = self.num_heads // num_groups
        per_group = (heads_per_group + 2) * self.head_dim
        T = mixed_qkv.shape[0]
        mixed = mixed_qkv.view(T, 1, num_groups, per_group)
        q_chunk = heads_per_group * self.head_dim
        query = mixed[..., :q_chunk]
        key   = mixed[..., q_chunk:q_chunk + self.head_dim]
        value = mixed[..., q_chunk + self.head_dim:]
        query = query.reshape(T, self.num_heads, self.head_dim)
        key   = key.reshape(T, num_groups, self.head_dim)
        value = value.reshape(T, num_groups, self.head_dim)

        # v18: optional SGLang fused QK-Norm + RoPE path.
        _use_fused_qk_rope = _os.environ.get("QWEN3_ALIGNED_USE_FUSED_QK_ROPE", "1") == "1" and query.dtype == _th.bfloat16
        if _use_fused_qk_rope:
            query, key, value = self._apply_fused_qk_norm_rope(query, key, value, packed_seq_params)
        else:
            # Per-head RMSNorm on head_dim (matches SGLang apply_qk_norm).
            query, key = self._apply_qk_norm(query, key)

            # RoPE (SGLang-aligned). CP>1 uses global offsets inside each seq.
            positions = self._local_rope_positions(packed_seq_params, query.shape[0])
            query = _apply_rope(
                query, packed_seq_params.cu_seqlens_q, self.head_dim, self.rotary_base, positions=positions
            )
            key = _apply_rope(
                key, packed_seq_params.cu_seqlens_kv, self.head_dim, self.rotary_base, positions=positions
            )

        cp_size, cp_rank, _ = self._cp_meta()
        if cp_size > 1:
            cu_q = packed_seq_params.cu_seqlens_q
            query = self._gather_unshuffle_thd(query, cu_q)
            key = self._gather_unshuffle_thd(key, cu_q)
            value = self._gather_unshuffle_thd(value, cu_q)

        # FA4 varlen attention. Output is [T, num_heads * head_dim].
        core_attn_out = self._fa4_attention(query, key, value, packed_seq_params)
        if cp_size > 1:
            from slime_plugins.models.qwen3_5_vl_utils import get_packed_cp_local_indices

            local_idx = get_packed_cp_local_indices(
                packed_seq_params.cu_seqlens_q, cp_size, cp_rank, core_attn_out.device
            )
            core_attn_out = core_attn_out.index_select(0, local_idx)
        # core_attn_out: [T, num_heads, head_dim] -> flatten heads
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], -1)
        # Restore [T, 1, num_heads*head_dim] so RowParallelLinear sees the
        # expected THD-with-batch-1 layout (matches Megatron's SelfAttention
        # which does the same unsqueeze via `packed_seq_params` handling).
        core_attn_out = core_attn_out.unsqueeze(1)

        output, bias = self.linear_proj(core_attn_out)
        return output, bias


# ---------------------------------------------------------------------------
# Spec factory
# ---------------------------------------------------------------------------

def get_qwen3_moe_aligned_spec(args, config, vp_stage):
    """Build a Qwen3-30B-A3B transformer block spec that replaces the default
    self_attention with :class:`Qwen3MoeAlignedSelfAttention` while keeping the
    MLP / MoE path (DeepGEMM forward hook aligns it with SGLang).
    """

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    # For every local layer, swap self_attention. input_layernorm stays as the
    # default (which for TE spec is IdentityOp because norm is fused into
    # linear_qkv). We must therefore also swap input_layernorm to a real norm
    # so the FP32-residual-sum RMSNorm branch in transformer_layer can read
    # `.weight`, AND swap linear_qkv to plain ColumnParallelLinear so the
    # normalized input is not re-normalized by TE.
    # Use TENorm which honours config.normalization ("RMSNorm" for Qwen3).
    # v11: TENorm replaced by _SGLangBatchInvariantRMSNorm (batch-invariant kernel).
    from megatron.core.transformer.identity_op import IdentityOp

    self_attn_module_spec = ModuleSpec(
        module=Qwen3MoeAlignedSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=Qwen3MoeAlignedSelfAttentionSubmodules(
            linear_qkv=ColumnParallelLinear,
            linear_proj=RowParallelLinear,
            q_layernorm=IdentityOp,  # unused; per-head norm is built internally
            k_layernorm=IdentityOp,
        ),
    )

    for layer_id in range(num_layers_to_build):
        layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
        layer_specs.submodules.self_attention = self_attn_module_spec
        # Ensure input_layernorm is a real RMSNorm (has .weight) instead of
        # the TE IdentityOp that fuses into linear_qkv.
        layer_specs.submodules.input_layernorm = _SGLangBatchInvariantRMSNorm
        transformer_layer_spec.layer_specs[layer_id] = layer_specs

    import os as _os
    if _os.environ.get("QWEN3_ALIGNED_PROBE", "0") == "1":
        _norm = transformer_layer_spec.layer_specs[0].submodules.input_layernorm
        _sa = transformer_layer_spec.layer_specs[0].submodules.self_attention
        print(f"[QWEN3-ALIGNED v12-PROBE] spec factory returning: input_layernorm={_norm}, self_attention={_sa}", flush=True)
    return transformer_layer_spec
