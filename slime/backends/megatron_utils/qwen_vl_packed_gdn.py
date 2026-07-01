"""Packed-sequence GDN patch for Qwen3.5/3.6 VL Bridge models.

Megatron-Bridge's Qwen3.5 VL providers build linear-attention layers with
Megatron-Core ``GatedDeltaNet``.  Slime's Qwen3.5 text plugin already supports
packed varlen GDN via FLA/FlashQLA, but Bridge VLM models bypass that plugin and
hit Megatron-Core's packed-sequence guard.  This patch keeps the Bridge weight
layout intact and only fills in the packed forward path.
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _get_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _gdn_timing_enabled(module) -> bool:
    if not _env_flag("SLIME_QWENVL_GDN_TIMING"):
        return False
    rank_filter = _env_int("SLIME_QWENVL_GDN_TIMING_RANK", 0)
    return rank_filter < 0 or _get_rank() == rank_filter


def _gdn_timing_should_sample(module) -> bool:
    if not _gdn_timing_enabled(module):
        return False
    count = getattr(module, "_slime_gdn_timing_count", 0) + 1
    module._slime_gdn_timing_count = count
    limit = _env_int("SLIME_QWENVL_GDN_TIMING_LIMIT", 64)
    interval = max(1, _env_int("SLIME_QWENVL_GDN_TIMING_INTERVAL", 32))
    return count <= limit and (count == 1 or count % interval == 0)


class _CudaStepTimer:
    def __init__(self, enabled: bool):
        self.enabled = enabled and torch.cuda.is_available()
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.start_wall = time.perf_counter()

    def run(self, name, fn):
        if not self.enabled:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        self.events.append((name, start, end))
        return result

    def summary(self) -> dict[str, float]:
        if not self.enabled:
            return {"wall_ms": (time.perf_counter() - self.start_wall) * 1000.0}
        torch.cuda.synchronize()
        summary = {name: start.elapsed_time(end) for name, start, end in self.events}
        summary["total_ms"] = sum(summary.values())
        return summary


def _get_packed_cu_seqlens(packed_seq_params):
    return (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )


def _local_cu_seqlens_for_kernel(module, cu_seqlens: torch.Tensor, total_seq_len: int) -> torch.Tensor:
    """Convert global packed cu_seqlens to the local physical sequence layout.

    Bridge Qwen3-VL builds THD packed params with padded *global* sequence
    boundaries. After CP preprocessing, each rank holds ``padded_len / cp``
    tokens per sample. FLA kernels operate on this local tensor and will
    illegal-access if they receive the full global boundary.
    """
    expected_total = int(cu_seqlens[-1].detach().item()) if cu_seqlens.numel() > 0 else total_seq_len
    if expected_total == total_seq_len:
        return cu_seqlens

    cp_size = max(int(getattr(module, "cp_size", 1)), 1)
    if expected_total % cp_size == 0 and expected_total // cp_size == total_seq_len:
        return torch.div(cu_seqlens, cp_size, rounding_mode="floor").to(dtype=cu_seqlens.dtype)

    if not getattr(module, "_slime_gdn_cu_length_warned", False):
        logger.warning(
            "Packed GDN cu_seqlens mismatch: cu_seqlens[-1]=%s physical_seq_len=%s cp_size=%s; "
            "falling back to one local range.",
            expected_total,
            total_seq_len,
            cp_size,
        )
        module._slime_gdn_cu_length_warned = True
    return torch.tensor([0, total_seq_len], dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def _get_gated_delta_rule(module):
    backend = "fla"
    try:
        from megatron.training.global_vars import get_args

        backend = getattr(get_args(), "qwen_gdn_backend", backend)
    except Exception:
        pass

    if getattr(module, "_slime_gdn_backend", None) == backend and hasattr(module, "_slime_gated_delta_rule"):
        return module._slime_gated_delta_rule

    try:
        from slime_plugins.models.qwen_gdn_backend import get_chunk_gated_delta_rule

        rule = get_chunk_gated_delta_rule(backend)
    except Exception as exc:
        if backend != "fla":
            logger.warning("Qwen GDN backend %s is unavailable: %s; falling back to fla.", backend, exc)
            backend = "fla"
        rule = module.gated_delta_rule

    module._slime_gdn_backend = backend
    module._slime_gated_delta_rule = rule
    if not getattr(module, "_slime_gdn_backend_logged", False):
        logger.info("Qwen VL packed GDN backend selected: %s", backend)
        module._slime_gdn_backend_logged = True
    return rule


def _packed_ranges_from_cu_seqlens(cu_seqlens: torch.Tensor, total_seq_len: int) -> list[tuple[int, int]]:
    """Return clipped packed sequence ranges covering the physical qkv tensor."""
    values = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
    if not values:
        return [(0, total_seq_len)]
    if values[0] != 0:
        values = [0, *values]

    ranges: list[tuple[int, int]] = []
    prev = 0
    for value in values[1:]:
        end = min(max(int(value), prev), total_seq_len)
        if end > prev:
            ranges.append((prev, end))
        prev = end

    if prev < total_seq_len:
        ranges.append((prev, total_seq_len))
    return ranges or [(0, total_seq_len)]


def _range_stats(ranges: list[tuple[int, int]]) -> dict[str, int | float]:
    lengths = [end - start for start, end in ranges]
    if not lengths:
        return {"ranges": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "ranges": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def _safe_packed_causal_conv1d(module, qkv, conv1d_weight, conv1d_bias, cu_seqlens):
    """Torch fallback for packed Qwen GDN conv.

    FLA's Triton causal_conv1d varlen path can illegal-access on short padded
    packed sequences. The grouped torch conv path matches Megatron-Core's
    deterministic implementation and resets convolution state at each packed
    sequence boundary.
    """
    total_seq_len = qkv.shape[1]
    outputs = []
    groups = module.conv_dim_local_tp // module.cp_size
    for start, end in _packed_ranges_from_cu_seqlens(cu_seqlens, total_seq_len):
        segment = qkv[:, start:end, :].transpose(1, 2).contiguous()
        conv_out = F.conv1d(
            input=segment,
            weight=conv1d_weight,
            bias=conv1d_bias,
            stride=module.conv1d.stride,
            padding=module.conv1d.padding,
            dilation=module.conv1d.dilation,
            groups=groups,
        )
        conv_out = module.act_fn(conv_out[..., : end - start])
        outputs.append(conv_out.transpose(1, 2))

    return torch.cat(outputs, dim=1).contiguous()


def _as_1d_int(value) -> int:
    if isinstance(value, tuple):
        return int(value[0])
    return int(value)


def _batched_safe_packed_causal_conv1d(module, qkv, conv1d_weight, conv1d_bias, cu_seqlens):
    """Run the safe packed conv as one torch conv with zero gaps between ranges."""
    total_seq_len = qkv.shape[1]
    ranges = _packed_ranges_from_cu_seqlens(cu_seqlens, total_seq_len)
    if len(ranges) <= 1:
        return _safe_packed_causal_conv1d(module, qkv, conv1d_weight, conv1d_bias, cu_seqlens)

    kernel_size = _as_1d_int(module.conv1d.kernel_size)
    dilation = _as_1d_int(module.conv1d.dilation)
    padding = _as_1d_int(module.conv1d.padding)
    reset_gap = max(padding, dilation * (kernel_size - 1))

    parts = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    zero_gap = qkv.new_zeros((qkv.shape[0], reset_gap, qkv.shape[2])) if reset_gap > 0 else None
    for idx, (start, end) in enumerate(ranges):
        if idx > 0 and zero_gap is not None:
            parts.append(zero_gap)
            cursor += reset_gap
        length = end - start
        parts.append(qkv[:, start:end, :])
        spans.append((cursor, length))
        cursor += length

    qkv_padded = torch.cat(parts, dim=1)
    groups = module.conv_dim_local_tp // module.cp_size
    conv_out = F.conv1d(
        input=qkv_padded.transpose(1, 2).contiguous(),
        weight=conv1d_weight,
        bias=conv1d_bias,
        stride=module.conv1d.stride,
        padding=module.conv1d.padding,
        dilation=module.conv1d.dilation,
        groups=groups,
    )
    conv_out = module.act_fn(conv_out[..., : qkv_padded.shape[1]]).transpose(1, 2)
    return torch.cat([conv_out[:, start : start + length, :] for start, length in spans], dim=1).contiguous()


def _fast_packed_causal_conv1d(module, qkv, conv1d_weight, conv1d_bias, cu_seqlens):
    from fla.modules.convolution import causal_conv1d

    out, _ = causal_conv1d(
        x=qkv.contiguous(),
        weight=conv1d_weight.squeeze(1).contiguous(),
        bias=conv1d_bias,
        activation="silu",
        cu_seqlens=cu_seqlens,
    )
    return out.contiguous()


def apply_qwen_vl_packed_gdn_patch() -> None:
    try:
        from megatron.core.ssm import gated_delta_net as gdn_mod
        from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push
    except ImportError:
        return

    GatedDeltaNet = gdn_mod.GatedDeltaNet
    if getattr(GatedDeltaNet, "_slime_qwen_vl_packed_patch", False):
        return

    original_forward = GatedDeltaNet.forward

    def patched_forward(
        self,
        hidden_states,
        attention_mask,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        if packed_seq_params is None:
            return original_forward(
                self,
                hidden_states,
                attention_mask,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
                **kwargs,
            )

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context is not None:
            raise NotImplementedError("GDN does not support inference for now.")
        if self.config.deterministic_mode:
            raise NotImplementedError("Packed GDN requires the FLA/FlashQLA kernel path.")

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size
        cu_seqlens = _get_packed_cu_seqlens(packed_seq_params)
        local_cu_seqlens = _local_cu_seqlens_for_kernel(self, cu_seqlens, seq_len)
        should_time = _gdn_timing_should_sample(self)
        timer = _CudaStepTimer(should_time)
        packed_ranges = None
        if should_time:
            packed_ranges = _packed_ranges_from_cu_seqlens(local_cu_seqlens, seq_len)

        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = timer.run("in_proj", lambda: self.in_proj(hidden_states))
        nvtx_range_pop(suffix="in_proj")

        qkvzba = timer.run(
            "cp2hp",
            lambda: gdn_mod.tensor_a2a_cp2hp(
                qkvzba,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
                split_sections=[
                    self.qk_dim_local_tp,
                    self.qk_dim_local_tp,
                    self.v_dim_local_tp,
                    self.v_dim_local_tp,
                    self.num_value_heads // self.tp_size,
                    self.num_value_heads // self.tp_size,
                ],
            ),
        )

        qkvzba = qkvzba.transpose(0, 1)
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.num_value_heads // self.tp_size // self.cp_size,
                self.num_value_heads // self.tp_size // self.cp_size,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)

        nvtx_range_push(suffix="conv1d")
        local_seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = gdn_mod.get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            gdn_mod.get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        use_fast_conv = _env_flag("SLIME_QWENVL_GDN_FAST_CONV")
        if use_fast_conv and not getattr(self, "_slime_gdn_fast_conv_logged", False):
            logger.info(
                "Qwen VL packed GDN uses FLA causal_conv1d fast path with backend=%s.",
                os.environ.get("SLIME_QWENVL_GDN_FAST_CONV_BACKEND", "triton"),
            )
            self._slime_gdn_fast_conv_logged = True
        use_batched_conv = (not use_fast_conv) and _env_flag("SLIME_QWENVL_GDN_BATCHED_CONV")
        if use_batched_conv and not getattr(self, "_slime_gdn_batched_conv_logged", False):
            logger.info("Qwen VL packed GDN uses batched safe torch causal conv.")
            self._slime_gdn_batched_conv_logged = True
        qkv = timer.run(
            "conv1d",
            lambda: (
                _fast_packed_causal_conv1d(self, qkv, conv1d_weight, conv1d_bias, local_cu_seqlens)
                if use_fast_conv
                else (
                    _batched_safe_packed_causal_conv1d(self, qkv, conv1d_weight, conv1d_bias, local_cu_seqlens)
                    if use_batched_conv
                    else _safe_packed_causal_conv1d(self, qkv, conv1d_weight, conv1d_bias, local_cu_seqlens)
                )
            ),
        )
        qkv = qkv[:, :local_seq_len]
        nvtx_range_pop(suffix="conv1d")

        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        use_l2norm_in_kernel = self.use_qk_l2norm and _env_flag("SLIME_QWENVL_GDN_L2NORM_IN_KERNEL")
        if use_l2norm_in_kernel and not getattr(self, "_slime_gdn_l2norm_in_kernel_logged", False):
            logger.info("Qwen VL packed GDN uses kernel-side QK L2 norm.")
            self._slime_gdn_l2norm_in_kernel_logged = True
        if self.use_qk_l2norm and not use_l2norm_in_kernel:
            query_key = timer.run("qk_l2norm", lambda: gdn_mod.l2norm(query_key.contiguous()))

        query, key = torch.split(
            query_key,
            [
                self.qk_dim_local_tp // self.key_head_dim // self.cp_size,
                self.qk_dim_local_tp // self.key_head_dim // self.cp_size,
            ],
            dim=2,
        )
        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        nvtx_range_push(suffix="g_and_beta")
        A_log_local_cp, dt_bias_local_cp = timer.run(
            "g_params",
            lambda: (
                gdn_mod.get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp),
                gdn_mod.get_parameter_local_cp(
                    self.dt_bias,
                    dim=0,
                    cp_group=self.pg_collection.cp,
                ),
            ),
        )
        g, beta = timer.run(
            "g_and_beta",
            lambda: (
                -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp),
                beta.sigmoid(),
            ),
        )
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = timer.run(
            "gated_delta_rule",
            lambda: _get_gated_delta_rule(self)(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=use_l2norm_in_kernel,
                cu_seqlens=local_cu_seqlens,
            ),
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        nvtx_range_push(suffix="gated_norm")
        norm_out = timer.run("gated_norm", lambda: self._apply_gated_norm(core_attn_out, gate))
        nvtx_range_pop(suffix="gated_norm")

        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        norm_out = timer.run(
            "hp2cp",
            lambda: gdn_mod.tensor_a2a_hp2cp(
                norm_out,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
            ),
        )

        nvtx_range_push(suffix="out_proj")
        out, out_bias = timer.run("out_proj", lambda: self.out_proj(norm_out))
        nvtx_range_pop(suffix="out_proj")

        if should_time:
            logger.info(
                "Qwen VL packed GDN timing: sample=%s batch=%s seq_len=%s local_seq_len=%s ranges=%s times_ms=%s",
                getattr(self, "_slime_gdn_timing_count", 0),
                batch,
                seq_len,
                local_seq_len,
                _range_stats(packed_ranges or []),
                {key: round(value, 3) for key, value in timer.summary().items()},
            )

        return out, out_bias

    GatedDeltaNet.forward = patched_forward
    GatedDeltaNet._slime_qwen_vl_packed_patch = True
    logger.info("Patched Megatron-Core GatedDeltaNet packed forward for Qwen VL Bridge.")


apply_qwen_vl_packed_gdn_patch()
