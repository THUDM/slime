# Patch _allreduce_non_tensor_model_parallel_grads (and its legacy alias
# _allreduce_layernorm_grads) in megatron.core.distributed.finalize_model_grads
# to coalesce/all_reduce TP-side grads in size-bounded chunks instead of one
# large _flatten_dense_tensors(grads). Lowers the peak contiguous-memory
# allocation during TP-side grad sync, avoiding OOM under allocator
# fragmentation when the combined grad buffer would otherwise be very large.
# SUM/AVG are element-wise, so chunking is mathematically equivalent.
# Chunk size: SLIME_GRAD_COALESCE_CHUNK_BYTES, default 1 GiB.
#
# Cross-compatible across the Megatron-LM versions slime is run against:
# the core_v0.13.0 line (DDP config exposes `use_custom_fsdp`, `_get_main_grad_attr`
# takes `(param, use_custom_fsdp)`, target function takes `(model, config)`) and
# the post-core_v0.15.0rc7 dev line (`use_megatron_fsdp`, single-arg
# `_get_main_grad_attr`, `(model, config, tp_group)`). API differences are
# resolved at runtime — no version-conditional imports.

import inspect
import logging
import os
import sys
import warnings

logger = logging.getLogger(__name__)

try:
    import torch
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import (
        _flatten_dense_tensors,
        _get_main_grad_attr,
        _reshard_if_dtensor,
        _unflatten_dense_tensors,
        _unshard_if_dtensor,
        get_attr_wrapped_model,
    )

    # post-core_v0.15.0rc7 dev takes (param); core_v0.13.0 line takes
    # (param, use_custom_fsdp=False).
    _gma_takes_fsdp_arg = len(inspect.signature(_get_main_grad_attr).parameters) >= 2

    def _grad_attr(param, fsdp_on):
        if _gma_takes_fsdp_arg:
            return _get_main_grad_attr(param, fsdp_on)
        return _get_main_grad_attr(param)

    def _fsdp_flag(ddp_config):
        return bool(getattr(ddp_config, "use_megatron_fsdp", False) or getattr(ddp_config, "use_custom_fsdp", False))

    _chunk_bytes = int(os.environ.get("SLIME_GRAD_COALESCE_CHUNK_BYTES") or (1 << 30))

    def _split_into_chunks(params, grads, target_bytes):
        """Greedy split keeping params/grads aligned. A single grad larger
        than target_bytes is placed alone in its own chunk."""
        chunks, cur_p, cur_g, cur_b = [], [], [], 0
        for p, g in zip(params, grads, strict=False):
            gb = g.numel() * g.element_size()
            if cur_g and cur_b + gb > target_bytes:
                chunks.append((cur_p, cur_g))
                cur_p, cur_g, cur_b = [], [], 0
            cur_p.append(p)
            cur_g.append(g)
            cur_b += gb
        if cur_g:
            chunks.append((cur_p, cur_g))
        return chunks

    def _allreduce_non_tensor_model_parallel_grads(model, config, tp_group=None):
        # post-core_v0.15.0rc7 dev passes tp_group; core_v0.13.0 line omits it.
        # Default-fill from parallel_state so the same body works for both call sites.
        if tp_group is None:
            tp_group = parallel_state.get_tensor_model_parallel_group()
        if tp_group.size() <= 1:
            return

        params_sum, grads_sum = [], []
        params_avg, grads_avg = [], []
        ddp_config = None
        for model_chunk in model:
            ddp_config = model_chunk.ddp_config
            fsdp_on = _fsdp_flag(ddp_config)
            for name, param in get_attr_wrapped_model(model_chunk, "named_parameters")():
                if not param.requires_grad:
                    continue
                if getattr(param, "average_gradients_across_tp_domain", False):
                    target_params, target_grads = params_avg, grads_avg
                elif (config.sequence_parallel and getattr(param, "sequence_parallel", False)) or (
                    config.qk_layernorm and ("q_layernorm" in name or "k_layernorm" in name)
                ):
                    target_params, target_grads = params_sum, grads_sum
                else:
                    continue

                grad_attr = _grad_attr(param, fsdp_on)
                grad = getattr(param, grad_attr)
                if grad is None:
                    continue
                target_params.append(param)
                if fsdp_on and hasattr(grad, "_local_tensor"):
                    target_grads.append(grad._local_tensor.data)
                else:
                    target_grads.append(_unshard_if_dtensor(grad).data)

        for params, grads, op in (
            (params_sum, grads_sum, torch.distributed.ReduceOp.SUM),
            (params_avg, grads_avg, torch.distributed.ReduceOp.AVG),
        ):
            if not grads:
                continue
            fsdp_on = _fsdp_flag(ddp_config)
            for p_chunk, g_chunk in _split_into_chunks(params, grads, _chunk_bytes):
                coalesced = _flatten_dense_tensors(g_chunk)
                torch.distributed.all_reduce(coalesced, op=op, group=tp_group)
                for param, buf, synced in zip(
                    p_chunk, g_chunk, _unflatten_dense_tensors(coalesced, g_chunk), strict=False
                ):
                    buf.copy_(synced)
                    grad_attr = _grad_attr(param, fsdp_on)
                    orig_grad = getattr(param, grad_attr)
                    if fsdp_on and hasattr(orig_grad, "_local_tensor"):
                        # buf already aliases orig_grad._local_tensor.data;
                        # restore original DTensor wrapper (post-rc7 dev semantics).
                        setattr(param, grad_attr, orig_grad)
                    else:
                        setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))
                del coalesced

    # The parent package re-exports a same-named function, shadowing the
    # submodule attribute. Pull the real module out of sys.modules to setattr.
    _fmg = sys.modules["megatron.core.distributed.finalize_model_grads"]
    _fmg._allreduce_non_tensor_model_parallel_grads = _allreduce_non_tensor_model_parallel_grads
    _fmg._allreduce_layernorm_grads = _allreduce_non_tensor_model_parallel_grads

    logger.info(
        "slime grad coalesce patch applied to "
        "megatron.core.distributed.finalize_model_grads."
        "_allreduce_non_tensor_model_parallel_grads (chunk=%d MiB)",
        _chunk_bytes // (1 << 20),
    )

except ImportError as exc:
    warnings.warn(
        f"slime grad coalesce patch not applied — Megatron import failed ({exc!r}). "
        "If this is a Megatron upgrade, the symbol layout may have changed; "
        "without this patch, large-model TP grad sync may OOM.",
        stacklevel=2,
    )
