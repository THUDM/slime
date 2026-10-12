"""Fused FP32 parameter updates for the stateless Adam optimizer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _stateless_adam_kernel(
    param,
    grad,
    size,
    step_size,
    denominator_scale,
    eps,
    weight_decay,
    decay_factor,
    MAXIMIZE: tl.constexpr,
    ADAM_W_MODE: tl.constexpr,
    HAS_WEIGHT_DECAY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    p = tl.load(param + offsets, mask=mask, other=0.0)
    g = tl.load(grad + offsets, mask=mask, other=0.0)
    if MAXIMIZE:
        g = -g
    if HAS_WEIGHT_DECAY:
        if ADAM_W_MODE:
            p = p * decay_factor
        else:
            g = tl.fma(weight_decay, p, g)
    denominator = tl.abs(g) * denominator_scale + eps
    # Match addcdiv_'s fused multiply-add without fusing the separate eager
    # operations that construct the denominator or apply decoupled decay.
    p = tl.fma(step_size, tl.div_rn(g, denominator), p)
    tl.store(param + offsets, p, mask=mask)


def stateless_adam_update(
    param,
    grad,
    *,
    lr,
    numerator_scale,
    denominator_scale,
    eps,
    weight_decay,
    adam_w_mode,
    maximize,
):
    if param.numel() == 0:
        return
    with torch.cuda.device(param.device):
        _stateless_adam_kernel[(triton.cdiv(param.numel(), 1024),)](
            param,
            grad,
            param.numel(),
            -lr * numerator_scale,
            denominator_scale,
            eps,
            weight_decay,
            1.0 - lr * weight_decay,
            MAXIMIZE=maximize,
            ADAM_W_MODE=adam_w_mode,
            HAS_WEIGHT_DECAY=weight_decay != 0,
            BLOCK_SIZE=1024,
            enable_fp_fusion=False,
        )
    torch.autograd.graph.increment_version(param)
