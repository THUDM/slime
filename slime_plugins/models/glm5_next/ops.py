"""Differentiable GLM-5.3 Flash operations (mcore_vlm-slime conventions)."""

import torch
import torch.nn.functional as F


class ClampedSwiGLU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.limit = config.activation_func_clamp_value

    @torch.compile(dynamic=True)
    def forward(self, x):
        gate, up = x.float().chunk(2, dim=-1)
        return (F.silu(gate.clamp(max=self.limit)) * up.clamp(-self.limit, self.limit)).to(x.dtype)


def mhc_pre(x, weight, scale, bias, n, norm_eps, eps, iterations):
    """Aggregate residual streams and compute the Sinkhorn mixing matrix.

    The projection precedes RMS scaling, and the first Sinkhorn iteration
    adds epsilon after row softmax. Both details match the checkpoint's
    TileKernels implementation and SGLang's inference kernel.
    """
    xf = x.float()
    mixes = F.linear(xf, weight.float()) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
    pre = torch.sigmoid(mixes[..., :n] * scale[0] + bias[:n]) + eps
    post = 2 * torch.sigmoid(mixes[..., n : 2 * n] * scale[1] + bias[n : 2 * n])
    residual = (mixes[..., 2 * n :] * scale[2] + bias[2 * n :]).unflatten(-1, (n, n))
    residual = residual.softmax(-1) + eps
    residual = residual / (residual.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        residual = residual / (residual.sum(-1, keepdim=True) + eps)
        residual = residual / (residual.sum(-2, keepdim=True) + eps)
    streams = xf.unflatten(-1, (n, -1))
    layer_input = (streams * pre.unsqueeze(-1)).sum(-2).to(x.dtype)
    return layer_input, post, residual


def mhc_post(y, x, post, residual):
    n = post.shape[-1]
    streams = x.float().unflatten(-1, (n, -1))
    output = residual.transpose(-1, -2) @ streams
    output = output + post.unsqueeze(-1) * y.float().unsqueeze(-2)
    return output.flatten(-2).to(x.dtype)


@torch.no_grad()
def pooled_indexer_topk(q, k, weights, gates, ape, cu_seqlens, topk, pool_size, *, query_start=0):
    """Select complete compressed pools and always include the causal tail.

    Keys/gates cover the full packed stream; queries may be one contiguous CP
    interval starting at query_start. Chunking queries bounds temporary score
    storage independently of the total sequence length. Indexer parameters
    are frozen, as in the reference RL recipe.
    """
    boundaries = cu_seqlens.tolist()
    width = ((topk + pool_size - 1 + 63) // 64) * 64
    indices = torch.full((q.shape[0], 1, width), -1, dtype=torch.int32, device=q.device)
    offsets = torch.arange(pool_size, device=q.device)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        local_start = max(start - query_start, 0)
        local_end = min(end - query_start, q.shape[0])
        if local_end <= local_start:
            continue
        length = end - start
        groups = length // pool_size
        if groups:
            group_k = k[start : start + groups * pool_size].view(groups, pool_size, -1)
            group_gate = gates[start : start + groups * pool_size].float().view(groups, pool_size, -1)
            group_weight = (group_gate + ape.float()).softmax(1)
            pooled_k = (group_k.float() * group_weight).sum(1).to(k.dtype)
        for first in range(local_start, local_end, 128):
            last = min(first + 128, local_end)
            positions = torch.arange(first + query_start - start + 1, last + query_start - start + 1, device=q.device)
            complete = positions // pool_size
            count = min(groups, topk // pool_size)
            if count:
                scores = torch.einsum("qhd,gd->qhg", q[first:last].float(), pooled_k.float()).relu_()
                scores = (scores * weights[first:last].float().unsqueeze(-1)).sum(1)
                scores.masked_fill_(torch.arange(groups, device=q.device)[None] >= complete[:, None], -torch.inf)
                selected = scores.topk(count, dim=-1).indices
                expanded = selected.unsqueeze(-1) * pool_size + offsets
                expanded = expanded + start
                valid = selected < complete[:, None]
                expanded.masked_fill_(~valid.unsqueeze(-1), -1)
                indices[first:last, 0, : count * pool_size] = expanded.flatten(1).int()
            tail = complete[:, None] * pool_size + offsets[:-1]
            tail = torch.where(tail < positions[:, None], tail + start, -1)
            indices[first:last, 0, topk : topk + pool_size - 1] = tail.int()
    return indices
