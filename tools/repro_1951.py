#!/usr/bin/env python
"""Synthetic reproducer for issue #1951 log-prob/entropy memory peaks."""

from __future__ import annotations

import argparse
import sys
import types
from contextlib import contextmanager

import torch
import torch.distributed as dist


class _FakeSingleRankGroup:
    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1


def _install_mock_fused_cross_entropy() -> None:
    """Install a single-rank Megatron fused CE stand-in for local repro runs."""
    megatron = sys.modules.setdefault("megatron", types.ModuleType("megatron"))
    core = sys.modules.setdefault("megatron.core", types.ModuleType("megatron.core"))
    fusions = sys.modules.setdefault("megatron.core.fusions", types.ModuleType("megatron.core.fusions"))
    fused = types.ModuleType("megatron.core.fusions.fused_cross_entropy")
    tensor_parallel = sys.modules.setdefault(
        "megatron.core.tensor_parallel", types.ModuleType("megatron.core.tensor_parallel")
    )
    utils = types.ModuleType("megatron.core.tensor_parallel.utils")

    class VocabUtility:
        @staticmethod
        def vocab_range_from_per_partition_vocab_size(partition_vocab_size: int, rank: int, world_size: int):
            assert world_size == 1
            assert rank == 0
            return 0, partition_vocab_size

    class _MockVocabParallelCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits: torch.Tensor, target: torch.Tensor, process_group):
            del process_group
            logits = logits.float()
            logits_max = logits.max(dim=-1, keepdim=True).values
            logits.sub_(logits_max)
            predicted_logits = logits.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            torch.exp(logits, out=logits)
            sum_exp_logits = logits.sum(dim=-1)
            logits.div_(sum_exp_logits.unsqueeze(-1))
            ctx.save_for_backward(logits, target)
            return sum_exp_logits.log() - predicted_logits

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            softmax, target = ctx.saved_tensors
            grad_input = softmax.clone()
            grad_input.scatter_add_(
                dim=-1,
                index=target.unsqueeze(-1),
                src=-torch.ones_like(target, dtype=grad_input.dtype).unsqueeze(-1),
            )
            grad_input.mul_(grad_output.unsqueeze(-1))
            return grad_input.to(torch.bfloat16), None, None

    def fused_vocab_parallel_cross_entropy(logits: torch.Tensor, target: torch.Tensor, process_group):
        return _MockVocabParallelCrossEntropy.apply(logits, target, process_group)

    fused.fused_vocab_parallel_cross_entropy = fused_vocab_parallel_cross_entropy
    utils.VocabUtility = VocabUtility
    fusions.fused_cross_entropy = fused
    tensor_parallel.utils = utils
    core.fusions = fusions
    core.tensor_parallel = tensor_parallel
    megatron.core = core
    sys.modules["megatron.core.fusions.fused_cross_entropy"] = fused
    sys.modules["megatron.core.tensor_parallel.utils"] = utils


@contextmanager
def _single_rank_all_reduce():
    original_all_reduce = dist.all_reduce

    def all_reduce(tensor, op=None, group=None, async_op=False):
        del tensor, op, group
        if async_op:
            raise NotImplementedError("async all_reduce is not needed by this repro")
        return None

    dist.all_reduce = all_reduce
    try:
        yield
    finally:
        dist.all_reduce = original_all_reduce


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", "-B", type=int, default=4096, help="Number of token positions.")
    parser.add_argument("--vocab", "-V", type=int, default=151936, help="Vocabulary dimension.")
    parser.add_argument("--chunk-size", type=int, default=-1, help="Forwarded to calculate_log_probs_and_entropy.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--with-entropy", action="store_true", help="Also compute entropy.")
    parser.add_argument("--backward", action="store_true", help="Run backward through the returned tensors.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-real-megatron", action="store_true", help="Do not install the local fused CE mock.")
    return parser.parse_args()


def _fmt_bytes(value: int) -> str:
    return f"{value / 1024**3:.3f} GiB ({value} bytes)"


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for peak-memory measurement")

    if not args.use_real_megatron:
        _install_mock_fused_cross_entropy()

    from slime.utils.ppo_utils import calculate_log_probs_and_entropy

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    logits = torch.randn(args.batch, args.vocab, device=device, dtype=dtype)
    if args.backward:
        logits.requires_grad_(True)
    tokens = torch.randint(args.vocab, (args.batch,), device=device)
    torch.cuda.synchronize()

    allocated_after_logits = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    with _single_rank_all_reduce():
        log_probs, entropy = calculate_log_probs_and_entropy(
            logits,
            tokens,
            _FakeSingleRankGroup(),
            with_entropy=args.with_entropy,
            chunk_size=args.chunk_size,
        )
        if args.backward:
            loss = log_probs.float().sum()
            if entropy is not None:
                loss = loss + entropy.float().sum()
            loss.backward()

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    current = torch.cuda.memory_allocated()

    print(f"shape=({args.batch}, {args.vocab}) dtype={args.dtype} with_entropy={args.with_entropy}")
    print(f"chunk_size={args.chunk_size} backward={args.backward} mock_megatron={not args.use_real_megatron}")
    print(f"allocated_after_logits={_fmt_bytes(allocated_after_logits)}")
    print(f"peak_during_call={_fmt_bytes(peak)}")
    print(f"peak_delta_after_logits={_fmt_bytes(peak - allocated_after_logits)}")
    print(f"allocated_after_call={_fmt_bytes(current)}")


if __name__ == "__main__":
    main()
