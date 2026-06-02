from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from _cp_dist_helpers import free_port


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUM_GPUS = 0


class _FakeSingleRankGroup:
    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1


def _install_megatron_stubs() -> None:
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
            assert world_size > 0
            assert 0 <= rank < world_size
            start = rank * partition_vocab_size
            return start, start + partition_vocab_size

    class _MockVocabParallelCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits: torch.Tensor, target: torch.Tensor, process_group):
            del process_group
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
            raise NotImplementedError("async all_reduce is not needed by this test")
        return None

    dist.all_reduce = all_reduce
    try:
        yield
    finally:
        dist.all_reduce = original_all_reduce


def _naive_log_probs_and_entropy(logits: torch.Tensor, tokens: torch.Tensor):
    log_softmax = torch.log_softmax(logits.float(), dim=-1)
    log_probs = log_softmax.gather(dim=-1, index=tokens.unsqueeze(-1))
    entropy = -(log_softmax.exp() * log_softmax).sum(dim=-1)
    return log_probs, entropy


def _make_inputs(requires_grad: bool = False):
    torch.manual_seed(1234)
    logits = torch.randn(9, 17, dtype=torch.float32)
    logits.requires_grad_(requires_grad)
    tokens = torch.randint(0, logits.size(-1), (logits.size(0),), dtype=torch.long)
    return logits, tokens


def test_fused_forward_matches_naive_reference():
    _install_megatron_stubs()
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy

    logits, tokens = _make_inputs()
    with _single_rank_all_reduce():
        log_probs, entropy = calculate_log_probs_and_entropy(
            logits.clone(), tokens, _FakeSingleRankGroup(), with_entropy=True
        )

    ref_log_probs, ref_entropy = _naive_log_probs_and_entropy(logits, tokens)
    torch.testing.assert_close(log_probs, ref_log_probs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(entropy, ref_entropy, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("with_entropy", [False, True])
def test_chunked_matches_unchunked(with_entropy: bool):
    _install_megatron_stubs()
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy

    logits, tokens = _make_inputs()
    with _single_rank_all_reduce():
        full_log_probs, full_entropy = calculate_log_probs_and_entropy(
            logits.clone(), tokens, _FakeSingleRankGroup(), with_entropy=with_entropy, chunk_size=-1
        )
        chunk_log_probs, chunk_entropy = calculate_log_probs_and_entropy(
            logits.clone(), tokens, _FakeSingleRankGroup(), with_entropy=with_entropy, chunk_size=4
        )

    torch.testing.assert_close(chunk_log_probs, full_log_probs, atol=1e-5, rtol=1e-5)
    if with_entropy:
        torch.testing.assert_close(chunk_entropy, full_entropy, atol=1e-5, rtol=1e-5)
    else:
        assert chunk_entropy is None


def test_no_entropy_chunked_backward_preserves_input_grad():
    _install_megatron_stubs()
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy

    logits, tokens = _make_inputs(requires_grad=True)
    with _single_rank_all_reduce():
        log_probs, entropy = calculate_log_probs_and_entropy(
            logits, tokens, _FakeSingleRankGroup(), with_entropy=False, chunk_size=4
        )
        assert entropy is None
        log_probs.float().sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_fused_backward_matches_naive_reference_with_bf16_tolerance():
    _install_megatron_stubs()
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy

    logits, tokens = _make_inputs(requires_grad=True)
    with _single_rank_all_reduce():
        log_probs, entropy = calculate_log_probs_and_entropy(logits, tokens, _FakeSingleRankGroup(), with_entropy=True)
        (log_probs.float().sum() + 0.13 * entropy.float().sum()).backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref_log_probs, ref_entropy = _naive_log_probs_and_entropy(ref_logits, tokens)
    (ref_log_probs.float().sum() + 0.13 * ref_entropy.float().sum()).backward()

    torch.testing.assert_close(logits.grad, ref_logits.grad, atol=4e-3, rtol=4e-3)


def _tp2_worker(rank: int, master_port: int, result_path: str) -> None:
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=2)
    try:
        _install_megatron_stubs()
        from slime.utils.ppo_utils import compute_log_probs_and_entropy

        torch.manual_seed(2024)
        batch_size = 7
        partition_vocab_size = 5
        full_vocab_size = partition_vocab_size * 2
        full_logits = torch.randn(batch_size, full_vocab_size, dtype=torch.float32)
        tokens = torch.randint(0, full_vocab_size, (batch_size,), dtype=torch.long)
        start = rank * partition_vocab_size
        local_logits = full_logits[:, start : start + partition_vocab_size].contiguous().requires_grad_(True)

        log_probs, entropy = compute_log_probs_and_entropy(local_logits, tokens, dist.group.WORLD)
        (log_probs.float().sum() + 0.13 * entropy.float().sum()).backward()

        gathered_grads = [torch.empty_like(local_logits.grad) for _ in range(2)]
        dist.all_gather(gathered_grads, local_logits.grad)

        if rank == 0:
            ref_logits = full_logits.detach().clone().requires_grad_(True)
            ref_log_probs, ref_entropy = _naive_log_probs_and_entropy(ref_logits, tokens)
            (ref_log_probs.float().sum() + 0.13 * ref_entropy.float().sum()).backward()

            torch.testing.assert_close(log_probs, ref_log_probs, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(entropy, ref_entropy, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(torch.cat(gathered_grads, dim=-1), ref_logits.grad, atol=4e-3, rtol=4e-3)
            with open(result_path, "w") as f:
                f.write("ok")
    finally:
        dist.destroy_process_group()


def test_tp2_fused_backward_matches_full_vocab_reference(tmp_path):
    result_path = str(tmp_path / "tp2_result.txt")
    mp.spawn(_tp2_worker, args=(free_port(), result_path), nprocs=2, join=True)
    assert (tmp_path / "tp2_result.txt").read_text() == "ok"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
