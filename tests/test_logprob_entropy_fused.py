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

    # mpu stub for the cp1 (no context-parallel) path used by loss.py helpers.
    mpu = types.ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_world_size = lambda: 1
    mpu.get_context_parallel_rank = lambda: 0
    mpu.get_tensor_model_parallel_group = lambda: _FakeSingleRankGroup()

    fused.fused_vocab_parallel_cross_entropy = fused_vocab_parallel_cross_entropy
    utils.VocabUtility = VocabUtility
    fusions.fused_cross_entropy = fused
    tensor_parallel.utils = utils
    core.fusions = fusions
    core.tensor_parallel = tensor_parallel
    core.mpu = mpu
    megatron.core = core
    sys.modules["megatron.core.fusions.fused_cross_entropy"] = fused
    sys.modules["megatron.core.tensor_parallel.utils"] = utils
    sys.modules["megatron.core.mpu"] = mpu


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


def test_fused_entropy_only_backward_matches_naive_reference():
    _install_megatron_stubs()
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy

    logits, tokens = _make_inputs(requires_grad=True)
    with _single_rank_all_reduce():
        log_probs, entropy = calculate_log_probs_and_entropy(logits, tokens, _FakeSingleRankGroup(), with_entropy=True)
        # Only entropy contributes to the loss, so autograd passes
        # grad_log_prob=None into the fused backward, exercising the
        # entropy-only branch.
        entropy.float().sum().backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    _, ref_entropy = _naive_log_probs_and_entropy(ref_logits, tokens)
    ref_entropy.float().sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
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


# ---------------------------------------------------------------------------
# Response-only / loss-mask gather (shrinks the [T, V] logits tensor before CE)
# ---------------------------------------------------------------------------


def _packed_thd_inputs(requires_grad: bool = False):
    """A small cp1/thd packed batch: logits [1, T, V] + per-sample token/length lists."""
    torch.manual_seed(7)
    total_lengths = [6, 5, 7]
    response_lengths = [3, 2, 4]
    vocab = 17
    T = sum(total_lengths)
    logits = torch.randn(1, T, vocab, dtype=torch.float32, requires_grad=requires_grad)
    unconcat_tokens = [torch.randint(0, vocab, (tl,), dtype=torch.long) for tl in total_lengths]
    return logits, unconcat_tokens, total_lengths, response_lengths


def _thd_args(response_only: bool, chunk_size: int = -1, loss_mask_only: bool = False):
    import types as _t

    return _t.SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        log_probs_chunk_size=chunk_size,
        log_probs_response_only=response_only,
        log_probs_loss_mask_only=loss_mask_only,
        allgather_cp=False,
    )


def _run_get_log_probs(
    logits, unconcat_tokens, total_lengths, response_lengths, args, with_entropy, full_loss_mask=None
):
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    _, res = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        max_seq_lens=None,
        full_loss_mask=full_loss_mask,
    )
    return res


@pytest.mark.parametrize("with_entropy", [False, True])
@pytest.mark.parametrize("chunk_size", [-1, 4])
def test_response_only_matches_full_path(with_entropy: bool, chunk_size: int):
    _install_megatron_stubs()

    logits, toks, tl, rl = _packed_thd_inputs()
    with _single_rank_all_reduce():
        full = _run_get_log_probs(logits.clone(), toks, tl, rl, _thd_args(False, chunk_size), with_entropy)
        gathered = _run_get_log_probs(logits.clone(), toks, tl, rl, _thd_args(True, chunk_size), with_entropy)

    assert len(full["log_probs"]) == len(gathered["log_probs"]) == len(tl)
    for a, b in zip(full["log_probs"], gathered["log_probs"], strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    if with_entropy:
        for a, b in zip(full["entropy"], gathered["entropy"], strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_response_only_backward_matches_full_path():
    _install_megatron_stubs()

    def _loss_and_grad(response_only: bool):
        logits, toks, tl, rl = _packed_thd_inputs(requires_grad=True)
        with _single_rank_all_reduce():
            res = _run_get_log_probs(logits, toks, tl, rl, _thd_args(response_only), with_entropy=True)
            loss = sum(lp.float().sum() for lp in res["log_probs"])
            loss = loss + 0.13 * sum(e.float().sum() for e in res["entropy"])
            loss.backward()
        return logits.grad

    full_grad = _loss_and_grad(False)
    gathered_grad = _loss_and_grad(True)
    # Non-response rows have no path to the loss in either case -> zero grad in both.
    torch.testing.assert_close(gathered_grad, full_grad, atol=4e-3, rtol=4e-3)


def test_loss_mask_only_zeros_masked_positions():
    _install_megatron_stubs()

    logits, toks, tl, rl = _packed_thd_inputs()
    T = logits.size(1)
    # mask aligned to the logits layout; drop half the response positions
    torch.manual_seed(0)
    full_loss_mask = (torch.rand(T) > 0.5).to(torch.float32)

    with _single_rank_all_reduce():
        full = _run_get_log_probs(logits.clone(), toks, tl, rl, _thd_args(False), with_entropy=False)
        masked = _run_get_log_probs(
            logits.clone(),
            toks,
            tl,
            rl,
            _thd_args(True, loss_mask_only=True),
            with_entropy=False,
            full_loss_mask=full_loss_mask,
        )

    from slime.backends.megatron_utils.loss import _response_keep_index

    keep = _response_keep_index(tl, rl, "thd", None, False, logits.device, T)
    # walk per-sample windows; kept(mask==1) positions equal full path, dropped positions are 0
    pos = 0
    for s_full, s_masked, length in zip(full["log_probs"], masked["log_probs"], rl, strict=True):
        for j in range(length):
            keep_pos = int(keep[pos + j])
            if full_loss_mask[keep_pos] > 0:
                torch.testing.assert_close(s_masked[j], s_full[j], atol=1e-6, rtol=1e-6)
            else:
                assert s_masked[j].item() == 0.0
        pos += length


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
