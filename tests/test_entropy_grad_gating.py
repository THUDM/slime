"""Unit tests for entropy gradient gating in ``calculate_log_probs_and_entropy``.

Background
----------
Entropy enters the policy loss as ``loss = pg_loss - args.entropy_coef * entropy_loss``
(see ``policy_loss_function`` in ``slime/backends/megatron_utils/loss.py``). When
``entropy_coef == 0`` the entropy term contributes no gradient, yet the entropy was still
computed *with* autograd enabled — retaining the ``[num_tokens, vocab]`` entropy graph and a
defensive ``logits.clone()`` per chunk. For long multi-turn rollouts that activation memory
dominates and can OOM.

The fix adds a ``need_entropy_grad`` flag: when ``False`` the entropy is computed under
``torch.no_grad()`` and the clone is skipped. These tests pin the contract:

1. The entropy *values* are identical whether or not grad is tracked.
2. With ``need_entropy_grad=False`` the returned entropy carries no autograd graph.
3. With ``need_entropy_grad=True`` the entropy is differentiable w.r.t. the logits.
4. The log-prob output is unaffected by the entropy flag.

These run on CPU with a ``world_size=1`` gloo group (the TP all-reduce in the entropy kernel
is a no-op for a single rank). ``compute_log_probs`` is monkeypatched to a cheap stub so the
test does not depend on Megatron's fused cross-entropy (absent from the CPU CI image); the
gating logic under test lives entirely in the entropy branch.
"""

import os

import pytest
import torch
import torch.distributed as dist

import slime.utils.ppo_utils as ppo_utils
from slime.utils.ppo_utils import calculate_log_probs_and_entropy


@pytest.fixture(scope="module")
def single_rank_group():
    """A real ``world_size=1`` gloo group for the entropy kernel's ``all_reduce`` calls."""
    if dist.is_initialized():
        yield dist.group.WORLD
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def stub_compute_log_probs(monkeypatch):
    """Avoid the Megatron fused-CE dependency: log-probs are not what these tests exercise."""

    def _fake_log_probs(logits, tokens, process_group, keep_mask=None):
        # Shape/contract-compatible stub: one scalar log-prob per token, differentiable.
        return logits.sum(dim=-1)

    monkeypatch.setattr(ppo_utils, "compute_log_probs", _fake_log_probs)


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [-1, 4])
def test_entropy_values_match_regardless_of_grad(single_rank_group, chunk_size):
    """no_grad entropy must equal grad-tracked entropy numerically (only the graph differs)."""
    torch.manual_seed(0)
    num_tokens, vocab = 11, 32
    logits = torch.randn(num_tokens, vocab, dtype=torch.float32)
    tokens = torch.randint(0, vocab, (num_tokens,))

    _, entropy_grad = calculate_log_probs_and_entropy(
        logits, tokens, single_rank_group, with_entropy=True, chunk_size=chunk_size, need_entropy_grad=True
    )
    _, entropy_nograd = calculate_log_probs_and_entropy(
        logits, tokens, single_rank_group, with_entropy=True, chunk_size=chunk_size, need_entropy_grad=False
    )

    assert entropy_grad.shape == (num_tokens,)
    assert entropy_nograd.shape == (num_tokens,)
    torch.testing.assert_close(entropy_grad.detach(), entropy_nograd)


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [-1, 4])
def test_need_entropy_grad_false_detaches_graph(single_rank_group, chunk_size):
    """need_entropy_grad=False -> entropy is a leaf with no autograd graph (memory is freed)."""
    torch.manual_seed(1)
    num_tokens, vocab = 9, 16
    logits = torch.randn(num_tokens, vocab, dtype=torch.float32, requires_grad=True)
    tokens = torch.randint(0, vocab, (num_tokens,))

    _, entropy = calculate_log_probs_and_entropy(
        logits, tokens, single_rank_group, with_entropy=True, chunk_size=chunk_size, need_entropy_grad=False
    )

    assert entropy.grad_fn is None
    assert not entropy.requires_grad


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [-1, 4])
def test_need_entropy_grad_true_is_differentiable(single_rank_group, chunk_size):
    """need_entropy_grad=True -> entropy backpropagates to the logits (the entropy-bonus path)."""
    torch.manual_seed(2)
    num_tokens, vocab = 9, 16
    logits = torch.randn(num_tokens, vocab, dtype=torch.float32, requires_grad=True)
    tokens = torch.randint(0, vocab, (num_tokens,))

    _, entropy = calculate_log_probs_and_entropy(
        logits, tokens, single_rank_group, with_entropy=True, chunk_size=chunk_size, need_entropy_grad=True
    )

    assert entropy.requires_grad
    assert entropy.grad_fn is not None
    entropy.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [-1, 4])
def test_log_probs_unaffected_by_entropy_flag(single_rank_group, chunk_size):
    """The log-prob output must not depend on need_entropy_grad."""
    torch.manual_seed(3)
    num_tokens, vocab = 11, 16
    logits = torch.randn(num_tokens, vocab, dtype=torch.float32)
    tokens = torch.randint(0, vocab, (num_tokens,))

    log_prob_a, _ = calculate_log_probs_and_entropy(
        logits, tokens, single_rank_group, with_entropy=True, chunk_size=chunk_size, need_entropy_grad=True
    )
    log_prob_b, _ = calculate_log_probs_and_entropy(
        logits, tokens, single_rank_group, with_entropy=True, chunk_size=chunk_size, need_entropy_grad=False
    )
    torch.testing.assert_close(log_prob_a, log_prob_b)


@pytest.mark.unit
def test_empty_input_returns_empty_entropy(single_rank_group):
    """Zero-length response (all positions masked away upstream) is handled without compute."""
    logits = torch.zeros(0, 16, dtype=torch.float32)
    tokens = torch.zeros(0, dtype=torch.long)

    _, entropy = calculate_log_probs_and_entropy(
        logits, tokens, single_rank_group, with_entropy=True, need_entropy_grad=False
    )
    assert entropy.shape == (0,)
