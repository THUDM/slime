"""CPU unit tests for ``slime.backends.megatron_utils.cp_utils``.

The reducer tests pin the per-rollout contract: a rollout split into N training
samples (compact / subagent) must contribute exactly one token-weighted
mean to the sum, even when first-fit packing puts those siblings into
different micro-batches at training time.

The routing-replay tests pin padding load distribution across samples,
context-parallel layouts, and sequence-parallel slices.

The CPU-only CI image does not ship megatron — ``_cp_dist_helpers``
stubs ``megatron.core.mpu`` at import time so the subsequent
``cp_utils`` import binds against the stub.

End-to-end report-formula invariance and multi-process distributed
checks live in ``test_metric_report.py`` and ``test_metric_report_dist.py``.
"""

from __future__ import annotations

# Import the helpers BEFORE the slime imports so the megatron stub lands
# in sys.modules first. pytest's prepend importmode puts this file's
# directory (``tests/``) on sys.path, which is what makes the bare-name
# import work without an ``__init__.py``.
import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils.cp_utils import (  # noqa: E402
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    prepare_routed_experts_for_routing_replay,
)


NUM_GPUS = 0


def _make_inputs(per_sample_lengths: list[int]):
    """Build (total_lengths, response_lengths, loss_masks) for samples of the given lengths.

    Each sample has loss_mask = all-ones (so mask sum == length); total length
    is response length + 4 fake prompt tokens (unused by the reducer in
    cp_size==1 mode).
    """
    response_lengths = list(per_sample_lengths)
    total_lengths = [r + 4 for r in response_lengths]
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    return total_lengths, response_lengths, loss_masks


def _denoms(*values: int) -> torch.Tensor:
    """Wrap per-sample denoms as the float tensor that the actor side promotes
    them to before calling the reducer."""
    return torch.tensor(values, dtype=torch.float32)


def _set_parallel_ranks(monkeypatch, *, cp_size=1, cp_rank=0, tp_size=1, tp_rank=0, ep_size=1):
    from megatron.core import mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: tp_size, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: tp_rank, raising=False)
    monkeypatch.setattr(mpu, "get_expert_model_parallel_world_size", lambda: ep_size, raising=False)


def _make_routing_replay_samples(num_samples=4, *, num_layers=4, topk=2):
    routed_experts = [torch.full((2, num_layers, topk), 7, dtype=torch.int32) for _ in range(num_samples)]
    tokens = [torch.zeros(3, dtype=torch.int64) for _ in range(num_samples)]
    return routed_experts, tokens


@pytest.mark.unit
def test_routing_replay_small_padding_balances_expert_parallel_ranks(monkeypatch):
    topk = 8
    for num_experts, ep_size in ((128, 8), (128, 32), (256, 8), (256, 32)):
        experts_per_rank = num_experts // ep_size
        for num_padding_rows in range(1, 6):
            _set_parallel_ranks(monkeypatch, ep_size=ep_size)
            routed_experts, tokens = _make_routing_replay_samples(
                num_padding_rows,
                num_layers=4,
                topk=topk,
            )

            actual = prepare_routed_experts_for_routing_replay(
                routed_experts,
                tokens,
                num_experts=num_experts,
                data_pad_size_multiplier=1,
                sequence_parallel=False,
                allgather_cp=False,
            )

            for sample_idx, expected in enumerate(routed_experts):
                torch.testing.assert_close(actual[sample_idx * 3 : sample_idx * 3 + 2], expected)

            padding = actual[torch.arange(2, num_padding_rows * 3, 3)]
            assert padding.dtype == torch.int32
            assert torch.all((padding >= 0) & (padding < num_experts))
            sorted_padding = padding.sort(dim=2).values
            assert torch.all(sorted_padding[:, :, 1:] != sorted_padding[:, :, :-1])

            ep_ranks = padding // experts_per_rank
            expected_row_fanout = min(ep_size, topk)
            for row_layer_ranks in ep_ranks.reshape((-1, topk)):
                assert torch.unique(row_layer_ranks).numel() == expected_row_fanout

            expected_active_ranks = min(ep_size, num_padding_rows * topk)
            for layer_experts, layer_ranks in zip(
                padding.transpose(0, 1),
                ep_ranks.transpose(0, 1),
                strict=True,
            ):
                assert torch.unique(layer_experts).numel() == num_padding_rows * topk
                histogram = torch.bincount(layer_ranks.flatten(), minlength=ep_size)
                assert torch.count_nonzero(histogram).item() == expected_active_ranks
                assert histogram.max().item() - histogram.min().item() <= 1


@pytest.mark.unit
def test_routing_replay_batch_padding_advances_per_row(monkeypatch):
    _set_parallel_ranks(monkeypatch)
    routed_experts, tokens = _make_routing_replay_samples(2)

    actual = prepare_routed_experts_for_routing_replay(
        routed_experts,
        tokens,
        num_experts=8,
        data_pad_size_multiplier=8,
        sequence_parallel=False,
        allgather_cp=False,
    )

    expected = torch.tensor([[4, 5], [6, 7]], dtype=torch.int32)
    torch.testing.assert_close(actual[-2:, 0], expected)


@pytest.mark.unit
@pytest.mark.parametrize(("allgather_cp", "expected_padding_rows"), [(False, (3, 3)), (True, (2, 4))])
def test_routing_replay_padding_uses_post_cp_order_before_sp(monkeypatch, allgather_cp, expected_padding_rows):
    routed_experts, tokens = _make_routing_replay_samples(3)

    for cp_rank in range(2):
        tp_outputs = []
        for tp_rank in range(2):
            _set_parallel_ranks(monkeypatch, cp_size=2, cp_rank=cp_rank, tp_size=2, tp_rank=tp_rank)
            tp_outputs.append(
                prepare_routed_experts_for_routing_replay(
                    routed_experts,
                    tokens,
                    num_experts=8,
                    data_pad_size_multiplier=1,
                    sequence_parallel=True,
                    allgather_cp=allgather_cp,
                )
            )

        padding_rows = torch.cat(tp_outputs)
        padding_rows = padding_rows[~(padding_rows == 7).all(dim=(1, 2))]
        expected = torch.arange(expected_padding_rows[cp_rank] * 2, dtype=torch.int32).reshape((-1, 2))
        torch.testing.assert_close(padding_rows[:, 0], expected)


@pytest.mark.unit
def test_default_reduces_to_per_sample_mean():
    """``sample_denoms=None`` reproduces the legacy per-sample-mean."""
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3])
    reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # per-sample means: 2, 5, 8 → sum = 15
    assert reducer(x).item() == pytest.approx(15.0)


@pytest.mark.unit
def test_per_rollout_denom_collapses_siblings_into_one_mean():
    """Pre-computed per-rollout mask sums make N sibling samples contribute one
    token-weighted mean instead of N per-sample means."""
    # 4 samples: rollout R0 owns indices 0,1,2 (mask sums 3+3+3=9); rollout R1
    # owns index 3 (mask sum 3). Pre-computed per-sample denom = group sum.
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    sample_denoms = _denoms(9, 9, 9, 3)
    reducer = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    # R0 token-mean: (1+2+...+9)/9 = 5.  R1 token-mean: (10+11+12)/3 = 11.  Sum = 16.
    assert reducer(x).item() == pytest.approx(16.0)


@pytest.mark.unit
def test_split_across_mbs_recovers_full_per_rollout_mean():
    """The critical contract: when a rollout's samples land in different mbs,
    summing each mb's reducer output equals one whole-step reducer call with
    the same pre-computed denominators. This is exactly the bug that motivated
    the precomputation — if the denom were computed per-mb (partial mask sum),
    the two halves wouldn't add up."""
    # 4 samples (same as above). Whole-step denoms = [9, 9, 9, 3].
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    sample_denoms = _denoms(9, 9, 9, 3)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    whole = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
    whole_value = whole(x).item()

    # mb_a holds samples 0, 1 of R0; mb_b holds sample 2 of R0 and sample 3 (R1).
    # Each mb carries the SAME per-sample denoms (precomputed at step level)
    # — that's what makes the split safe.
    mb_a = get_sum_of_sample_mean(total_lengths[:2], response_lengths[:2], loss_masks[:2], sample_denoms[:2])
    mb_b = get_sum_of_sample_mean(total_lengths[2:], response_lengths[2:], loss_masks[2:], sample_denoms[2:])
    split_value = mb_a(x[:6]).item() + mb_b(x[6:]).item()

    assert split_value == pytest.approx(whole_value)


@pytest.mark.unit
def test_split_with_per_mb_denom_would_be_wrong():
    """Sanity-check the bug we're guarding against: if the caller naively
    computes per-rollout denoms from each mb's own samples (the local mask
    sum, NOT the precomputed whole-rollout sum), the two halves DON'T add up
    to the whole-step value. This pins down WHY the precomputation must
    happen at the step level."""
    total_lengths, response_lengths, loss_masks = _make_inputs([3, 3, 3, 3])
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    whole = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, _denoms(9, 9, 9, 3))
    whole_value = whole(x).item()

    # Wrong denom: each mb only sees its own samples of R0.
    # mb_a's "rollout mask sum" for R0 would be 3+3=6 (instead of 9). mb_b's
    # would be 3. Different from the true whole-rollout total.
    mb_a_wrong = get_sum_of_sample_mean(total_lengths[:2], response_lengths[:2], loss_masks[:2], _denoms(6, 6))
    mb_b_wrong = get_sum_of_sample_mean(total_lengths[2:], response_lengths[2:], loss_masks[2:], _denoms(3, 3))
    wrong_total = mb_a_wrong(x[:6]).item() + mb_b_wrong(x[6:]).item()

    assert wrong_total != pytest.approx(whole_value), (
        "Expected the per-mb denom path to produce a different (incorrect) value; "
        "if these match, the regression test is no longer guarding the precomputation contract."
    )


@pytest.mark.unit
def test_cp_chunking_preserves_per_rollout_mean_report(monkeypatch):
    """Turning CP on must not change the reducer's output.

    Real flow: each CP rank only sees its chunk of the response tokens; the
    reducer's CP>1 branch slices ``loss_mask`` to match. Summing each CP
    rank's reducer output across CP ranks reproduces the cp=1 result, which
    is what train_one_step then divides by ``step_global_batch_size``.
    """
    from megatron.core import mpu as _mpu

    # Use lengths that line up cleanly with the CP chunking
    # (chunk_size = ceil(total_length / (2*cp_size))).
    total_lengths = [12, 12]  # 2 samples
    response_lengths = [8, 8]  # 4 prompt + 8 response each
    loss_masks = [torch.ones(r, dtype=torch.float32) for r in response_lengths]
    sample_denoms = torch.tensor([16.0, 16.0], dtype=torch.float32)  # = sum of both mask totals (one rollout)
    x_full = [
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ]
    x_concat = torch.cat(x_full)

    # --- cp=1 baseline ---
    monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda: 0)
    reducer_cp1 = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
    baseline = reducer_cp1(x_concat).item()

    # --- cp=2: sum partial reducer outputs across the two CP ranks ---
    monkeypatch.setattr(_mpu, "get_context_parallel_world_size", lambda: 2)
    cp_total = 0.0
    for cp_rank in range(2):
        monkeypatch.setattr(_mpu, "get_context_parallel_rank", lambda r=cp_rank: r)
        # Slice each sample's response-token tensor to the chunks this CP
        # rank owns, mirroring what the forward pass would feed in.
        x_chunks_per_sample = []
        for tl, rl, x in zip(total_lengths, response_lengths, x_full, strict=True):
            prompt_length = tl - rl
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(tl, rl)
            chunk_0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            chunk_1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            x_chunks_per_sample.append(torch.cat([chunk_0, chunk_1]))
        x_for_rank = torch.cat(x_chunks_per_sample)
        reducer_cp2 = get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks, sample_denoms)
        cp_total += reducer_cp2(x_for_rank).item()

    assert cp_total == pytest.approx(baseline)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
