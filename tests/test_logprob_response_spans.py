from argparse import Namespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from megatron.core import mpu
import slime.backends.megatron_utils.loss as megatron_loss
from slime.backends.megatron_utils.loss import _build_topp_keep_mask, get_log_probs_and_entropy

NUM_GPUS = 0


def _set_cp(monkeypatch, *, size: int, rank: int) -> None:
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)


def _kept_ids(row: torch.Tensor) -> list[int]:
    return row.nonzero(as_tuple=False).squeeze(-1).tolist()


def _logprob_args(*, chunk_size: int = -1) -> Namespace:
    return Namespace(
        allgather_cp=False,
        entropy_coef=1.0,
        log_probs_chunk_size=chunk_size,
        rollout_temperature=0.7,
    )


def _packed_logprob_inputs() -> dict:
    return {
        "unconcat_tokens": [
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            torch.tensor([6, 7, 8, 9], dtype=torch.long),
        ],
        "total_lengths": [5, 4],
        "response_lengths": [2, 3],
        "with_entropy": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, {2: [107]}),
        (1, {1: [104], 2: [105], 3: [106]}),
    ],
)
def test_top_p_mask_aligns_with_zigzag_cp_response_rows(monkeypatch, rank, expected):
    _set_cp(monkeypatch, size=2, rank=rank)
    keep = _build_topp_keep_mask(
        4,
        200,
        torch.device("cpu"),
        top_p_token_ids=[[104, 105, 106, 107]],
        top_p_token_offsets=[[0, 1, 2, 3, 4]],
        total_lengths=[8],
        response_lengths=[4],
        allgather_cp=False,
    )

    masked_rows = {row: _kept_ids(keep[row]) for row in range(keep.size(0)) if not keep[row].all()}
    assert masked_rows == expected

    selected_rows = torch.tensor([True, False, True, False])
    compact_keep = _build_topp_keep_mask(
        4,
        200,
        torch.device("cpu"),
        top_p_token_ids=[[104, 105, 106, 107]],
        top_p_token_offsets=[[0, 1, 2, 3, 4]],
        total_lengths=[8],
        response_lengths=[4],
        allgather_cp=False,
        selected_rows=selected_rows,
    )
    assert torch.equal(compact_keep, keep[selected_rows])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, {1: [102], 2: [103]}),
        (1, {0: [104], 1: [105]}),
    ],
)
def test_top_p_mask_aligns_with_allgather_cp_response_rows(monkeypatch, rank, expected):
    _set_cp(monkeypatch, size=2, rank=rank)
    keep = _build_topp_keep_mask(
        3,
        200,
        torch.device("cpu"),
        top_p_token_ids=[[102, 103, 104, 105]],
        top_p_token_offsets=[[0, 1, 2, 3, 4]],
        total_lengths=[6],
        response_lengths=[4],
        allgather_cp=True,
    )

    masked_rows = {row: _kept_ids(keep[row]) for row in range(keep.size(0)) if not keep[row].all()}
    assert masked_rows == expected

    selected_rows = torch.tensor([False, True, True])
    compact_keep = _build_topp_keep_mask(
        3,
        200,
        torch.device("cpu"),
        top_p_token_ids=[[102, 103, 104, 105]],
        top_p_token_offsets=[[0, 1, 2, 3, 4]],
        total_lengths=[6],
        response_lengths=[4],
        allgather_cp=True,
        selected_rows=selected_rows,
    )
    assert torch.equal(compact_keep, keep[selected_rows])


@pytest.mark.unit
def test_top_p_mask_aligns_with_cp1_response_rows(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    keep = _build_topp_keep_mask(
        9,
        30,
        torch.device("cpu"),
        top_p_token_ids=[[13, 99, 14], [21, 22, 99, 23]],
        top_p_token_offsets=[[0, 2, 3], [0, 1, 3, 4]],
        total_lengths=[5, 4],
        response_lengths=[2, 3],
        allgather_cp=False,
    )

    masked_rows = {row: _kept_ids(keep[row]) for row in range(keep.size(0)) if not keep[row].all()}
    assert masked_rows == {2: [13], 3: [14], 5: [21], 6: [22], 7: [23]}

    selected_rows = torch.tensor([False, False, True, False, True, True, False, True, False])
    compact_keep = _build_topp_keep_mask(
        9,
        30,
        torch.device("cpu"),
        top_p_token_ids=[[13, 99, 14], [21, 22, 99, 23]],
        top_p_token_offsets=[[0, 2, 3], [0, 1, 3, 4]],
        total_lengths=[5, 4],
        response_lengths=[2, 3],
        allgather_cp=False,
        selected_rows=selected_rows,
    )
    assert compact_keep.shape == (selected_rows.sum().item(), 30)
    assert torch.equal(compact_keep, keep[selected_rows])


@pytest.mark.unit
def test_full_logits_with_sparse_mask_preserve_existing_outputs_and_gradients(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    torch.manual_seed(2253)
    logits_data = torch.randn(1, 9, 32)
    full_loss_masks = torch.tensor([[0, 0, 1, 0, 0, 1, 0, 1, 0]])
    legacy_logits = logits_data.clone().requires_grad_()
    masked_logits = logits_data.clone().requires_grad_()
    call_kwargs = {
        "args": _logprob_args(),
        **_packed_logprob_inputs(),
    }

    _, legacy = get_log_probs_and_entropy(legacy_logits, **call_kwargs)
    _, masked = get_log_probs_and_entropy(masked_logits, full_loss_masks=full_loss_masks, **call_kwargs)

    for key in ("log_probs", "entropy"):
        legacy_values = torch.cat(legacy[key])
        masked_values = torch.cat(masked[key])
        assert torch.equal(masked_values, legacy_values)

    legacy_loss = torch.cat(legacy["log_probs"]).sum() + torch.cat(legacy["entropy"]).sum()
    masked_loss = torch.cat(masked["log_probs"]).sum() + torch.cat(masked["entropy"]).sum()
    legacy_loss.backward()
    masked_loss.backward()
    assert torch.equal(masked_logits.grad, legacy_logits.grad)


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [-1, 2])
def test_compact_logits_match_selected_full_rows_and_gradients(monkeypatch, chunk_size):
    _set_cp(monkeypatch, size=1, rank=0)
    torch.manual_seed(2253)
    logits_data = torch.randn(1, 9, 32)
    selected_rows = torch.tensor([False, False, True, False, False, True, False, True, False])
    full_loss_masks = selected_rows.unsqueeze(0)
    selected_response_rows = torch.tensor([True, False, True, False, True])
    full_logits = logits_data.clone().requires_grad_()
    compact_logits = logits_data[:, selected_rows].clone().requires_grad_()
    call_kwargs = {
        "args": _logprob_args(chunk_size=chunk_size),
        "full_loss_masks": full_loss_masks,
        **_packed_logprob_inputs(),
    }

    _, full = get_log_probs_and_entropy(full_logits, **call_kwargs)
    _, compact = get_log_probs_and_entropy(compact_logits, **call_kwargs)

    for key in ("log_probs", "entropy"):
        full_values = torch.cat(full[key])
        compact_values = torch.cat(compact[key])
        assert torch.allclose(compact_values[selected_response_rows], full_values[selected_response_rows])
        assert torch.equal(compact_values[~selected_response_rows], torch.zeros(2))

    weights = selected_response_rows.float()
    full_loss = (torch.cat(full["log_probs"]) * weights).sum() + (torch.cat(full["entropy"]) * weights).sum()
    compact_loss = torch.cat(compact["log_probs"]).sum() + torch.cat(compact["entropy"]).sum()
    full_loss.backward()
    compact_loss.backward()
    assert torch.allclose(compact_logits.grad, full_logits.grad[:, selected_rows])
    assert torch.equal(full_logits.grad[:, ~selected_rows], torch.zeros_like(full_logits.grad[:, ~selected_rows]))


@pytest.mark.unit
def test_compact_logits_reject_invalid_row_count(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    full_loss_masks = torch.tensor([[0, 0, 1, 0, 0, 1, 0, 1, 0]])

    with pytest.raises(ValueError, match="logits row count must match"):
        get_log_probs_and_entropy(
            torch.randn(1, 4, 32),
            args=_logprob_args(),
            full_loss_masks=full_loss_masks,
            **_packed_logprob_inputs(),
        )


@pytest.mark.unit
def test_empty_compact_logits_keep_response_shapes_and_autograd(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    logits = torch.empty((1, 0, 32), dtype=torch.float32, requires_grad=True)
    full_loss_masks = torch.zeros((1, 9), dtype=torch.bool)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("empty compact logits must skip vocab-parallel softmax")

    monkeypatch.setattr(megatron_loss, "calculate_log_probs_and_entropy", fail_if_called)

    _, outputs = get_log_probs_and_entropy(
        logits,
        args=_logprob_args(chunk_size=1),
        full_loss_masks=full_loss_masks,
        **_packed_logprob_inputs(),
    )

    assert [value.shape for value in outputs["log_probs"]] == [(2,), (3,)]
    assert [value.shape for value in outputs["entropy"]] == [(2,), (3,)]
    assert torch.count_nonzero(torch.cat(outputs["log_probs"])) == 0
    assert torch.count_nonzero(torch.cat(outputs["entropy"])) == 0
    loss = torch.cat(outputs["log_probs"]).sum() + torch.cat(outputs["entropy"]).sum()
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
