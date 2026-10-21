import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from megatron.core import mpu
from slime.backends.megatron_utils.loss import _build_topp_keep_mask, _fill_topp_mask_rows, get_responses

NUM_GPUS = 0


def _set_cp(monkeypatch, *, size: int, rank: int) -> None:
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)


def _kept_ids(row: torch.Tensor) -> list[int]:
    return row.nonzero(as_tuple=False).squeeze(-1).tolist()


@pytest.mark.unit
@pytest.mark.parametrize("cp_size,cp_rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("response_lengths", [[1, 0, 3], [11, 8, 6], [6, 4, 1]])
@pytest.mark.parametrize("temperature,apply_temperature", [(1.0, True), (0.7, True), (0.7, False)])
def test_response_gather_preserves_packed_cp_tokens_and_gradients(
    monkeypatch, cp_size, cp_rank, response_lengths, temperature, apply_temperature
):
    _set_cp(monkeypatch, size=cp_size, rank=cp_rank)
    total_lengths = [12, 9, 7]
    tokens = [torch.arange(length) + 100 * i for i, length in enumerate(total_lengths)]
    expected_tokens, response_rows = [], []
    local_length = 0
    # Construct the rank's global positions independently of the production
    # offset helper, including padding and samples with no response on a rank.
    for sample_tokens, total, response in zip(tokens, total_lengths, response_lengths, strict=True):
        chunk = (total + 2 * cp_size - 1) // (2 * cp_size)
        positions = list(range(cp_rank * chunk, (cp_rank + 1) * chunk))
        positions += list(range((2 * cp_size - cp_rank - 1) * chunk, (2 * cp_size - cp_rank) * chunk))
        rows = [i for i, position in enumerate(positions) if total - response <= position + 1 < total]
        response_rows.append(torch.tensor([local_length + i for i in rows], dtype=torch.long))
        expected_tokens.append(sample_tokens[[positions[i] + 1 for i in rows]])
        local_length += len(positions)

    logits = torch.randn(1, local_length, 5, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    actual = list(
        get_responses(
            logits,
            args=Namespace(rollout_temperature=temperature, allgather_cp=False),
            unconcat_tokens=tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            apply_temperature=apply_temperature,
        )
    )
    divisor = temperature if apply_temperature else 1.0
    expected_logits = [reference[0, rows] / divisor for rows in response_rows]
    for (actual_logits, actual_tokens), expected, sample_tokens in zip(
        actual, expected_logits, expected_tokens, strict=True
    ):
        torch.testing.assert_close(actual_logits, expected)
        torch.testing.assert_close(actual_tokens, sample_tokens)
    sum((i + 1) * rows.square().sum() for i, (rows, _) in enumerate(actual)).backward()
    sum((i + 1) * rows.square().sum() for i, rows in enumerate(expected_logits)).backward()
    torch.testing.assert_close(logits.grad, reference.grad)


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


@pytest.mark.unit
@pytest.mark.parametrize("tensor_ids", [False, True])
@pytest.mark.parametrize("vocab_start", [0, 4, 8, 16])
@pytest.mark.parametrize("response_start,length", [(0, 5), (1, 3), (3, 8), (5, 2), (1, 0)])
def test_sparse_top_p_rows_preserve_empty_support_duplicates_and_unrecorded_rows(
    tensor_ids, vocab_start, response_start, length
):
    support = [[0, 4, 4, 7], [], [2, 6, 9], [8], []]
    offsets = [0]
    ids = []
    for row in support:
        ids.extend(row)
        offsets.append(len(ids))
    expected = torch.ones(9, 4, dtype=torch.bool)
    for i in range(response_start, min(response_start + length, len(support))):
        expected[2 + i - response_start] = torch.tensor(
            [token in support[i] for token in range(vocab_start, vocab_start + 4)]
        )
    keep = torch.ones_like(expected)
    _fill_topp_mask_rows(
        keep,
        torch.tensor(ids, dtype=torch.int32) if tensor_ids else ids,
        offsets,
        response_start,
        2,
        length,
        vocab_start,
        vocab_start + 4,
    )
    torch.testing.assert_close(keep, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
