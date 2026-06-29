import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from megatron.core import mpu
from slime.backends.megatron_utils.loss import _build_topp_keep_mask


NUM_GPUS = 0


def _set_cp(monkeypatch, *, size: int, rank: int) -> None:
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)


def _kept_ids(row: torch.Tensor) -> list[int]:
    return row.nonzero(as_tuple=False).squeeze(-1).tolist()


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
