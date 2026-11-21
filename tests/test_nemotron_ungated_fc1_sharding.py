"""CPU tests for how Nemotron's ungated ``linear_fc1`` is split across TP ranks and put back.

Nemotron's experts use a squared ReLU with no gate, so ``linear_fc1`` is ``up_proj`` alone.
slime's shared sharder treats every ``linear_fc1`` as ``cat(gate, up)`` unless the parameter
carries ``partition_sections``; ``mark_ungated_linear_fc1_sections`` sets one section over
the whole width, which makes both directions a plain contiguous split.

A wrong split has the right shape, so nothing raises. The shared expert is the only
``linear_fc1`` that runs at TP > 1, and the bug it caused showed up only as a train /
rollout log-prob gap of ~0.5. These tests pin the layout directly. Row ``i`` of every
tensor holds the value ``i``, so a test can read which rows a rank got.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# The loader half only needs torch; mount the source package without Megatron's runtime
# patch initialization when Megatron is not installed (as the conversion tests do).
try:
    _has_megatron = importlib.util.find_spec("megatron.core") is not None
except ModuleNotFoundError:
    _has_megatron = False
if not _has_megatron:
    _megatron_utils = types.ModuleType("slime.backends.megatron_utils")
    _megatron_utils.__path__ = [str(Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils")]
    sys.modules.setdefault("slime.backends.megatron_utils", _megatron_utils)

from slime.backends.megatron_utils.hf_to_megatron.common import _tensor_parallel_shard  # noqa: E402

NUM_GPUS = 0

NAME = "decoder.layers.1.mlp.shared_experts.linear_fc1.weight"
BIAS = "decoder.layers.1.mlp.shared_experts.linear_fc1.bias"


def _rows(count: int, width: int = 3) -> torch.Tensor:
    return torch.arange(count, dtype=torch.float32).unsqueeze(1).expand(count, width).contiguous()


def _row_ids(tensor: torch.Tensor) -> list[int]:
    return tensor[:, 0].long().tolist()


def _shard(tensor, rank, tp, *, sections=None, stride=1, name=NAME):
    return _tensor_parallel_shard(
        name,
        tensor,
        parallel_size=tp,
        parallel_rank=rank,
        partition_dim=0,
        partition_stride=stride,
        partition_sections=sections,
    )


@pytest.mark.unit
def test_without_sections_the_loader_splits_an_fc1_as_if_it_were_gated():
    # This is the old behaviour: half the rows are "gate", half "up", and each rank gets a
    # slice of each, so rank 0 holds rows 0,1 and 4,5 rather than 0..3.
    tensor = _rows(8)
    assert _row_ids(_shard(tensor, 0, 2)) == [0, 1, 4, 5]
    assert _row_ids(_shard(tensor, 1, 2)) == [2, 3, 6, 7]


@pytest.mark.unit
def test_one_section_gives_each_rank_a_contiguous_block():
    tensor = _rows(8)
    assert _row_ids(_shard(tensor, 0, 2, sections=(8,))) == [0, 1, 2, 3]
    assert _row_ids(_shard(tensor, 1, 2, sections=(8,))) == [4, 5, 6, 7]


@pytest.mark.unit
@pytest.mark.parametrize("tp", [2, 4])
def test_the_blocks_of_all_ranks_cover_the_tensor_once_in_order(tp):
    tensor = _rows(16)
    blocks = [_shard(tensor, rank, tp, sections=(16,)) for rank in range(tp)]
    assert all(block.shape[0] == 16 // tp for block in blocks)
    assert torch.equal(torch.cat(blocks), tensor)


@pytest.mark.unit
def test_the_bias_is_split_the_same_way():
    bias = torch.arange(8, dtype=torch.float32)
    assert _shard(bias, 0, 2, sections=(8,), name=BIAS).tolist() == [0, 1, 2, 3]
    assert _shard(bias, 1, 2, sections=(8,), name=BIAS).tolist() == [4, 5, 6, 7]


@pytest.mark.unit
def test_a_section_that_does_not_divide_by_tp_is_refused():
    with pytest.raises(ValueError, match="divisible by TP"):
        _shard(_rows(9), 0, 2, sections=(9,))


@pytest.mark.unit
def test_sections_need_a_stride_of_one():
    with pytest.raises(ValueError, match="partition_stride=1"):
        _shard(_rows(8), 0, 2, sections=(8,), stride=2)


@pytest.mark.unit
def test_tp1_returns_the_tensor_unchanged():
    tensor = _rows(8)
    assert _shard(tensor, 0, 1, sections=(8,)) is tensor


# -------------------------------------------------- the export side (needs Megatron)


@pytest.fixture
def merge():
    pytest.importorskip("megatron.core.transformer.transformer_layer")
    from slime.backends.megatron_utils.update_weight.common import _merge_param_partitions

    return _merge_param_partitions


@pytest.mark.unit
@pytest.mark.parametrize("tp", [2, 4])
def test_load_then_export_returns_the_original_with_sections(merge, tp):
    tensor = _rows(16)
    shards = [_shard(tensor, rank, tp, sections=(16,)) for rank in range(tp)]
    assert torch.equal(merge(NAME, shards, 0, (16,)), tensor)


@pytest.mark.unit
def test_the_gated_rule_on_both_sides_also_round_trips_which_is_why_it_hid(merge):
    # The old code was symmetric, so the round trip was intact and every weight-sync check
    # passed. What differs is the layout a rank holds: gate/up interleaved, which an ungated
    # ColumnParallelLinear does not read that way.
    tensor = _rows(8)
    shards = [_shard(tensor, rank, 2) for rank in range(2)]
    assert torch.equal(merge(NAME, shards, 0), tensor)
    assert _row_ids(shards[0]) != [0, 1, 2, 3]


@pytest.mark.unit
def test_mixing_the_two_rules_does_not_round_trip(merge):
    # Loading with sections but exporting by the name-based gated rule (or the reverse) puts
    # the rows in the wrong places: right shape, wrong content. Both sides must be tagged.
    tensor = _rows(8)
    shards = [_shard(tensor, rank, 2, sections=(8,)) for rank in range(2)]
    assert not torch.equal(merge(NAME, shards, 0), tensor)


@pytest.mark.unit
def test_the_merged_tensor_remembers_its_sections(merge):
    shards = [_shard(_rows(8), rank, 2, sections=(8,)) for rank in range(2)]
    assert merge(NAME, shards, 0, (8,)).partition_sections == (8,)
