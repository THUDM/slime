import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent.trajectory import TurnRecord, TurnSegment, merge_turn_segments, merge_turns


NUM_GPUS = 0


def _turn(prompt_ids: list[int], output_ids: list[int], output_log_probs: list[float] | None = None) -> TurnRecord:
    return TurnRecord(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        finish_reason="stop",
        output_log_probs=(
            output_log_probs if output_log_probs is not None else [-token_id / 100 for token_id in output_ids]
        ),
    )


@pytest.mark.unit
def test_merge_turns_preserves_matched_prefix_on_prompt_drift():
    segment = merge_turns(
        [
            _turn([10], [11]),
            _turn([10, 11, 21], [12]),
            _turn([10, 11, 21, 12, 31], [13]),
            _turn([10, 11, 21, 12, 22], [14]),
        ]
    )

    assert segment is not None
    assert segment.prompt_ids == [10]
    assert segment.response_ids == [11, 21, 12, 22, 14]
    assert segment.loss_mask == [1, 0, 1, 0, 1]
    assert segment.rollout_log_probs == [-0.11, 0.0, -0.12, 0.0, -0.14]


@pytest.mark.unit
def test_merge_turns_drops_middle_turn_when_next_prompt_skips_it():
    segment = merge_turns(
        [
            _turn([10], [11]),
            _turn([10, 11, 21], [12]),
            _turn([10, 11, 22], [13]),
            _turn([10, 11, 22, 13, 31], [14]),
        ]
    )

    assert segment is not None
    assert segment.prompt_ids == [10]
    assert segment.response_ids == [11, 22, 13, 31, 14]
    assert segment.loss_mask == [1, 0, 1, 0, 1]
    assert segment.rollout_log_probs == [-0.11, 0.0, -0.13, 0.0, -0.14]


@pytest.mark.unit
def test_merge_turns_handles_consecutive_prompt_drifts():
    segment = merge_turns(
        [
            _turn([10], [11]),
            _turn([10, 11, 21], [12]),
            _turn([10, 11, 22], [13]),
            _turn([10, 11, 23], [14]),
            _turn([10, 11, 23, 14, 31], [15]),
        ]
    )

    assert segment is not None
    assert segment.prompt_ids == [10]
    assert segment.response_ids == [11, 23, 14, 31, 15]
    assert segment.loss_mask == [1, 0, 1, 0, 1]
    assert segment.rollout_log_probs == [-0.11, 0.0, -0.14, 0.0, -0.15]


@pytest.mark.unit
def test_merge_turns_masks_whole_output_when_prompt_drift_splits_it():
    segment = merge_turns(
        [
            _turn([10], [11, 12, 13, 14]),
            _turn([10, 11, 12, 99, 14], [15]),
        ]
    )

    assert segment is not None
    assert segment.prompt_ids == [10]
    assert segment.response_ids == [11, 12, 99, 14, 15]
    assert segment.loss_mask == [0, 0, 0, 0, 1]
    assert segment.rollout_log_probs == [0.0, 0.0, 0.0, 0.0, -0.15]


@pytest.mark.unit
def test_merge_turns_masks_whole_output_when_prompt_drift_changes_token_count():
    segment = merge_turns(
        [
            _turn([10], [11, 12, 13, 14]),
            _turn([10, 11, 12, 99, 100, 14], [15]),
        ]
    )

    assert segment is not None
    assert segment.prompt_ids == [10]
    assert segment.response_ids == [11, 12, 99, 100, 14, 15]
    assert segment.loss_mask == [0, 0, 0, 0, 0, 1]
    assert segment.rollout_log_probs == [0.0, 0.0, 0.0, 0.0, 0.0, -0.15]


@pytest.mark.unit
def test_merge_turns_restarts_when_prompt_base_changes():
    segment = merge_turns(
        [
            _turn([10], [11]),
            _turn([20, 21], [22]),
            _turn([20, 21, 22, 23], [24]),
        ]
    )

    assert segment is not None
    assert segment.prompt_ids == [20, 21]
    assert segment.response_ids == [22, 23, 24]
    assert segment.loss_mask == [1, 0, 1]
    assert segment.rollout_log_probs == [-0.22, 0.0, -0.24]


@pytest.mark.unit
def test_merge_turn_segments_keeps_oversized_segments():
    segments = [TurnSegment(turns=[_turn([10, 11, 12], [13, 14])])]

    merged = merge_turn_segments(segments)

    assert len(merged) == 1
    assert merged[0].prompt_ids == [10, 11, 12]
    assert merged[0].response_ids == [13, 14]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
