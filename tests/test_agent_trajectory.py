import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent.trajectory import TurnRecord, merge_turns


NUM_GPUS = 0


def _turn(prompt_ids: list[int], output_ids: list[int]) -> TurnRecord:
    return TurnRecord(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        output_loss_mask=[1] * len(output_ids),
        finish_reason="stop",
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
