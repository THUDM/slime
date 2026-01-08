import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAU2_DIR = ROOT / "examples" / "tau-bench" / "tau2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TAU2_DIR))

reward = pytest.importorskip("reward")


def test_normalize_rewards_masks_removed_samples():
    rewards = [1.0, 2.0, 3.0, 4.0]
    valid_mask = [1.0, 1.0, 0.0, 1.0]
    normalized = reward._normalize_rewards(
        rewards,
        valid_mask=valid_mask,
        n_samples_per_prompt=2,
        apply_std=False,
    )
    assert pytest.approx(normalized) == [-0.5, 0.5, 0.0, 0.0]
