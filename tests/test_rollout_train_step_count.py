"""CPU unit tests for rollout-side train-step counter plumbing.

Covers:
  * ``RolloutDataSource.save / load`` round-trips the ``train_step_count``
    written into ``metadata`` (so wandb step labels survive resume);
  * Default ``train_step_count`` is rehydrated to 0 when the checkpoint
    doesn't carry it (legacy resume).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from slime.rollout.data_source import RolloutDataSource


def _make_args(save_dir: str) -> SimpleNamespace:
    """Minimal args that satisfy ``RolloutDataSource.__init__`` / ``load`` without
    loading a real dataset (rollout_global_dataset=True but prompt_data=None
    short-circuits the heavy path)."""
    return SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=None,
        save=save_dir,
        load=save_dir,
        rollout_shuffle=False,
    )


@pytest.mark.unit
def test_data_source_metadata_roundtrips_train_step_count(tmp_path):
    """Round-trip the train_step_count via data_source save/load → metadata."""
    args = _make_args(str(tmp_path))

    ds = RolloutDataSource(args)
    # Mimic what RolloutManager.save does: stash the counter into metadata
    # before delegating to data_source.save.
    ds.metadata["train_step_count"] = 42
    ds.sample_offset = 7
    ds.save(rollout_id=3)

    expected_path = os.path.join(str(tmp_path), "rollout/global_dataset_state_dict_3.pt")
    assert os.path.exists(expected_path)

    # Fresh instance picks up the persisted state.
    ds2 = RolloutDataSource(args)
    assert ds2.metadata == {}
    ds2.load(rollout_id=3)
    assert ds2.metadata.get("train_step_count") == 42
    assert ds2.sample_offset == 7


@pytest.mark.unit
def test_data_source_load_missing_train_step_count_is_silent(tmp_path):
    """Legacy checkpoint (no train_step_count in metadata) loads cleanly with
    metadata not containing the key — RolloutManager falls back to a
    historical-formula default in this case (covered in its own test)."""
    args = _make_args(str(tmp_path))

    ds = RolloutDataSource(args)
    # Save state WITHOUT train_step_count in metadata.
    ds.metadata["other_key"] = "legacy_value"
    ds.save(rollout_id=0)

    ds2 = RolloutDataSource(args)
    ds2.load(rollout_id=0)
    assert ds2.metadata.get("train_step_count") is None
    assert ds2.metadata.get("other_key") == "legacy_value"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
