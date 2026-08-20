from pathlib import Path

import pytest

from slime.utils.checkpoint_retention import prune_megatron_checkpoints


def _checkpoint(root: Path, iteration: int) -> Path:
    path = root / f"iter_{iteration:07d}"
    path.mkdir()
    (path / "state.distcp").write_text("checkpoint")
    return path


@pytest.mark.unit
def test_prune_megatron_checkpoints_keeps_latest_completed_history(tmp_path):
    checkpoints = [_checkpoint(tmp_path, iteration) for iteration in (10, 20, 30)]
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("30\n")

    removed = prune_megatron_checkpoints(tmp_path, retain_count=2)

    assert removed == [checkpoints[0]]
    assert not checkpoints[0].exists()
    assert checkpoints[1].is_dir()
    assert checkpoints[2].is_dir()


@pytest.mark.unit
def test_prune_megatron_checkpoints_never_removes_inflight_directory(tmp_path):
    completed = _checkpoint(tmp_path, 20)
    inflight = _checkpoint(tmp_path, 30)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20\n")

    assert prune_megatron_checkpoints(tmp_path, retain_count=1) == []
    assert completed.is_dir()
    assert inflight.is_dir()


@pytest.mark.unit
def test_prune_megatron_checkpoints_ignores_release_marker(tmp_path):
    checkpoint = _checkpoint(tmp_path, 10)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release\n")

    assert prune_megatron_checkpoints(tmp_path, retain_count=1) == []
    assert checkpoint.is_dir()


@pytest.mark.unit
def test_prune_megatron_checkpoints_rejects_zero_retention(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        prune_megatron_checkpoints(tmp_path, retain_count=0)
