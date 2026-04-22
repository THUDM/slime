"""Tests for slime.utils.checkpoint_utils."""

import os
import shutil

import pytest

from slime.utils.checkpoint_utils import cleanup_old_checkpoints, should_run_cleanup


def _megatron_path_fn(save_dir):
    """Return a path_fn that maps rollout_id → Megatron iter dir."""
    return lambda rid: os.path.join(save_dir, f"iter_{rid:07d}")


def _hf_path_fn(template):
    """Return a path_fn that maps rollout_id → HF checkpoint dir."""
    return lambda rid: template.format(rollout_id=rid)


def _make_megatron_dirs(tmp_path, iters):
    """Create Megatron-style iter_NNNNNNN directories."""
    save_dir = str(tmp_path / "ckpt")
    os.makedirs(save_dir, exist_ok=True)
    for i in iters:
        os.makedirs(os.path.join(save_dir, f"iter_{i:07d}"))
    return save_dir


def _make_hf_dirs(tmp_path, template, rollout_ids):
    """Create HF checkpoint directories from template."""
    for rid in rollout_ids:
        os.makedirs(template.format(rollout_id=rid))


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


class TestCleanupOldCheckpoints:
    def test_deletes_oldest_keeps_newest(self, tmp_path):
        save_dir = _make_megatron_dirs(tmp_path, [10, 20, 30, 40, 50])
        deleted = cleanup_old_checkpoints(
            [10, 20, 30, 40, 50], keep=2, path_fn=_megatron_path_fn(save_dir),
        )
        remaining = sorted(os.listdir(save_dir))
        assert remaining == ["iter_0000040", "iter_0000050"]
        assert len(deleted) == 3

    def test_noop_under_limit(self, tmp_path):
        save_dir = _make_megatron_dirs(tmp_path, [10, 20])
        deleted = cleanup_old_checkpoints(
            [10, 20], keep=3, path_fn=_megatron_path_fn(save_dir),
        )
        assert deleted == []
        assert len(os.listdir(save_dir)) == 2

    def test_noop_at_exact_limit(self, tmp_path):
        save_dir = _make_megatron_dirs(tmp_path, [10, 20, 30])
        deleted = cleanup_old_checkpoints(
            [10, 20, 30], keep=3, path_fn=_megatron_path_fn(save_dir),
        )
        assert deleted == []

    def test_keep_one(self, tmp_path):
        save_dir = _make_megatron_dirs(tmp_path, [5, 10, 15])
        deleted = cleanup_old_checkpoints(
            [5, 10, 15], keep=1, path_fn=_megatron_path_fn(save_dir),
        )
        remaining = os.listdir(save_dir)
        assert remaining == ["iter_0000015"]
        assert len(deleted) == 2

    def test_keep_zero_deletes_all(self, tmp_path):
        save_dir = _make_megatron_dirs(tmp_path, [10, 20, 30])
        cleanup_old_checkpoints(
            [10, 20, 30], keep=0, path_fn=_megatron_path_fn(save_dir),
        )
        remaining = [e for e in os.listdir(save_dir) if e.startswith("iter_")]
        assert remaining == []

    def test_empty_list(self, tmp_path):
        save_dir = _make_megatron_dirs(tmp_path, [])
        deleted = cleanup_old_checkpoints(
            [], keep=2, path_fn=_megatron_path_fn(save_dir),
        )
        assert deleted == []

    def test_missing_dir_skipped(self, tmp_path):
        """path_fn returns a path that doesn't exist on disk — silently skipped."""
        save_dir = str(tmp_path / "ckpt")
        os.makedirs(save_dir)
        # Only create dir for rollout_id 10
        os.makedirs(os.path.join(save_dir, "iter_0000010"))
        deleted = cleanup_old_checkpoints(
            [0, 5, 10], keep=1, path_fn=_megatron_path_fn(save_dir),
        )
        # 0 and 5 don't exist on disk, so they're skipped
        assert deleted == []
        assert os.path.isdir(os.path.join(save_dir, "iter_0000010"))

    def test_rmtree_failure_does_not_crash(self, tmp_path, monkeypatch):
        """If shutil.rmtree raises OSError, cleanup logs a warning and continues."""
        save_dir = _make_megatron_dirs(tmp_path, [10, 20, 30])

        original_rmtree = shutil.rmtree
        call_count = 0

        def flaky_rmtree(path, *a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("simulated transient error")
            return original_rmtree(path, *a, **kw)

        monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)
        deleted = cleanup_old_checkpoints(
            [10, 20, 30], keep=1, path_fn=_megatron_path_fn(save_dir),
        )
        assert len(deleted) == 1
        remaining = sorted(e for e in os.listdir(save_dir) if e.startswith("iter_"))
        assert remaining == ["iter_0000010", "iter_0000030"]


# ---------------------------------------------------------------------------
# Megatron-specific behavior
# ---------------------------------------------------------------------------


class TestMegatronCheckpoints:
    def test_preserves_latest_checkpointed_iteration_txt(self, tmp_path):
        save_dir = _make_megatron_dirs(tmp_path, [10, 20, 30])
        txt_path = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
        with open(txt_path, "w") as f:
            f.write("30\n")
        cleanup_old_checkpoints(
            [10, 20, 30], keep=1, path_fn=_megatron_path_fn(save_dir),
        )
        assert os.path.isfile(txt_path)
        remaining = [e for e in os.listdir(save_dir) if e.startswith("iter_")]
        assert remaining == ["iter_0000030"]

    def test_ignores_non_iter_entries(self, tmp_path):
        save_dir = _make_megatron_dirs(tmp_path, [10, 20])
        os.makedirs(os.path.join(save_dir, "some_other_dir"))
        with open(os.path.join(save_dir, "some_file.txt"), "w") as f:
            f.write("hello")
        cleanup_old_checkpoints(
            [10, 20], keep=1, path_fn=_megatron_path_fn(save_dir),
        )
        # Non-matching entries untouched
        assert os.path.isdir(os.path.join(save_dir, "some_other_dir"))
        assert os.path.isfile(os.path.join(save_dir, "some_file.txt"))

    def test_peak_equals_limit(self, tmp_path):
        """Simulate cleanup(keep=limit-1) + save cycle — peak never exceeds limit."""
        save_dir = str(tmp_path / "ckpt")
        os.makedirs(save_dir)
        limit = 2
        saved_rollout_ids = []

        for rollout_id in range(5):
            cleanup_old_checkpoints(
                saved_rollout_ids, keep=limit - 1, path_fn=_megatron_path_fn(save_dir),
            )
            os.makedirs(os.path.join(save_dir, f"iter_{rollout_id:07d}"))
            saved_rollout_ids.append(rollout_id)
            n = len([e for e in os.listdir(save_dir) if e.startswith("iter_")])
            assert n <= limit, f"peak exceeded limit: {n} checkpoints on disk"

    def test_only_cleans_own_run(self, tmp_path):
        """Checkpoints from a previous run are not touched."""
        save_dir = _make_megatron_dirs(tmp_path, [1, 2, 3])  # previous run
        saved_rollout_ids = []  # current run starts fresh

        # Current run saves rollout 4, 5, 6 with keep=2
        for rollout_id in [4, 5, 6]:
            cleanup_old_checkpoints(
                saved_rollout_ids, keep=2 - 1, path_fn=_megatron_path_fn(save_dir),
            )
            os.makedirs(os.path.join(save_dir, f"iter_{rollout_id:07d}"))
            saved_rollout_ids.append(rollout_id)

        # Previous run's checkpoints still intact
        assert os.path.isdir(os.path.join(save_dir, "iter_0000001"))
        assert os.path.isdir(os.path.join(save_dir, "iter_0000002"))
        assert os.path.isdir(os.path.join(save_dir, "iter_0000003"))
        # Current run: keep=2, so 5 and 6 remain, 4 deleted
        assert not os.path.isdir(os.path.join(save_dir, "iter_0000004"))
        assert os.path.isdir(os.path.join(save_dir, "iter_0000005"))
        assert os.path.isdir(os.path.join(save_dir, "iter_0000006"))


# ---------------------------------------------------------------------------
# HF-specific behavior
# ---------------------------------------------------------------------------


class TestHfCheckpoints:
    def test_deletes_oldest_by_rollout_id(self, tmp_path):
        template = str(tmp_path / "hf_ckpt_{rollout_id}")
        rollout_ids = [0, 5, 10, 15, 20]
        _make_hf_dirs(tmp_path, template, rollout_ids)
        deleted = cleanup_old_checkpoints(
            rollout_ids, keep=2, path_fn=_hf_path_fn(template),
        )
        assert len(deleted) == 3
        assert os.path.isdir(template.format(rollout_id=15))
        assert os.path.isdir(template.format(rollout_id=20))
        assert not os.path.isdir(template.format(rollout_id=0))

    def test_noop_under_limit(self, tmp_path):
        template = str(tmp_path / "hf_ckpt_{rollout_id}")
        rollout_ids = [0, 5]
        _make_hf_dirs(tmp_path, template, rollout_ids)
        deleted = cleanup_old_checkpoints(
            rollout_ids, keep=3, path_fn=_hf_path_fn(template),
        )
        assert deleted == []

    def test_noop_at_exact_limit(self, tmp_path):
        template = str(tmp_path / "hf_{rollout_id}")
        rollout_ids = [0, 5, 10]
        _make_hf_dirs(tmp_path, template, rollout_ids)
        deleted = cleanup_old_checkpoints(
            rollout_ids, keep=3, path_fn=_hf_path_fn(template),
        )
        assert deleted == []

    def test_none_template_skips(self, tmp_path):
        """When save_hf is None, no dirs exist — all skipped."""
        deleted = cleanup_old_checkpoints(
            [0, 1, 2], keep=1, path_fn=lambda rid: str(tmp_path / f"nonexistent_{rid}"),
        )
        assert deleted == []

    def test_keep_one(self, tmp_path):
        template = str(tmp_path / "hf_{rollout_id}")
        rollout_ids = [0, 5, 10]
        _make_hf_dirs(tmp_path, template, rollout_ids)
        deleted = cleanup_old_checkpoints(
            rollout_ids, keep=1, path_fn=_hf_path_fn(template),
        )
        assert len(deleted) == 2
        assert not os.path.isdir(template.format(rollout_id=0))
        assert not os.path.isdir(template.format(rollout_id=5))
        assert os.path.isdir(template.format(rollout_id=10))

    def test_keep_zero_deletes_all(self, tmp_path):
        template = str(tmp_path / "hf_{rollout_id}")
        rollout_ids = [0, 5, 10]
        _make_hf_dirs(tmp_path, template, rollout_ids)
        cleanup_old_checkpoints(
            rollout_ids, keep=0, path_fn=_hf_path_fn(template),
        )
        for rid in rollout_ids:
            assert not os.path.isdir(template.format(rollout_id=rid))

    def test_peak_equals_limit(self, tmp_path):
        """Simulate cleanup(keep=limit-1) + save + track cycle — peak never exceeds limit."""
        template = str(tmp_path / "hf_{rollout_id}")
        limit = 2
        saved_rollout_ids = []

        for rollout_id in range(5):
            cleanup_old_checkpoints(
                saved_rollout_ids, keep=limit - 1, path_fn=_hf_path_fn(template),
            )
            os.makedirs(template.format(rollout_id=rollout_id))
            saved_rollout_ids.append(rollout_id)
            n = sum(1 for rid in saved_rollout_ids
                    if os.path.isdir(template.format(rollout_id=rid)))
            assert n <= limit, f"peak exceeded limit: {n} HF checkpoints on disk"

    def test_rmtree_failure_does_not_crash(self, tmp_path, monkeypatch):
        template = str(tmp_path / "hf_{rollout_id}")
        rollout_ids = [0, 5, 10]
        _make_hf_dirs(tmp_path, template, rollout_ids)

        original_rmtree = shutil.rmtree
        call_count = 0

        def flaky_rmtree(path, *a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("simulated transient error")
            return original_rmtree(path, *a, **kw)

        monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)
        deleted = cleanup_old_checkpoints(
            rollout_ids, keep=1, path_fn=_hf_path_fn(template),
        )
        assert len(deleted) == 1
        assert os.path.isdir(template.format(rollout_id=0))
        assert not os.path.isdir(template.format(rollout_id=5))
        assert os.path.isdir(template.format(rollout_id=10))


# ---------------------------------------------------------------------------
# should_run_cleanup — rank selection logic
# ---------------------------------------------------------------------------


class TestShouldRunCleanup:
    def test_shared_global_rank_0(self):
        megatron, hf = should_run_cleanup("shared", global_rank=0, local_rank=0)
        assert megatron is True
        assert hf is True

    def test_shared_global_rank_nonzero(self):
        megatron, hf = should_run_cleanup("shared", global_rank=3, local_rank=3)
        assert megatron is False
        assert hf is False

    def test_shared_global_rank_nonzero_local_rank_zero(self):
        """On node 1, local_rank=0 but global_rank=8 — shared uses global rank."""
        megatron, hf = should_run_cleanup("shared", global_rank=8, local_rank=0)
        assert megatron is False
        assert hf is False

    def test_local_local_rank_0_global_rank_0(self):
        """Node 0, local_rank=0 — both should cleanup."""
        megatron, hf = should_run_cleanup("local", global_rank=0, local_rank=0)
        assert megatron is True
        assert hf is True

    def test_local_local_rank_0_global_rank_nonzero(self):
        """Node 1, local_rank=0 but global_rank=8 — megatron yes, HF no."""
        megatron, hf = should_run_cleanup("local", global_rank=8, local_rank=0)
        assert megatron is True
        assert hf is False

    def test_local_local_rank_nonzero(self):
        """local_rank=3 — neither should cleanup."""
        megatron, hf = should_run_cleanup("local", global_rank=3, local_rank=3)
        assert megatron is False
        assert hf is False
