"""CPU unit tests for ``check_apply_results`` (issue #2104).

``_finalize_sync`` previously called ``ray.get(object_refs)`` and discarded
the ``(success, msg)`` tuples returned by each SGLang receiver engine.  A
failed delta-weight apply was therefore silent, leaving the sender snapshot
permanently ahead of the receiver.

``check_apply_results`` is the extracted helper that closes this gap.
These tests exercise it directly — no GPU, Ray, or distributed runtime needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest

# Load _apply_result_check.py directly so the test doesn't trigger the
# package-level __init__.py files that import torch / ray / megatron.
_spec = importlib.util.spec_from_file_location(
    "_apply_result_check",
    Path(__file__).parent.parent
    / "slime/backends/megatron_utils/update_weight/_apply_result_check.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_check_apply_results = _mod.check_apply_results


class TestCheckApplyResults:
    """Unit tests for the apply-result inspector added to fix issue #2104."""

    def test_all_successful_results_do_not_raise(self):
        """When every engine reports success, no exception is raised."""
        _check_apply_results([(True, ""), (True, "ok"), (True, "weight version 42")])

    def test_empty_results_do_not_raise(self):
        """An empty result list (no engines) is also valid."""
        _check_apply_results([])

    def test_single_failed_engine_raises_runtime_error(self):
        """A single (False, msg) result raises RuntimeError."""
        with pytest.raises(RuntimeError, match="1/2 engine"):
            _check_apply_results([(True, ""), (False, "checksum mismatch")])

    def test_error_message_contains_failed_engine_index(self, caplog):
        """The per-engine log line includes the engine index."""
        import logging

        with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
            _check_apply_results([(True, ""), (False, "io error on shard 3")])

        assert "Engine[1]" in caplog.text
        assert "io error on shard 3" in caplog.text

    def test_all_failed_engines_raise_with_correct_count(self):
        """The RuntimeError message reflects the total failure count."""
        with pytest.raises(RuntimeError, match="3/3"):
            _check_apply_results(
                [(False, "decode error"), (False, "nccl timeout"), (False, "oom")]
            )

    def test_non_tuple_results_treated_as_success(self):
        """
        If an engine returns something other than a (bool, str) tuple (e.g. None
        or a bare bool True from an older SGLang version), it is not treated as
        a failure — the helper only flags explicit ``(False, msg)`` pairs.
        """
        _check_apply_results([True, None, (True, "ok")])
