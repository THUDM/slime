"""Shared infrastructure for the CP-related multi-process CPU tests.

Why this module exists
----------------------
The CP / metric-report / backward-grad-norm tests all want to:

1. Stub ``megatron.core.mpu`` *before* importing
   ``slime.backends.megatron_utils.cp_utils`` (the CPU CI image has no real
   megatron).
2. Spawn ``dp_size * cp_size`` workers with real ``torch.distributed`` and
   exercise the actual production helpers (``get_sum_of_sample_mean``,
   ``reduce_train_step_metrics``, ``gather_and_reduce_log_dict``,
   ``rollout_log_metric_contribution``).
3. Chunk each sample's response tensor across CP ranks the same way the
   real forward pass does — using
   ``get_logits_and_tokens_offset_with_cp`` so the slicing stays in lock-
   step with the production reducer.

Putting that here keeps the per-feature test files focused on the
behaviour they check (numerics / report formulas / backward) rather than
on plumbing.

Mapping to Megatron
-------------------
- ``mp.spawn(...)`` + gloo backend mirrors the per-rank entry-point that
  ``torch.distributed.run`` would create for a real launch.
- ``dp_cp_group = new_group(range(world_size))`` matches
  ``parallel_state.get_data_parallel_group(with_context_parallel=True)``
  (Megatron-LM ``finalize_model_grads.py:437``).  In the no-TP / no-PP
  CPU test setup the whole world *is* that group.
- The per-rank CP chunking mirrors what the attention layer feeds into
  the loss in Megatron: each CP rank only sees its 2-chunk slice of the
  response tokens (cf. ``cp_utils.get_logits_and_tokens_offset_with_cp``,
  the same helper used by the real forward pass).
"""

from __future__ import annotations

import os
import socket
import sys
import types


# --- Stub ``megatron.core.mpu`` (must run before cp_utils is imported) ---
#
# Both this module and any test file that imports it should *import this
# helper first*. Doing so installs the stub at import time so that the
# subsequent ``from slime.backends.megatron_utils.cp_utils import ...`` in
# the test file binds ``cp_utils.mpu`` to this stub.
#
# In spawned workers, ``mp.spawn`` re-imports the test module fresh, which
# re-runs this stub installation; then the worker mutates the stub's
# ``get_context_parallel_*`` attributes via ``_stub_megatron_in_worker``
# below to pin (cp_size, cp_rank) for that worker.
_fake_mpu = types.ModuleType("megatron.core.mpu")
_fake_mpu.get_context_parallel_world_size = lambda: 1
_fake_mpu.get_context_parallel_rank = lambda: 0
_fake_core = types.ModuleType("megatron.core")
_fake_core.mpu = _fake_mpu
_fake_megatron = types.ModuleType("megatron")
_fake_megatron.core = _fake_core
sys.modules.setdefault("megatron", _fake_megatron)
sys.modules.setdefault("megatron.core", _fake_core)
sys.modules.setdefault("megatron.core.mpu", _fake_mpu)


def stub_megatron_in_worker(cp_size: int, cp_rank: int) -> None:
    """Override ``mpu.get_context_parallel_*`` inside an ``mp.spawn`` worker.

    ``mp.spawn`` pickles the worker function by name and re-imports the
    test module in the child — that re-runs the top-of-file stub install
    with ``cp_size=1``. By the time the worker runs, ``cp_utils`` has
    already bound its module-level ``mpu`` reference to the stub.

    So we must MUTATE the stub module's attributes in place rather than
    replace ``sys.modules['megatron.core.mpu']`` — replacing the module
    would leave ``cp_utils.mpu`` pointing at the now-shadowed stub.
    """
    from megatron.core import mpu  # the stub installed at import time

    mpu.get_context_parallel_world_size = lambda: cp_size
    mpu.get_context_parallel_rank = lambda: cp_rank


def free_port() -> int:
    """Pick an unused TCP port for ``init_process_group``'s rendezvous.

    Equivalent to what ``torchrun`` does when ``--master-port`` is not
    set; we just need a port nothing else is bound to so multiple
    parametrized test cases can spawn without colliding.
    """
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def init_worker_process_group(rank: int, world_size: int, master_port: int):
    """Stand up gloo ``torch.distributed`` and return the DP*CP group.

    The CPU CI image ships gloo but not NCCL; in the no-TP / no-PP setup
    the DP-with-CP group is the whole world, mirroring
    ``parallel_state.get_data_parallel_group(with_context_parallel=True)``
    in Megatron-LM ``finalize_model_grads.py:437``.
    """
    import torch.distributed as _dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    _dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return _dist.new_group(ranks=list(range(world_size)))


def cp_chunk_response_tensor(x, total_length: int, response_length: int):
    """Slice a sample's response tensor to what the current CP rank sees.

    Mirrors the real forward pass: at CP > 1 each rank's attention only
    consumes the two response-token chunks selected by
    ``get_logits_and_tokens_offset_with_cp`` (the same helper used by the
    production reducer in ``cp_utils.get_sum_of_sample_mean``). So the
    "x" we feed into the reducer on a CP rank must be sliced the same
    way to keep the numbers honest.

    Importing locally so callers don't pay the import cost before
    ``stub_megatron_in_worker`` has had a chance to pin (cp_size, cp_rank).
    """
    import torch

    from slime.backends.megatron_utils.cp_utils import get_logits_and_tokens_offset_with_cp

    prompt_length = total_length - response_length
    _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(total_length, response_length)
    c0 = x[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
    c1 = x[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
    return torch.cat([c0, c1])


# ---------------------------------------------------------------------------
# Shared four-rollout fixture, used by both the metric-report distributed
# tests and the backward-grad-norm test. Keeping the data in one place so
# the "train report matches rollout report matches grad-norm baseline"
# contract is anchored on the same numbers everywhere.
#
# Four samples (1 rollout each), total_length=12 (4 prompt + 8 response),
# loss_mask=all-ones. x values differ by orders of magnitude so any cross-
# rank summation bug shows up as a visibly wrong number.
#
# Per-sample token-mean: 4.5 / 45 / 450 / 4500.
# Per-rollout-mean report (sum / num_rollouts):
#       (4.5 + 45 + 450 + 4500) / 4 = 1249.875
# Per-token-loss report (sum_x / total_tokens):
#       (36 + 360 + 3600 + 36000) / 32 = 1249.875
# (the two paths agree by construction so the test expectations stay
# simple — the *report formulas* are still distinct as exercised inside
# ``reduce_train_step_metrics``.)
# ---------------------------------------------------------------------------
FOUR_ROLLOUT_TOTAL_LENGTHS = [12, 12, 12, 12]
FOUR_ROLLOUT_RESPONSE_LENGTHS = [8, 8, 8, 8]
FOUR_ROLLOUT_X_VALUES = [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
    [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0],
]
FOUR_ROLLOUT_EXPECTED_REPORT = 1249.875
