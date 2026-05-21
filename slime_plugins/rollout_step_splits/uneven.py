"""Example custom rollout-step splitters.

Plug via ``--custom-rollout-step-split-path slime_plugins.rollout_step_splits.uneven.<fn>``.
Each splitter has signature ``fn(args, total_lengths) -> list[list[int]]`` and returns
a list of per-step sample-index groups; sum of group sizes typically equals
``len(total_lengths)`` but the splitter is allowed to drop samples (a warning is
logged on the rollout side).
"""

from __future__ import annotations

from argparse import Namespace


def uneven_3_steps_7_8_9(args: Namespace, total_lengths: list[int]) -> list[list[int]]:
    """Split a 24-sample rollout into 3 uneven training steps of size 7 / 8 / 9.

    Used by the ``test_qwen2.5_0.5B_uneven_bs_short`` CI test to exercise the
    pack-first-distribute-second scheduler with non-uniform per-step batch
    sizes. Asserts the expected sample count so the test fails loudly if the
    rollout produced something unexpected.
    """
    n = len(total_lengths)
    assert n == 24, f"uneven_3_steps_7_8_9 expects exactly 24 samples, got {n}"
    return [
        list(range(0, 7)),
        list(range(7, 15)),
        list(range(15, 24)),
    ]
