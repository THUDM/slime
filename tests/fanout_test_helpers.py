"""Compact-rollout helpers for ``test_qwen2.5_0.5B_fanout_short.py``.

These helpers are imported by module path from the Ray job started by the E2E
test. They live on the test-only portion of ``PYTHONPATH`` because they are
test fixtures, not part of slime's public or internal runtime API.

``compact_generate`` fans one input sample out to N siblings sharing the same
``rollout_id``. The normal BatchBuilder handles prompt-group reward statistics
and per-rollout loss averaging; this fixture does not override either path.
"""

import copy
import os

MAX_FANOUT = 3

# Each invocation appends one line. The test file reads this after train
# completes to assert the framework actually drove the custom path for
# every prompt (no silent bypass / no double-submission).
COUNTER_FILE_ENV = "SLIME_FANOUT_TEST_COUNTER_FILE"


async def compact_generate(args, sample, sampling_params):
    """One prompt → N siblings, deterministic N = 1 + (index % MAX_FANOUT).

    Strategy: call sglang once, deepcopy N-1 times. Bounded GPU cost —
    we're pinning the framework's per-rollout handling, not generation
    diversity.
    """
    from slime.rollout.sglang_rollout import generate

    counter_path = os.environ.get(COUNTER_FILE_ENV)
    if counter_path:
        try:
            with open(counter_path, "a") as f:
                f.write(f"{sample.index}\n")
        except OSError:
            # Counter file is best-effort — never fail training because of it.
            pass

    base_sample = await generate(args, sample, sampling_params)

    n = 1 + (sample.index % MAX_FANOUT)
    siblings = []
    for _ in range(n):
        s = copy.deepcopy(base_sample)
        # Critical invariant: all siblings share ``rollout_id`` so the
        # per-rollout reducer aggregates them as ONE rollout (not N) and
        # the rollout-aware step splitter keeps them in the same step.
        # ``group_index`` is inherited via ``deepcopy`` and identifies the
        # prompt group for the production GRPO normalization path.
        s.rollout_id = sample.index
        siblings.append(s)
    return siblings
