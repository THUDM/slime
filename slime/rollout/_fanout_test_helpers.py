"""Test-internal compact-rollout helpers used by ``test_qwen2.5_0.5B_fanout_short.py``.

The underscore prefix marks this as test infrastructure — it is not part
of the user-facing slime API and is not re-exported anywhere. It lives
in ``slime/`` only so the test can reference it by a dotted module path
(``--custom-generate-function-path`` resolves a string via
``importlib.import_module``, which can't handle the dots in the e2e
test's filename).

``compact_generate`` fans one input sample out to N siblings sharing the
same ``rollout_id``. That's the contract the rest of the framework
(per-rollout step splitter, per-rollout-mean reducer,
``_validate_rollout_id_annotated`` validator) is built around.
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
        # ``group_index`` is inherited via ``deepcopy`` so production
        # reward normalization keeps the siblings in their prompt group.
        s.rollout_id = sample.index
        siblings.append(s)
    return siblings
