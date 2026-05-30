"""Test-internal compact-rollout helpers used by ``test_qwen2.5_0.5B_fanout_short.py``.

The underscore prefix marks this as test infrastructure — it is not part
of the user-facing slime API and is not re-exported anywhere. It lives
in ``slime/`` only so the test can reference it by a dotted module path
(``--custom-generate-function-path`` / ``--custom-reward-post-process-path``
resolve a string via ``importlib.import_module``, which can't handle the
dots in the e2e test's filename).

Two helpers:

  - ``compact_generate``: fans one input sample out to N siblings
    sharing the same ``group_id``. That's the contract the rest of the
    framework (group-aware step splitter, per-group reducer,
    ``_validate_group_id_annotated`` validator) is built around.

  - ``grpo_normalize_by_group_index``: replaces the default
    ``_post_process_rewards`` reshape-by-shape logic with a proper
    ``group_index``-keyed grouping. The default at
    ``slime/ray/rollout.py:618`` assumes every prompt produced exactly
    ``n_samples_per_prompt`` samples and reshapes by that constant; when
    compact/fanout makes the per-prompt count uneven, the reshape fails
    and the fallback ``view(-1, total)`` collapses everything into ONE
    group, destroying per-prompt centering. ``group_index`` (set by the
    data source per-prompt, preserved through ``deepcopy``) is the right
    key here.
"""

import copy
import os
from collections import defaultdict


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
        # Critical invariant: all siblings share ``group_id`` so the
        # per-group reducer aggregates them as ONE group (not N) and
        # the group-aware step splitter keeps them in the same step.
        # ``group_index`` is inherited via ``deepcopy`` and is what the
        # post-process reward hook below groups on for GRPO normalize.
        s.group_id = sample.index
        siblings.append(s)
    return siblings


def grpo_normalize_by_group_index(args, samples):
    """Drop-in ``--custom-reward-post-process-path`` for compact/fanout.

    The default ``_post_process_rewards`` (``slime/ray/rollout.py:618``)
    reshapes the flat reward tensor as ``(-1, n_samples_per_prompt)``
    when ``total == n_samples_per_prompt * rollout_batch_size``, falling
    back to ``view(-1, total)`` (= one giant group) otherwise. With
    fanout the count per prompt is uneven, so the fallback fires and
    centering is computed across ALL samples in the batch instead of
    per-prompt — that's silently wrong for GRPO.

    This helper groups by ``Sample.group_index`` (the data-source-set
    per-prompt counter, preserved through deepcopy in
    ``compact_generate``) and applies the same mean-center + optional
    std-normalize the default does, just with the correct grouping.

    Returns ``(raw_rewards, normalized_rewards)`` matching the input
    ``samples`` order — same shape as the default's return contract.
    """
    import torch

    raw_rewards = [s.get_reward_value(args) for s in samples]

    # group_index → list of (original_position, raw_reward)
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s.group_index].append((i, raw_rewards[i]))

    out = [0.0] * len(samples)
    use_std = getattr(args, "grpo_std_normalization", True)
    for indexed in groups.values():
        positions = [p for p, _ in indexed]
        rewards = torch.tensor([r for _, r in indexed], dtype=torch.float)
        rewards = rewards - rewards.mean()
        if use_std:
            rewards = rewards / (rewards.std() + 1e-6)
        for pos, r in zip(positions, rewards.tolist(), strict=True):
            out[pos] = r

    return raw_rewards, out
