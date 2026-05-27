"""Load a forged rollout dump from disk so memory-test runs can keep
sglang alive while bypassing real generation.

Plug in by setting:
  --rollout-function-path slime.rollout.forge_load.generate_rollout
  --load-forge-rollout-data <path>

The path follows the same {rollout_id} format convention as
--load-debug-rollout-data:
  - Literal path (recommended for memory tests):
      --load-forge-rollout-data /path/to/forged_dump/rollout_data/0.pt
    Every rollout reuses the same file (rollout_id is left untouched so
    the framework's per-rollout bookkeeping still works).
  - Template path (matches --save-debug-rollout-data layout):
      --load-forge-rollout-data /path/to/dumps/{rollout_id}.pt
    Each rollout loads its own file. If a rollout_id has no file we fall
    back to 0.pt for the training path; eval has no equivalent fallback.

Unlike --load-debug-rollout-data, this path does NOT set
skip_sglang=True / debug_train_only=True (see
slime/utils/arguments.py: skip_sglang computation in _pre_parse_mode and
the debug_train_only flip when load_debug_rollout_data is set), so
sglang servers, router, weight_update and the full colocate
offload/onload dance still run. That is exactly what we want when
measuring real GPU memory.
"""
import logging
import os
from pathlib import Path

import torch

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _resolve_path(args, rollout_id: int, evaluation: bool) -> str | None:
    tpl = getattr(args, "load_forge_rollout_data", None)
    if not tpl:
        raise RuntimeError(
            "--load-forge-rollout-data not set. Pass the dump path, "
            "e.g. /path/to/rollout_data/0.pt (literal) or "
            "/path/to/rollout_data/{rollout_id}.pt (template)."
        )
    # In literal-path mode (no {rollout_id} placeholder) we can't distinguish
    # train vs eval files, so eval is a no-op. Use template mode if you want
    # to also replay an eval dump (--load-forge-rollout-data .../{rollout_id}.pt
    # with eval_<id>.pt files alongside the train ones).
    if evaluation and "{rollout_id}" not in tpl:
        return None
    rid_str = ("eval_" if evaluation else "") + str(rollout_id)
    path = tpl.format(rollout_id=rid_str)
    if os.path.exists(path):
        return path
    # Fallback only for the training path: many memory tests have just 0.pt
    # but want --num-rollout > 1. Eval has no equivalent fallback (we don't
    # want to silently feed training samples to the eval pipeline).
    if not evaluation:
        fallback = tpl.format(rollout_id="0")
        if os.path.exists(fallback):
            logger.info("forge_load: %s missing, falling back to %s", path, fallback)
            return fallback
    return None


def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    path = _resolve_path(args, rollout_id, evaluation)

    if evaluation:
        # Eval is optional for a memory-test run. If no eval dump, no-op.
        if path is None:
            logger.info("forge_load: no eval dump found; returning empty eval result")
            return RolloutFnEvalOutput(data={})
        logger.info("forge_load: loading eval samples from %s", path)
        blob = torch.load(path, weights_only=False)
        samples = [Sample.from_dict(s) for s in blob["samples"]]
        # See train-path note: don't overwrite rollout_id.
        reward_key = args.eval_reward_key or args.reward_key
        rewards = [
            s.reward if (not reward_key or s.reward is None) else s.reward[reward_key]
            for s in samples
        ]
        return RolloutFnEvalOutput(
            data={
                "forge_eval": {
                    "rewards": [r if r is not None else 0.0 for r in rewards],
                    "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
                    "samples": samples,
                }
            }
        )

    if path is None:
        raise RuntimeError(
            f"forge_load: no dump found for rollout_id={rollout_id} "
            f"(--load-forge-rollout-data={args.load_forge_rollout_data!r})"
        )

    logger.info("forge_load: loading samples from %s", path)
    blob = torch.load(path, weights_only=False)
    samples = [Sample.from_dict(s) for s in blob["samples"]]
    # IMPORTANT: do NOT overwrite sample.rollout_id with the current rollout_id.
    # Default-shape rollouts leave rollout_id=None and slime falls back to
    # sample.index in slime/ray/rollout.py (the dp-schedule grouping key).
    # Forcing all samples to share one rollout_id collapses them into a single
    # "rollout", which trips the num_rollouts >= global_batch_size assert in
    # slime/utils/dp_schedule.py.
    logger.info(
        "forge_load: loaded %d samples for rollout_id=%d from %s",
        len(samples), rollout_id, Path(path).name,
    )
    return RolloutFnTrainOutput(samples=samples)
