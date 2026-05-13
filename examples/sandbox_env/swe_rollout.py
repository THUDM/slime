"""Slime rollout entrypoint.

This file is intentionally limited to Slime-facing concerns: tokenizer/mask
state, sample finalization, group concurrency, and the public
``generate_rollout`` function used by ``train_async.py``.  Sandbox execution
for one sample lives in ``sandbox_runtime``.
"""

from __future__ import annotations

import asyncio
import copy
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .sandbox_runtime import (
    decode_swe_rollout_config,
    env_or_arg,
    prepare_batch_log_dir,
    prompt_to_text,
    run_sample_in_sandbox,
)


_SLIME_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _SLIME_REPO_ROOT not in sys.path:
    sys.path.insert(0, _SLIME_REPO_ROOT)

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.async_utils import run
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample


_ROLLOUT_STATE: dict[str, Any] = {}


def build_fallback_messages(prompt: str, failure_text: str) -> list[dict[str, Any]]:
    """Builds a minimal user+assistant message pair used when the sandbox run fails and no real trajectory exists."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": failure_text or "Execution failed.", "step_loss_mask": 1},
    ]


def _get_rollout_state(args) -> dict[str, Any]:
    """Lazily initializes and caches the tokenizer and loss mask generator for the current checkpoint."""
    cache_key = f"{args.hf_checkpoint}|{args.loss_mask_type}"
    if cache_key in _ROLLOUT_STATE:
        return _ROLLOUT_STATE[cache_key]
    generate_state = GenerateState(args)
    tokenizer = generate_state.tokenizer
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
    state = {
        "generate_state": generate_state,
        "tokenizer": tokenizer,
        "mask_generator": mask_generator,
    }
    _ROLLOUT_STATE[cache_key] = state
    return state


def _finalize_sample(
    args,
    sample: Sample,
    *,
    final_messages: list[dict[str, Any]],
    final_tools: list[dict[str, Any]] | None,
    reward: float,
    status: Sample.Status,
    turn_responses: list[str],
    trajectory: list[dict[str, Any]],
    extra_metadata: dict[str, Any],
) -> Sample:
    """Converts raw sandbox output into a finalized Slime Sample with token IDs, loss mask, reward, and metadata."""
    state = _get_rollout_state(args)
    mask_generator: MultiTurnLossMaskGenerator = state["mask_generator"]
    if status == Sample.Status.FAILED:
        reward = 0.0
    token_ids, loss_mask = mask_generator.get_loss_mask(final_messages, tools=final_tools)
    response_length = mask_generator.get_response_lengths([loss_mask])[0]
    if response_length <= 0:
        fallback_messages = build_fallback_messages(
            prompt_to_text(sample.prompt),
            "No assistant tokens were produced; converted to failure sample.",
        )
        token_ids, loss_mask = mask_generator.get_loss_mask(fallback_messages, tools=None)
        response_length = mask_generator.get_response_lengths([loss_mask])[0]
        final_messages = fallback_messages
        status = Sample.Status.FAILED
        reward = 0.0
    sample.tokens = token_ids
    sample.response_length = response_length
    sample.loss_mask = loss_mask[-response_length:]
    sample.response = "\n\n".join(resp for resp in turn_responses if resp.strip())
    sample.reward = reward
    sample.status = status
    sample.metadata = {
        **(sample.metadata or {}),
        "trajectory": trajectory,
        "training_messages": final_messages,
        **extra_metadata,
    }
    return sample


async def _run_single_sample(
    args,
    sample: Sample,
    *,
    evaluation: bool,
    rollout_id: int,
    sample_idx: int,
    sample_semaphore: asyncio.Semaphore,
) -> Sample:
    """Acquires the sample semaphore, runs one sample in the sandbox, and returns the finalized Sample."""
    cfg = decode_swe_rollout_config(args)
    sample = copy.deepcopy(sample)
    metadata = dict(sample.metadata or {})
    prompt_text = prompt_to_text(sample.prompt)

    async with sample_semaphore:
        result = await run_sample_in_sandbox(
            args=args,
            rollout_state=_get_rollout_state(args),
            metadata=metadata,
            prompt_text=prompt_text,
            cfg=cfg,
            evaluation=evaluation,
            rollout_id=rollout_id,
            sample_idx=sample_idx,
        )

    training_messages = result.final_messages
    turn_responses = result.turn_responses
    if not training_messages:
        training_messages = build_fallback_messages(prompt_text, result.failure_reason or "Sandbox rollout failed.")
        if not turn_responses:
            turn_responses.append(result.failure_reason or "Sandbox rollout failed.")
    return _finalize_sample(
        args,
        sample,
        final_messages=training_messages,
        final_tools=result.final_tools,
        reward=result.reward,
        status=Sample.Status.FAILED if result.failed else Sample.Status.COMPLETED,
        turn_responses=turn_responses,
        trajectory=result.trajectory,
        extra_metadata=result.extra_metadata,
    )


async def _run_group(
    args,
    group: list[Sample],
    *,
    evaluation: bool,
    rollout_id: int,
    sample_idx_offset: int,
    sample_semaphore: asyncio.Semaphore,
) -> list[Sample]:
    """Runs all samples in one GRPO group concurrently; cancels remaining tasks if any raises."""
    tasks = [
        asyncio.create_task(
            _run_single_sample(
                args,
                sample,
                evaluation=evaluation,
                rollout_id=rollout_id,
                sample_idx=sample_idx_offset + local_idx,
                sample_semaphore=sample_semaphore,
            )
        )
        for local_idx, sample in enumerate(group)
    ]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def generate_rollout_async(args, rollout_id: int, data_buffer, evaluation: bool = False) -> RolloutFnTrainOutput:
    """Main async rollout driver: fetches groups, dispatches with concurrency limits, collects results and metrics."""
    if evaluation:
        raise ValueError("SWE rollout currently implements training rollout only.")

    fixed_seed = int(os.environ.get("SWE_BATCH_SEED", "42"))
    random.seed(fixed_seed)
    np.random.seed(fixed_seed)

    target_group_count = max(1, int(getattr(args, "rollout_batch_size", 1)))
    over_sampling_batch_size = max(
        target_group_count,
        int(os.environ.get("SWE_OVER_SAMPLING_BATCH_SIZE", str(target_group_count))),
    )
    groups = data_buffer.get_samples(over_sampling_batch_size)
    start_time = time.time()
    default_group_concurrency = target_group_count
    default_sample_concurrency = max(
        1,
        target_group_count * int(getattr(args, "n_samples_per_prompt", 1)),
    )
    max_sample_concurrency = int(os.environ.get("SWE_MAX_SAMPLE_CONCURRENCY", "0"))
    group_concurrency = max(1, int(os.environ.get("SWE_GROUP_CONCURRENCY", default_group_concurrency)))
    sample_concurrency = max(1, int(os.environ.get("SWE_SAMPLE_CONCURRENCY", default_sample_concurrency)))
    if max_sample_concurrency > 0 and sample_concurrency > max_sample_concurrency:
        sample_concurrency = max_sample_concurrency
    total_groups = len(groups)
    total_samples = sum(len(group) for group in groups)
    print(
        f"[rollout] rollout_id={rollout_id} target_groups={target_group_count} submitted_groups={total_groups} "
        f"total_samples={total_samples} group_concurrency={group_concurrency} "
        f"sample_concurrency={sample_concurrency}",
        file=sys.stderr,
    )
    sample_semaphore = asyncio.Semaphore(sample_concurrency)
    log_root = env_or_arg(args, "SWE_LOG_ROOT", "swe_log_root", None)
    prepare_batch_log_dir(log_root)

    async def _run_group_with_order(
        group_idx: int,
        group: list[Sample],
        group_sample_idx_offset: int,
    ) -> tuple[int, list[Sample]]:
        """Runs a group and returns its original index so completion order can be restored."""
        result = await _run_group(
            args,
            group,
            evaluation=evaluation,
            rollout_id=rollout_id,
            sample_idx_offset=group_sample_idx_offset,
            sample_semaphore=sample_semaphore,
        )
        return group_idx, result

    pending_tasks: set[asyncio.Task[tuple[int, list[Sample]]]] = set()
    completed_groups: dict[int, list[Sample]] = {}
    completed_group_order: list[int] = []
    sample_idx_offset = 0

    for group_idx, group in enumerate(groups):
        if len(completed_groups) >= target_group_count:
            break
        while len(pending_tasks) >= group_concurrency:
            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                finished_group_idx, finished_group = await task
                if len(completed_groups) < target_group_count:
                    completed_groups[finished_group_idx] = finished_group
                    completed_group_order.append(finished_group_idx)
            if len(completed_groups) >= target_group_count:
                break
        if len(completed_groups) >= target_group_count:
            break
        pending_tasks.add(asyncio.create_task(_run_group_with_order(group_idx, group, sample_idx_offset)))
        sample_idx_offset += len(group)

    try:
        while pending_tasks and len(completed_groups) < target_group_count:
            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                finished_group_idx, finished_group = await task
                if len(completed_groups) < target_group_count:
                    completed_groups[finished_group_idx] = finished_group
                    completed_group_order.append(finished_group_idx)
    finally:
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)

    selected_group_indices = completed_group_order[:target_group_count]
    processed_groups = [completed_groups[group_idx] for group_idx in selected_group_indices]
    flat_samples = [sample for group in processed_groups for sample in group]
    success_count = sum(1 for sample in flat_samples if float(sample.reward or 0.0) > 0.0)
    mean_turns = 0.0
    if flat_samples:
        mean_turns = sum(len((sample.metadata or {}).get("trajectory", [])) for sample in flat_samples) / len(flat_samples)
    metrics = {
        "swe/pass_rate": success_count / max(1, len(flat_samples)),
        "swe/success_count": success_count,
        "swe/sample_count": len(flat_samples),
        "swe/mean_turns": mean_turns,
        "swe/target_group_count": target_group_count,
        "swe/submitted_group_count": total_groups,
        "swe/completed_group_count": len(processed_groups),
        "swe/cancelled_group_count": max(0, total_groups - len(processed_groups)),
        "swe/group_concurrency": group_concurrency,
        "swe/sample_concurrency": sample_concurrency,
        "swe/rollout_time_sec": time.time() - start_time,
    }
    return RolloutFnTrainOutput(samples=processed_groups, metrics=metrics)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Synchronous entry point called by train_async.py; blocks until the async rollout completes."""
    return run(generate_rollout_async(args, rollout_id, data_buffer, evaluation=evaluation))
