"""Custom reward hooks for teacher-routed MOPD.

slime calls the custom rm either per sample (``async_rm``) or with the whole
group (``batched_async_rm``, used by ``--group-rm`` and fan-out generate paths).
Both entry points are provided:

- ``mopd_reward_func``: route one sample, query only its teacher.
- ``mopd_batched_reward_func``: group samples by teacher and query each teacher's
  sub-batch concurrently.

Each sample is forwarded to exactly one teacher, so routing already avoids the
``num_teachers``-per-sample blowup; the batched variant additionally amortizes
HTTP overhead per teacher.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import sys
from typing import Any

from slime_mopd.config import load_mopd_config
from slime_mopd.routing import pop_default_fallback_count, resolve_teacher, set_sample_teacher
from slime_mopd.sglang_client import query_teacher

logger = logging.getLogger(__name__)

_EXEC_SEMAPHORE = asyncio.Semaphore(8)


def _extract_python_code(text: str) -> str:
    """Extract the last fenced Python code block, or return the raw text."""

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def _run_humaneval_check_sync(code: str, timeout_s: float = 5.0) -> float:
    """Execute ``code`` in an isolated subprocess and return 1.0 on success."""

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return 1.0 if proc.returncode == 0 else 0.0
    except Exception:
        return 0.0


async def mopd_humaneval_reward(args: Any, sample: Any, **_: Any) -> float:
    """Evaluate a HumanEval completion by running its unit test suite.

    Expects ``sample.metadata`` (or ``sample.label``) to carry:
    - ``test``: the ``def check(candidate): ...`` test block
    - ``entry_point``: function name passed to ``check(...)``
    - ``canonical_prompt`` (optional): function header + imports prepended when
      the model returned only a body or omitted the imports.
    """

    metadata = getattr(sample, "metadata", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    test_code = metadata.get("test") or getattr(sample, "label", "") or ""
    entry_point = metadata.get("entry_point") or ""
    canonical_prompt = metadata.get("canonical_prompt") or ""
    if not test_code or not entry_point:
        return 0.0

    response = str(getattr(sample, "response", "") or "")
    # Strip <think>...</think> reasoning blocks before extracting code.
    response_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    extracted = _extract_python_code(response_clean)
    if f"def {entry_point}" not in extracted and canonical_prompt:
        program = f"{canonical_prompt}\n{extracted}\n\n{test_code}\n\ncheck({entry_point})\n"
    else:
        program = f"from typing import *\nimport math\nimport collections\nimport itertools\nimport functools\nimport heapq\nimport re\n\n{extracted}\n\n{test_code}\n\ncheck({entry_point})\n"

    timeout_s = float(getattr(args, "humaneval_timeout_s", 5.0))
    async with _EXEC_SEMAPHORE:
        return await asyncio.to_thread(_run_humaneval_check_sync, program, timeout_s)


async def mopd_deepscaler_reward(args: Any, sample: Any, **_: Any) -> float:
    """Boxed-answer reward for the math eval split.

    Needed because the MOPD launcher passes a global ``--custom-rm-path
    slime_mopd.rm.mopd_zero_reward`` (OPD ignores the environment reward), and a
    global custom rm shadows ``rm_type`` for eval datasets too. Naming this path
    per eval dataset restores the built-in ``deepscaler`` scoring.
    """

    from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward

    response = str(getattr(sample, "response", "") or "")
    label = getattr(sample, "label", None)
    return float(get_deepscaler_rule_based_reward(response, label))


async def mopd_zero_reward(args: Any, sample: Any, **_: Any) -> float:
    """Pure-distillation reward for the Megatron teacher path.

    Teacher log-probs come from the actor, so the rollout reward is a constant 0
    and the learning signal is the OPD KL penalty only. Use a real rm instead if
    task rewards should be mixed in.
    """

    return 0.0


async def mopd_reward_func(args: Any, sample: Any, **_: Any) -> dict[str, Any]:
    """Per-sample custom rm: query the sample's routed teacher."""

    config = load_mopd_config(args)
    teacher = resolve_teacher(config, args, sample)
    set_sample_teacher(sample, teacher.id)
    return await query_teacher(args, config, teacher, sample)


async def mopd_batched_reward_func(args: Any, samples: list[Any], **_: Any) -> list[Any]:
    """Batched custom rm: gather by teacher, query concurrently, scatter back."""

    config = load_mopd_config(args)
    groups: dict[str, list[tuple[int, Any]]] = {}
    teacher_by_id = {}
    for position, sample in enumerate(samples):
        teacher = resolve_teacher(config, args, sample)
        set_sample_teacher(sample, teacher.id)
        teacher_by_id[teacher.id] = teacher
        groups.setdefault(teacher.id, []).append((position, sample))

    results: list[Any] = [None] * len(samples)
    semaphore = asyncio.Semaphore(max(1, int(config.max_concurrency)))

    async def run_one(teacher, position: int, sample: Any) -> None:
        async with semaphore:
            results[position] = await query_teacher(args, config, teacher, sample)

    await asyncio.gather(
        *(
            run_one(teacher_by_id[teacher_id], position, sample)
            for teacher_id, items in groups.items()
            for position, sample in items
        )
    )

    fallback = pop_default_fallback_count()
    if fallback:
        logger.warning(
            "mopd: %d/%d samples had no teacher annotation/dataset binding and used default_teacher %r",
            fallback,
            len(samples),
            config.default_teacher,
        )
    if logger.isEnabledFor(logging.INFO):
        summary = {teacher_id: len(items) for teacher_id, items in sorted(groups.items())}
        logger.info("mopd: batched routing sent %d samples across teachers %s", len(samples), summary)
    return results
