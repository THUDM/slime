import asyncio
import atexit
import logging
import random

import aiohttp

logger = logging.getLogger(__name__)

from slime.utils.misc import load_function
from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl


_remote_rm_semaphore: asyncio.Semaphore | None = None
_remote_rm_session: aiohttp.ClientSession | None = None

TRANSIENT_ERRORS = (
    aiohttp.ClientConnectionError,
    aiohttp.ServerTimeoutError,
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
)


def _get_remote_rm_semaphore(max_concurrent: int = 64) -> asyncio.Semaphore:
    global _remote_rm_semaphore
    if _remote_rm_semaphore is None:
        _remote_rm_semaphore = asyncio.Semaphore(max_concurrent)
    return _remote_rm_semaphore


def _get_remote_rm_session() -> aiohttp.ClientSession:
    global _remote_rm_session
    if _remote_rm_session is None or _remote_rm_session.closed:
        timeout = aiohttp.ClientTimeout(total=120, connect=10)
        connector = aiohttp.TCPConnector(limit=64)
        _remote_rm_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _remote_rm_session


def _cleanup_remote_rm_session():
    global _remote_rm_session
    if _remote_rm_session is not None and not _remote_rm_session.closed:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_remote_rm_session.close())
            else:
                loop.run_until_complete(_remote_rm_session.close())
        except Exception:
            pass


atexit.register(_cleanup_remote_rm_session)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, TRANSIENT_ERRORS):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500 or exc.status == 429
    return False


async def remote_rm(args, sample: Sample, max_retries: int = 60):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    semaphore = _get_remote_rm_semaphore()
    session = _get_remote_rm_session()
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(args.rm_url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                if not _is_retryable(e) or attempt + 1 >= max_retries:
                    if attempt > 0:
                        logger.warning(f"remote_rm failed after {attempt + 1} attempts, url={args.rm_url}")
                    raise
                backoff = min(2 ** attempt, 30) + random.random()
                logger.info(f"remote_rm error: {type(e).__name__}: {e}, retrying in {backoff:.1f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(backoff)


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
