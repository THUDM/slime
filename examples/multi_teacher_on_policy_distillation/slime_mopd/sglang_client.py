"""Minimal async client for querying teacher token log-probabilities.

The SGLang teacher payload mirrors ``slime.rollout.on_policy_distillation``:
prefill the full student token sequence with ``max_new_tokens=0`` and read back
``meta_info.input_token_logprobs``, so the teacher scores exactly the tokens the
student sampled.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_shared_session: aiohttp.ClientSession | None = None


def _build_payload(args: Any, sample: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": getattr(args, "rollout_temperature", 1.0),
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    multimodal = getattr(sample, "multimodal_inputs", None) or {}
    images = multimodal.get("images") if isinstance(multimodal, dict) else None
    if images:
        from slime.utils.processing_utils import encode_image_for_rollout_engine

        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]

    return payload


async def _get_session(config) -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(limit=max(1, int(config.max_concurrency)), enable_cleanup_closed=True)
        timeout = aiohttp.ClientTimeout(total=float(config.request_timeout_s))
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _shared_session


async def query_teacher(args: Any, config, teacher, sample: Any) -> dict[str, Any]:
    """POST one sample's tokens to a teacher and return the raw response."""

    url = teacher.rm_url or getattr(args, "rm_url", None)
    if not url:
        raise ValueError(f"teacher {teacher.id!r} has no rm_url and args.rm_url is unset")

    payload = _build_payload(args, sample)
    session = await _get_session(config)
    last_error: Exception | None = None
    for attempt in range(max(1, int(config.max_retries))):
        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as error:
            last_error = error
            if attempt + 1 >= max(1, int(config.max_retries)):
                break
            backoff = min(2**attempt, 30) + random.random()
            logger.info(
                "mopd: teacher %s query failed (%s), retrying in %.1fs (%d/%d)",
                teacher.id,
                type(error).__name__,
                backoff,
                attempt + 1,
                int(config.max_retries),
            )
            await asyncio.sleep(backoff)

    raise RuntimeError(f"teacher {teacher.id!r} query failed after {int(config.max_retries)} attempts") from last_error
