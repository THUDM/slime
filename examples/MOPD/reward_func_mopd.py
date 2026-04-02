"""MOPD reward function with domain-to-teacher routing via OPD_DOMAIN_MODEL_MAP.

The upstream ``reward_func_route_by_domain`` uses ``metadata["domain"]``
directly as the SGLang model name.  This breaks for *stem* and *structured*
because the YAML only defines teachers named ``tool``, ``code``, ``math``.

This wrapper reads ``OPD_DOMAIN_MODEL_MAP`` (e.g.
``tool:tool,stem:tool,structured:tool,code:code,math:math``) to resolve
the correct teacher before calling ``get_model_url``.
"""

import asyncio
import logging
import os

import aiohttp

from slime.rollout.sglang_rollout import get_model_url
from slime.utils.processing_utils import encode_image_for_rollout_engine

logger = logging.getLogger(__name__)

_RETRY_MAX = int(os.getenv("OPD_RETRY_MAX", "3"))
_RETRY_BACKOFF = float(os.getenv("OPD_RETRY_BACKOFF", "1.0"))

_DOMAIN_MODEL_MAP_CACHE = None


def _parse_domain_model_map():
    global _DOMAIN_MODEL_MAP_CACHE
    if _DOMAIN_MODEL_MAP_CACHE is not None:
        return _DOMAIN_MODEL_MAP_CACHE
    raw = os.getenv("OPD_DOMAIN_MODEL_MAP", "")
    mapping = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        domain, model_name = item.split(":", 1)
        domain = domain.strip()
        model_name = model_name.strip()
        if domain and model_name:
            mapping[domain] = model_name
    _DOMAIN_MODEL_MAP_CACHE = mapping
    logger.info("OPD_DOMAIN_MODEL_MAP resolved: %s", mapping)
    return mapping


def _resolve_teacher_model(sample):
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    domain = metadata.get("domain")
    mapping = _parse_domain_model_map()
    if isinstance(domain, str) and domain in mapping:
        return mapping[domain]
    if isinstance(domain, str):
        return domain
    return "tool"


async def reward_func_route_by_domain(args, sample, **kwargs):
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    model_name = _resolve_teacher_model(sample)
    url = get_model_url(args, model_name, "/generate")

    last_exc = None
    for attempt in range(_RETRY_MAX + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as exc:
            last_exc = exc
            is_server_error = isinstance(exc, aiohttp.ClientResponseError) and exc.status >= 500
            is_conn_error = isinstance(exc, aiohttp.ClientConnectionError)
            if (is_server_error or is_conn_error) and attempt < _RETRY_MAX:
                delay = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "OPD teacher %s attempt %d/%d failed (%s), retrying in %.1fs",
                    model_name, attempt + 1, _RETRY_MAX + 1, exc, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise RuntimeError(
                f"OPD teacher {model_name} request failed for {url}: {exc}"
            ) from None
    raise RuntimeError(
        f"OPD teacher {model_name} request failed for {url}: {last_exc}"
    ) from None
