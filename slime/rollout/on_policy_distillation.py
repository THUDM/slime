import asyncio
import logging
import os

import aiohttp
import torch

from slime.rollout.sglang_rollout import get_model_url
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_RETRY_MAX = int(os.getenv("OPD_RETRY_MAX", "3"))
_RETRY_BACKOFF = float(os.getenv("OPD_RETRY_BACKOFF", "1.0"))


def _raise_request_failure(url, exc, model_name=None):
    model_prefix = f"teacher {model_name} " if model_name else "teacher "
    if isinstance(exc, aiohttp.ClientResponseError):
        raise RuntimeError(
            f"OPD {model_prefix}request failed with HTTP {exc.status} for {url}: {exc.message}"
        ) from None
    raise RuntimeError(f"OPD {model_prefix}request failed for {url}: {exc}") from None


def _build_payload(sample):
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

    return payload


def _parse_domain_model_map():
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
    return mapping


def _get_model_url(args, model_name):
    routers = getattr(args, "sglang_model_routers", None) or {}
    if model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}/generate"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


_RM_TYPE_TO_DOMAIN = {
    "code_execution": "code",
    "math": "math",
}


def _resolve_teacher_model(args, sample):
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    domain = metadata.get("domain")
    if not isinstance(domain, str):
        rm_type = metadata.get("rm_type")
        if isinstance(rm_type, str) and rm_type in _RM_TYPE_TO_DOMAIN:
            domain = _RM_TYPE_TO_DOMAIN[rm_type]
    mapping = _parse_domain_model_map()
    if isinstance(domain, str) and domain in mapping:
        return mapping[domain]
    if isinstance(domain, str):
        return domain
    return "tool"


async def reward_func(args, sample, **kwargs):
    payload = _build_payload(sample)
    last_exc = None
    for attempt in range(_RETRY_MAX + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(args.rm_url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as exc:
            last_exc = exc
            is_server_error = isinstance(exc, aiohttp.ClientResponseError) and exc.status >= 500
            is_conn_error = isinstance(exc, aiohttp.ClientConnectionError)
            if (is_server_error or is_conn_error) and attempt < _RETRY_MAX:
                delay = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning("OPD teacher attempt %d/%d failed (%s), retrying in %.1fs",
                               attempt + 1, _RETRY_MAX + 1, exc, delay)
                await asyncio.sleep(delay)
                continue
            _raise_request_failure(args.rm_url, exc)
    _raise_request_failure(args.rm_url, last_exc)


async def reward_func_route_by_domain(args, sample, **kwargs):
    payload = _build_payload(sample)
    model_name = _resolve_teacher_model(args, sample)
    url = _get_model_url(args, model_name)
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
                logger.warning("OPD teacher %s attempt %d/%d failed (%s), retrying in %.1fs",
                               model_name, attempt + 1, _RETRY_MAX + 1, exc, delay)
                await asyncio.sleep(delay)
                continue
            _raise_request_failure(url, exc, model_name=model_name)
    _raise_request_failure(url, last_exc, model_name=model_name)


async def reward_func_route_by_domain(args, sample, **kwargs):
    """Like ``reward_func`` but routes to the per-domain teacher via ``get_model_url``."""
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

    model_name = sample.metadata.get("domain", None)
    if model_name is None:
        raise KeyError("sample.metadata['domain'] is not set")
    url = get_model_url(args, model_name, "/generate")
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
