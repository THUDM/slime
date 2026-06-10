"""Multi-Teacher On-Policy Distillation (MOPD) rollout support for SGLang.

This module provides reward_func and post_process_rewards for fetching log-probs
from multiple domain-specific teacher SGLang servers. Each teacher is identified
by a domain name and has its own rm_url.

Usage:
  --use-mopd
  --mopd-teachers '[{"name": "math_teacher", "domain": "math"}, {"name": "code_teacher", "domain": "code"}]'
  --custom-rm-path slime.rollout.mopd.reward_func
  --custom-reward-post-process-path slime.rollout.mopd.post_process_rewards

The teacher rm_urls are configured via --mopd-teachers JSON, where each entry
can contain an optional "rm_url" field. Alternatively, they can be specified
via the MOPD_TEACHER_URLS environment variable as a JSON dict mapping domain -> URL.
"""

import asyncio
import json
import logging
import os

import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _get_mopd_teacher_configs(args) -> list[dict]:
    """Parse MOPD teacher configurations from args.

    Returns:
        List of teacher config dicts, each containing at least 'name' and 'domain'.
        May also contain 'rm_url' for SGLang mode.
    """
    teachers_str = args.mopd_teachers
    if isinstance(teachers_str, str):
        return json.loads(teachers_str)
    return teachers_str


def _build_payload(sample):
    """Build the SGLang request payload for log-prob extraction."""
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


async def _fetch_teacher_logprobs(session: aiohttp.ClientSession, rm_url: str, payload: dict) -> dict:
    """Fetch log-probs from a single teacher SGLang server."""
    async with session.post(rm_url, json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


def _resolve_teacher_urls(args, teacher_configs: list[dict]) -> dict[str, str]:
    """Resolve rm_url for each teacher domain.

    Priority:
    1. 'rm_url' field in the teacher config
    2. MOPD_TEACHER_URLS environment variable
    3. Fallback: args.rm_url (all teachers share the same URL)
    """
    env_urls = {}
    env_urls_str = os.environ.get("MOPD_TEACHER_URLS", "")
    if env_urls_str:
        env_urls = json.loads(env_urls_str)

    url_map = {}
    for teacher_cfg in teacher_configs:
        domain = teacher_cfg["domain"]
        rm_url = teacher_cfg.get("rm_url") or env_urls.get(domain)
        if rm_url is None:
            rm_url = args.rm_url
        url_map[domain] = rm_url

    return url_map


def _get_sample_domains(sample, all_domains: list[str]) -> list[str] | None:
    """Get the list of teacher domains that should be queried for this sample.

    If sample.metadata contains a 'mopd_domains' key, return those domains
    (filtered to only include valid configured domains).
    Otherwise, return None to indicate all domains should be queried.

    When there is only one configured domain, always returns None since
    routing is unnecessary — all samples must use the single teacher.

    Args:
        sample: The sample to check.
        all_domains: List of all configured domain names.

    Returns:
        List of domain names to query, or None to query all.
    """
    # With only one teacher, routing is unnecessary — always query the only domain
    if len(all_domains) <= 1:
        return None

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample_domains = metadata.get("mopd_domains")
    if sample_domains is None:
        return None  # Query all domains

    if isinstance(sample_domains, str):
        sample_domains = [sample_domains]

    # Filter to only include valid configured domains
    valid_domains = [d for d in sample_domains if d in all_domains]
    if not valid_domains:
        logger.warning(
            f"Sample has mopd_domains={sample_domains} but none match configured domains {all_domains}. "
            f"Falling back to all domains."
        )
        return None

    return valid_domains


async def _reward_func_single(args, sample, **kwargs):
    """Query MOPD teacher servers for a single sample.

    If sample.metadata contains 'mopd_domains' (a list of domain names or a single
    string), only the specified teachers are queried. Otherwise, all teachers are queried.

    Returns:
        dict mapping domain -> raw teacher response (JSON from SGLang).
        This dict is stored in sample.reward and later processed by post_process_rewards.
    """
    teacher_configs = _get_mopd_teacher_configs(args)
    url_map = _resolve_teacher_urls(args, teacher_configs)
    all_domains = list(url_map.keys())

    # Determine which domains to query for this sample
    target_domains = _get_sample_domains(sample, all_domains)
    if target_domains is not None:
        url_map = {d: url_map[d] for d in target_domains}

    payload = _build_payload(sample)

    results = {}

    async with aiohttp.ClientSession() as session:
        tasks = []
        domains = []
        for domain, rm_url in url_map.items():
            domains.append(domain)
            tasks.append(_fetch_teacher_logprobs(session, rm_url, payload))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for domain, resp in zip(domains, responses):
        if isinstance(resp, Exception):
            logger.warning(
                f"MOPD teacher '{domain}' failed: {resp}. Skipping this teacher."
            )
            continue
        results[domain] = resp

    return results


async def reward_func(args, sample_or_samples, **kwargs):
    """Query all MOPD teacher servers for the given sample(s).

    Supports both per-sample and batch calling conventions:
    - When called via async_rm: receives a single Sample, returns a dict
      (domain -> raw teacher response).
    - When called via batched_async_rm: receives a list of Samples, returns
      a list of dicts (one per sample).

    The rm_url for each teacher is determined from:
    1. The 'rm_url' field in the teacher config (if present)
    2. The MOPD_TEACHER_URLS environment variable
    3. Fallback: args.rm_url
    """
    if isinstance(sample_or_samples, list):
        # Batch mode: called from batched_async_rm with a list of samples
        tasks = [_reward_func_single(args, s, **kwargs) for s in sample_or_samples]
        return await asyncio.gather(*tasks)
    else:
        # Single sample mode: called from async_rm
        return await _reward_func_single(args, sample_or_samples, **kwargs)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process MOPD teacher responses and extract per-domain teacher log-probs.

    This function:
    1. Extracts log-probs from each teacher server response
    2. Stores them in sample.mopd_teacher_log_probs[domain]
    3. Returns scalar rewards compatible with GRPO/PPO

    The raw_rewards for each sample is expected to be a dict mapping domain -> response,
    as returned by mopd.reward_func.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    for sample, reward_val, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
        if sample.mopd_teacher_log_probs is None:
            sample.mopd_teacher_log_probs = {}

        if not isinstance(reward_val, dict):
            # If reward_func didn't return a dict (e.g., fallback case), skip
            continue

        for domain, teacher_response in reward_val.items():
            try:
                # Extract log-probs from sglang response format
                log_probs = torch.tensor(
                    [item[0] for item in teacher_response["meta_info"]["input_token_logprobs"][1:]],
                    dtype=torch.float32,
                )
                # Trim to response length
                log_probs = log_probs[-response_length:]
                sample.mopd_teacher_log_probs[domain] = log_probs
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(
                    f"MOPD: Failed to extract log-probs for domain '{domain}': {e}"
                )

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure MOPD distillation, we use 0.0 as the task reward.
    # The learning signal comes from the MOPD advantage applied in compute_advantages_and_returns.
    # If you have task rewards, configure them separately via reward model.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards