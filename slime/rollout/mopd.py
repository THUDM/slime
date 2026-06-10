"""Multi-Teacher On-Policy Distillation (MOPD) rollout support for SGLang.

This module provides reward_func and post_process_rewards for fetching teacher
data from multiple domain-specific SGLang teacher servers. Each teacher is
identified by a domain name and has its own rm_url.

Supports three distillation modes (controlled by --mopd-distill-type):
  - token_level: Extract per-token log-probs from SGLang's input_token_logprobs.
  - top_k: Extract top-k log-probs and token indices per position using
    SGLang's top_logprobs_num parameter.
  - full_vocab: Not supported with SGLang teacher mode (requires Megatron
    in-process teacher for full vocabulary logits). Raises a clear error
    if full_vocab is requested with --mopd-teacher-mode=sglang.

Usage (pure distillation, alpha=0):
  --use-mopd --mopd-teacher-mode sglang
  --mopd-teachers '[{"name": "math_teacher", "domain": "math"}]'
  (custom-rm-path and custom-reward-post-process-path are auto-configured)

Usage (with task rewards, alpha>0):
  --use-mopd --mopd-teacher-mode sglang --mopd-alpha 0.5
  --mopd-teachers '[{"name": "math_teacher", "domain": "math"}]'
  --rm-type math
  (combined_reward_func and combined_post_process_rewards are auto-configured)

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


def _get_all_domain_names(args) -> list[str]:
    """Get all configured MOPD teacher domain names from args.

    Returns:
        List of domain name strings (e.g., ['origin', 'enhanced']).
    """
    configs = _get_mopd_teacher_configs(args)
    return [c.get("domain", c.get("name", "")) for c in configs]


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


def _build_payload(sample, args):
    """Build the SGLang request payload for teacher data extraction.

    The payload differs based on --mopd-distill-type:
    - token_level: return_logprob=True, no top_logprobs_num
    - top_k: return_logprob=True, top_logprobs_num=mopd_topk_k
    - full_vocab: raises ValueError (not supported with SGLang)
    """
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

    # Determine distill type
    mopd_distill_type = getattr(args, "mopd_distill_type", "token_level")

    if mopd_distill_type == "top_k":
        topk_k = getattr(args, "mopd_topk_k", 1024)
        payload["top_logprobs_num"] = topk_k
    elif mopd_distill_type == "full_vocab":
        raise ValueError(
            "MOPD full_vocab mode is not supported with SGLang teacher mode. "
            "SGLang cannot efficiently return full-vocabulary logits. "
            "Use --mopd-teacher-mode=megatron for full_vocab, or switch to "
            "--mopd-distill-type=top_k for an accurate approximation with "
            "much lower memory usage."
        )
    # token_level: no additional parameters needed

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    return payload


async def _fetch_teacher_logprobs(
    session: aiohttp.ClientSession,
    rm_url: str,
    payload: dict,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    """Fetch log-probs from a single teacher SGLang server with retry.

    Retries on transient network errors (connection issues, partial reads,
    server errors) that are likely to succeed on a subsequent attempt.
    Non-retryable errors (e.g., 4xx client errors) are raised immediately.

    Args:
        session: The aiohttp client session.
        rm_url: The teacher server URL.
        payload: The request payload.
        max_retries: Maximum number of retry attempts (default 3).
        retry_delay: Base delay in seconds between retries (default 5.0).
            Actual delay is ``retry_delay * (attempt + 1)`` with jitter.

    Returns:
        The parsed JSON response from the teacher server.

    Raises:
        The last exception if all retries are exhausted.
    """
    import random

    last_exc = None
    for attempt in range(max_retries):
        try:
            async with session.post(rm_url, json=payload) as resp:
                # 4xx errors are client errors — retrying won't help.
                if resp.status >= 400 and resp.status < 500:
                    resp.raise_for_status()
                # 5xx or network-level errors are transient — retry.
                resp.raise_for_status()
                result = await resp.json()

                # Validate that the response contains logprob data.
                # SGLang's return_logprob is a per-request parameter (not a
                # server-side flag).  If meta_info lacks input_token_logprobs,
                # the most likely cause is that the URL points to the wrong
                # SGLang instance (e.g. the student rollout server) or a
                # gateway that strips request fields.
                meta_info = result.get("meta_info", {})
                if not isinstance(meta_info, dict) or "input_token_logprobs" not in meta_info:
                    logger.error(
                        f"MOPD: SGLang teacher response from {rm_url} does NOT contain "
                        f"'input_token_logprobs' in meta_info. Check that the URL "
                        f"points to a SGLang server with return_logprob support. "
                        f"Response meta_info keys: {list(meta_info.keys()) if isinstance(meta_info, dict) else meta_info}. "
                        f"Request payload had return_logprob={payload.get('return_logprob')}, "
                        f"logprob_start_len={payload.get('logprob_start_len')}."
                    )
                else:
                    logger.info(
                        f"MOPD: SGLang teacher response from {rm_url} OK, "
                        f"input_token_logprobs count={len(meta_info['input_token_logprobs'])}"
                    )

                return result
        except (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError,
                aiohttp.ServerDisconnectedError, asyncio.TimeoutError,
                aiohttp.ClientResponseError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                # 5xx server errors are retryable; ClientPayloadError (e.g.
                # ContentLengthError) is typically caused by the server closing
                # the connection mid-stream and is also retryable.
                is_retryable = True
                if isinstance(exc, aiohttp.ClientResponseError) and exc.status < 500:
                    is_retryable = False
                if is_retryable:
                    delay = retry_delay * (attempt + 1) + random.uniform(0, 2)
                    logger.warning(
                        f"MOPD teacher request to {rm_url} failed (attempt {attempt + 1}/{max_retries}): "
                        f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
            raise
    raise last_exc  # Should not reach here, but just in case


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
            logger.warning(
                f"MOPD: No explicit URL configured for teacher domain '{domain}', "
                f"falling back to args.rm_url ({rm_url}). "
                f"Set 'rm_url' in --mopd-teachers or the MOPD_TEACHER_URLS "
                f"environment variable to override."
            )
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


# Default timeout for MOPD teacher HTTP requests (in seconds).
# Individual teacher requests for long sequences (especially multimodal)
# can take several minutes, so we set a generous timeout.
_MOPD_TEACHER_TIMEOUT = aiohttp.ClientTimeout(total=600, connect=30, sock_read=300)


async def _reward_func_single(args, sample, **kwargs):
    """Query MOPD teacher servers for a single sample.

    If sample.metadata contains 'mopd_domains' (a list of domain names or a single
    string), only the specified teachers are queried. Otherwise, all teachers are queried.

    Each teacher request is retried on transient errors (connection resets,
    partial reads, server errors) before giving up. When a teacher is
    permanently unreachable after retries, it is skipped with a warning and
    its domain data will be missing from the result dict.

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

    payload = _build_payload(sample, args)

    # Read retry config from args (with sensible defaults)
    max_retries = getattr(args, "mopd_teacher_max_retries", 3)
    retry_delay = getattr(args, "mopd_teacher_retry_delay", 5.0)

    results = {}

    async with aiohttp.ClientSession(timeout=_MOPD_TEACHER_TIMEOUT) as session:
        tasks = []
        domains = []
        for domain, rm_url in url_map.items():
            domains.append(domain)
            tasks.append(
                _fetch_teacher_logprobs(session, rm_url, payload,
                                        max_retries=max_retries,
                                        retry_delay=retry_delay)
            )

        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for domain, resp in zip(domains, responses):
        if isinstance(resp, Exception):
            logger.warning(
                f"MOPD teacher '{domain}' failed after retries: {resp}. "
                f"Skipping this teacher."
            )
            continue
        results[domain] = resp

    # Record which domains were targeted so the extraction code can
    # distinguish between "not queried" (domain routed away) and
    # "queried but failed" (should fill with -inf fallback).
    results["__target_domains__"] = list(domains)

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


def _extract_teacher_data_from_responses(args, samples: list[Sample]):
    """Extract per-domain teacher data from MOPD teacher responses stored in sample.reward.

    This is the core extraction logic shared by post_process_rewards and
    combined_post_process_rewards. It reads teacher responses from sample.reward
    (which should be a dict mapping domain -> SGLang response JSON) and populates
    sample.mopd_teacher_log_probs, sample.mopd_teacher_topk_logits, and
    sample.mopd_teacher_topk_indices.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    mopd_distill_type = getattr(args, "mopd_distill_type", "token_level")

    for sample, reward_val, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
        if sample.mopd_teacher_log_probs is None:
            sample.mopd_teacher_log_probs = {}

        if mopd_distill_type == "top_k":
            if sample.mopd_teacher_topk_logits is None:
                sample.mopd_teacher_topk_logits = {}
            if sample.mopd_teacher_topk_indices is None:
                sample.mopd_teacher_topk_indices = {}

        if not isinstance(reward_val, dict):
            # If reward_func didn't return a dict (e.g., fallback case), skip
            continue

        for domain, teacher_response in reward_val.items():
            # Skip internal metadata keys
            if domain.startswith("__") and domain.endswith("__"):
                continue
            try:
                meta_info = teacher_response["meta_info"]
                input_token_logprobs = meta_info["input_token_logprobs"]

                # --- token_level: always extract (needed even for top_k for IS weights) ---
                # input_token_logprobs format: list of [log_prob, token_id, token_text]
                # Skip the first entry (prompt token before any generation)
                logprobs_from_response = input_token_logprobs[1:]
                if len(logprobs_from_response) < response_length:
                    logger.warning(
                        f"MOPD: SGLang returned {len(logprobs_from_response)} logprob entries "
                        f"for domain '{domain}', but response_length={response_length}. "
                        f"Padding with -inf for missing positions."
                    )
                log_probs = torch.tensor(
                    [item[0] for item in logprobs_from_response],
                    dtype=torch.float32,
                )
                if log_probs.size(0) < response_length:
                    # Pad shorter log_probs with -inf so downstream code
                    # doesn't misalign position indices.
                    log_probs = torch.nn.functional.pad(
                        log_probs, (0, response_length - log_probs.size(0)), value=float("-inf")
                    )
                # Trim to response length (in case SGLang returns more)
                log_probs = log_probs[-response_length:]
                sample.mopd_teacher_log_probs[domain] = log_probs

                valid_mask = log_probs.isfinite()
                if valid_mask.any():
                    logger.info(
                        f"MOPD: Received teacher logprobs for domain '{domain}': "
                        f"len={log_probs.size(0)}, valid={valid_mask.sum().item()}, "
                        f"mean={log_probs[valid_mask].mean().item():.4f}"
                    )
                else:
                    logger.info(
                        f"MOPD: Received teacher logprobs for domain '{domain}': "
                        f"len={log_probs.size(0)}, all -inf"
                    )

                # --- top_k: extract top-k log-probs and indices per position ---
                if mopd_distill_type == "top_k":
                    # SGLang returns top-k logprobs in meta_info["input_top_logprobs"]
                    # Format: list (one entry per position) of list of (log_prob, token_id, token_text)
                    # tuples. Same length as input_token_logprobs.
                    input_top_logprobs = meta_info.get("input_top_logprobs")
                    if input_top_logprobs is None:
                        logger.warning(
                            f"MOPD top_k: SGLang response for domain '{domain}' does not contain "
                            f"'input_top_logprobs'. Make sure top_logprobs_num is set in the "
                            f"SGLang request payload. Falling back to token_level for this domain."
                        )
                        continue

                    # Skip first entry (same as input_token_logprobs), trim to response
                    top_logprobs_response = input_top_logprobs[1:]
                    if len(top_logprobs_response) < response_length:
                        logger.warning(
                            f"MOPD top_k: SGLang returned {len(top_logprobs_response)} "
                            f"top-logprobs entries for domain '{domain}', but "
                            f"response_length={response_length}. Missing positions "
                            f"will be padded with -inf logits and index 0."
                        )
                        # Pad with None entries so the loop below generates
                        # [-inf, ..., -inf] / [0, ..., 0] for missing positions
                        top_logprobs_response = top_logprobs_response + [None] * (response_length - len(top_logprobs_response))
                    if len(top_logprobs_response) > response_length:
                        top_logprobs_response = top_logprobs_response[-response_length:]

                    # Each position: list of (log_prob, token_id, token_text) tuples
                    # (sorted by log_prob desc). token_text is None when
                    # return_text_in_logprobs is not set.
                    # Convert to: topk_logits[pos] = [log_prob_0, log_prob_1, ...]
                    #             topk_indices[pos] = [token_id_0, token_id_1, ...]
                    # Padding entries use -inf logit so downstream TP sharding and
                    # valid_topk_mask can correctly identify them as invalid.
                    topk_k = getattr(args, "mopd_topk_k", 1024)
                    NEG_INF = float("-inf")

                    topk_logits_list = []   # [seq_len][k] float
                    topk_indices_list = []  # [seq_len][k] int
                    short_positions = 0  # Count positions with fewer than topk_k entries

                    for pos_data in top_logprobs_response:
                        if pos_data is None or len(pos_data) == 0:
                            # No top-k data for this position (e.g., padding)
                            topk_logits_list.append([NEG_INF] * topk_k)
                            topk_indices_list.append([0] * topk_k)
                            short_positions += 1
                            continue

                        # pos_data: list of (log_prob, token_id, token_text) tuples
                        pos_logits = []
                        pos_indices = []
                        for entry in pos_data[:topk_k]:
                            # entry: (log_prob, token_id, token_text)
                            pos_logits.append(float(entry[0]))
                            pos_indices.append(int(entry[1]))

                        # Pad to topk_k if fewer entries returned
                        # Use -inf logit for padding so downstream valid_topk_mask
                        # detection (checking for -inf entries) works correctly.
                        if len(pos_logits) < topk_k:
                            short_positions += 1
                            while len(pos_logits) < topk_k:
                                pos_logits.append(NEG_INF)
                                pos_indices.append(0)

                        topk_logits_list.append(pos_logits)
                        topk_indices_list.append(pos_indices)

                    if short_positions > 0:
                        logger.warning(
                            f"MOPD top_k: {short_positions}/{len(top_logprobs_response)} "
                            f"positions in domain '{domain}' returned fewer than {topk_k} "
                            f"top-k entries from SGLang. Padded with -inf logits. "
                            f"Consider reducing --mopd-topk-k or checking SGLang's "
                            f"top_logprobs_num setting."
                        )

                    sample.mopd_teacher_topk_logits[domain] = topk_logits_list
                    sample.mopd_teacher_topk_indices[domain] = topk_indices_list

            except (KeyError, IndexError, TypeError) as e:
                # Provide an actionable message for the most common cause:
                # SGLang server not returning logprobs.
                if isinstance(e, KeyError) and str(e) in ("'input_token_logprobs'", "input_token_logprobs"):
                    meta_keys = list(teacher_response.get("meta_info", {}).keys()) if isinstance(teacher_response.get("meta_info"), dict) else "N/A"
                    logger.error(
                        f"MOPD: SGLang response for domain '{domain}' missing "
                        f"'input_token_logprobs'. meta_info keys: {meta_keys}. "
                        f"Check teacher URL configuration."
                    )
                else:
                    logger.warning(
                        f"MOPD: Failed to extract teacher data for domain '{domain}': {e}"
                    )

    # --- Fill in missing domains with zero/fallback data ---
    # When a teacher request fails (e.g., ContentLengthError, connection reset),
    # the domain is absent from reward_val and thus from the sample's dicts.
    # Previously this would produce None placeholders in the training data
    # pipeline, which could cause NCCL deadlocks because different DP ranks
    # may end up with different computational graphs.
    #
    # IMPORTANT: We only fill fallback data for domains that were *actually
    # queried* for this sample (i.e., in target_domains). Domains that were
    # excluded by per-sample domain routing (sample.metadata["mopd_domains"])
    # should NOT be filled — they intentionally don't participate in this
    # sample's loss computation, and filling them would incorrectly produce
    # -inf fallback tensors that contribute zero KL but still occupy memory
    # and trigger unnecessary backward operations.
    all_configured_domains = _get_all_domain_names(args)
    for sample, reward_val, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
        if sample.mopd_teacher_log_probs is None:
            sample.mopd_teacher_log_probs = {}

        # Determine which domains were targeted for this sample.
        # If __target_domains__ is present in reward_val (set by
        # _reward_func_single), use it. Otherwise, fall back to all
        # configured domains (backward compatible).
        if isinstance(reward_val, dict) and "__target_domains__" in reward_val:
            target_domains = reward_val["__target_domains__"]
        else:
            target_domains = all_configured_domains

        for domain in target_domains:
            if domain not in sample.mopd_teacher_log_probs:
                logger.warning(
                    f"MOPD: Teacher data for domain '{domain}' is missing for a sample "
                    f"(was queried but extraction failed or request failed). "
                    f"Filling with -inf log-probs (zero KL contribution)."
                )
                sample.mopd_teacher_log_probs[domain] = torch.full(
                    (response_length,), float('-inf'), dtype=torch.float32
                )
            if mopd_distill_type == "top_k":
                if sample.mopd_teacher_topk_logits is None:
                    sample.mopd_teacher_topk_logits = {}
                if sample.mopd_teacher_topk_indices is None:
                    sample.mopd_teacher_topk_indices = {}
                if domain not in sample.mopd_teacher_topk_logits:
                    topk_k = getattr(args, "mopd_topk_k", 1024)
                    NEG_INF = float("-inf")
                    sample.mopd_teacher_topk_logits[domain] = [
                        [NEG_INF] * topk_k for _ in range(response_length)
                    ]
                    sample.mopd_teacher_topk_indices[domain] = [
                        [0] * topk_k for _ in range(response_length)
                    ]


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process MOPD teacher responses and extract per-domain teacher data.

    This is the standalone post_process_rewards for pure MOPD distillation (alpha=0)
    where no task reward is needed. It reads teacher responses from sample.reward
    (which should be a dict mapping domain -> SGLang response JSON as returned by
    reward_func) and populates sample.mopd_teacher_* fields.

    For combined MOPD + task rewards (alpha>0), use combined_post_process_rewards instead.

    Returns:
        Tuple of (scalar_rewards, scalar_rewards) for GRPO/PPO compatibility.
        All rewards are 0.0 since the learning signal comes from distillation.
    """
    _extract_teacher_data_from_responses(args, samples)

    # Reset sample.reward to scalar 0.0 — the SGLang response dict is no longer
    # needed and leaving it as a dict causes downstream code (e.g., metrics
    # logging, reward aggregation) to break since they expect numeric values.
    for sample in samples:
        sample.reward = 0.0

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards


# ---------------------------------------------------------------------------
# Combined reward functions (MOPD teacher data + task rewards)
# ---------------------------------------------------------------------------
# When --mopd-alpha > 0, MOPD combines distillation advantages with task
# rewards. This requires both collecting teacher data from SGLang AND getting
# task rewards from the standard reward model. Since custom_rm_path replaces
# the standard reward model, we provide combined wrappers that invoke both.
#
# The key idea:
#   - combined_reward_func: fetches teacher data from SGLang, stores it in
#     sample.metadata["_mopd_teacher_responses"], then calls the standard
#     reward model (rm_type-based) to get task rewards.
#   - combined_post_process_rewards: extracts teacher log-probs from
#     sample.metadata["_mopd_teacher_responses"], then applies standard
#     reward post-processing (GRPO normalization, etc.) to the task rewards
#     stored in sample.reward.
# ---------------------------------------------------------------------------

_MOPD_TEACHER_RESPONSES_KEY = "_mopd_teacher_responses"


async def combined_reward_func(args, sample_or_samples, **kwargs):
    """Combined reward function: MOPD teacher data collection + task rewards.

    This function:
    1. Fetches MOPD teacher data from SGLang servers (via reward_func).
    2. Stores the teacher responses in sample.metadata for later extraction.
    3. Calls the standard reward model (rm_type-based) to get task rewards.

    Returns the task reward (float) for single sample, or list of task rewards
    for batch mode. The MOPD teacher data is stored in sample metadata.

    NOTE: This function temporarily sets args.custom_rm_path to None to bypass
    the custom RM and call the standard rm_type-based reward model. This is
    safe because reward evaluation is sequential per rollout batch within a
    single worker process. However, if concurrent reward evaluation is ever
    introduced, this would need to be refactored to avoid data races.
    """
    from slime.rollout.rm_hub import async_rm, batched_async_rm

    # Step 1: Fetch MOPD teacher data
    if isinstance(sample_or_samples, list):
        mopd_results = await reward_func(args, sample_or_samples, **kwargs)

        # Store teacher responses in metadata, then get task rewards
        # Temporarily save custom_rm_path so we can bypass it for task rewards
        original_custom_rm_path = args.custom_rm_path
        args.custom_rm_path = None
        try:
            task_rewards = await batched_async_rm(args, sample_or_samples, **kwargs)
        finally:
            args.custom_rm_path = original_custom_rm_path

        # Store MOPD teacher responses in sample metadata
        for sample, mopd_result in zip(sample_or_samples, mopd_results):
            if isinstance(sample.metadata, dict):
                sample.metadata[_MOPD_TEACHER_RESPONSES_KEY] = mopd_result
            else:
                sample.metadata = {_MOPD_TEACHER_RESPONSES_KEY: mopd_result}

        return task_rewards
    else:
        sample = sample_or_samples
        mopd_result = await reward_func(args, sample, **kwargs)

        # Store teacher response in metadata
        if isinstance(sample.metadata, dict):
            sample.metadata[_MOPD_TEACHER_RESPONSES_KEY] = mopd_result
        else:
            sample.metadata = {_MOPD_TEACHER_RESPONSES_KEY: mopd_result}

        # Get task reward (bypass custom_rm_path to use rm_type)
        original_custom_rm_path = args.custom_rm_path
        args.custom_rm_path = None
        try:
            task_reward = await async_rm(args, sample, **kwargs)
        finally:
            args.custom_rm_path = original_custom_rm_path

        return task_reward


def combined_post_process_rewards(args, samples: list[Sample], **kwargs):
    """Combined post-processing: extract MOPD teacher data + standard reward normalization.

    This function:
    1. Extracts MOPD teacher log-probs from sample.metadata["_mopd_teacher_responses"]
       (stored by combined_reward_func), populates sample.mopd_teacher_* fields.
    2. Applies standard reward post-processing (GRPO normalization, etc.) to the
       task rewards stored in sample.reward.

    Returns:
        Tuple of (raw_rewards, processed_rewards) for GRPO/PPO compatibility.
    """
    # Step 1: Extract MOPD teacher data from metadata
    # Temporarily swap sample.reward to contain the teacher responses so that
    # _extract_teacher_data_from_responses can read them via get_reward_value.
    # Save original task rewards first.
    original_rewards = []
    for sample in samples:
        original_rewards.append(sample.reward)
        teacher_responses = None
        if isinstance(sample.metadata, dict):
            teacher_responses = sample.metadata.get(_MOPD_TEACHER_RESPONSES_KEY)
        sample.reward = teacher_responses  # Temporarily set for extraction

    # Extract teacher data (populates sample.mopd_teacher_log_probs, etc.)
    _extract_teacher_data_from_responses(args, samples)

    # Clean up temporary metadata and restore task rewards
    for sample, original_reward in zip(samples, original_rewards):
        if isinstance(sample.metadata, dict):
            sample.metadata.pop(_MOPD_TEACHER_RESPONSES_KEY, None)
        sample.reward = original_reward

    # Step 2: Apply standard reward post-processing
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        return raw_rewards, rewards.flatten().tolist()

    return raw_rewards, raw_rewards