import asyncio
import logging

import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _get_teacher_semaphore(args):
    loop = asyncio.get_running_loop()
    loop_key = id(loop)
    semaphores_by_loop = getattr(args, "_opd_teacher_semaphores", None)
    if semaphores_by_loop is None:
        semaphores_by_loop = {}
        args._opd_teacher_semaphores = semaphores_by_loop

    concurrency = int(getattr(args, "opd_teacher_concurrency", 0))
    if concurrency <= 0:
        return None

    semaphore = semaphores_by_loop.get(loop_key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(concurrency)
        semaphores_by_loop[loop_key] = semaphore
    return semaphore


def _teacher_payload_summary(payload):
    if "input_ids" in payload:
        return f"input_tokens={len(payload['input_ids'])}"
    if "text" in payload:
        return f"text_chars={len(payload['text'])}"
    return "payload_size=unknown"


async def _post_teacher_once(session, teacher_url, payload):
    async with session.post(teacher_url, json=payload) as resp:
        if resp.status >= 400:
            body = (await resp.text())[:500]
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=f"{resp.reason}; body={body}",
                headers=resp.headers,
            )
        return await resp.json()


async def _post_teacher_json(args, teacher_url, payload, sample, warning_prefix):
    retries = max(0, int(getattr(args, "opd_teacher_retries", 2)))
    timeout = aiohttp.ClientTimeout(total=float(getattr(args, "opd_teacher_timeout", 300.0)))
    semaphore = _get_teacher_semaphore(args)
    last_exc = None

    for attempt in range(retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if semaphore is None:
                    return await _post_teacher_once(session, teacher_url, payload)
                async with semaphore:
                    return await _post_teacher_once(session, teacher_url, payload)
        except (TimeoutError, asyncio.TimeoutError, aiohttp.ClientError) as exc:
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(min(0.2 * (2**attempt), 2.0))
                continue

    logger.warning(
        "%s failed after %s attempt(s); using student rollout logprobs as teacher logprobs "
        "for sample index=%s (%s): %s: %s",
        warning_prefix,
        retries + 1,
        getattr(sample, "index", None),
        _teacher_payload_summary(payload),
        type(last_exc).__name__,
        last_exc,
    )
    return {"_opd_teacher_fallback": True, "_opd_teacher_fallback_reason": type(last_exc).__name__}


def _use_student_log_probs_for_teacher(sample: Sample, reason: str) -> None:
    resp_len = sample.response_length
    if resp_len == 0:
        sample.teacher_log_probs = torch.empty(0, dtype=torch.float32)
    else:
        rollout_lps = sample.rollout_log_probs
        if rollout_lps is None:
            values = torch.zeros(resp_len, dtype=torch.float32)
        else:
            values = torch.as_tensor(rollout_lps, dtype=torch.float32).flatten()
            if values.numel() >= resp_len:
                values = values[-resp_len:]
            else:
                values = torch.cat([torch.zeros(resp_len - values.numel(), dtype=torch.float32), values])
        sample.teacher_log_probs = values

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["opd_teacher_fallback"] = True
    sample.metadata["opd_teacher_fallback_reason"] = reason


def _load_student_tokenizer(args):
    if not hasattr(args, "_opd_student_tok"):
        from transformers import AutoTokenizer

        args._opd_student_tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    return args._opd_student_tok


def _get_opd_mask_token_sequences(args) -> list[list[int]]:
    token_strings = getattr(args, "opd_mask_teacher_logprob_tokens", None)
    if not token_strings:
        return []

    cached = getattr(args, "_opd_mask_teacher_logprob_token_ids", None)
    if cached is not None:
        return cached

    tokenizer = _load_student_tokenizer(args)
    sequences = []
    for token_string in token_strings:
        token_ids = tokenizer.encode(token_string, add_special_tokens=False)
        if not token_ids:
            logger.warning("Skipping empty OPD teacher-logprob mask token string: %r", token_string)
            continue
        sequences.append(token_ids)

    args._opd_mask_teacher_logprob_token_ids = sequences
    return sequences


def _iter_matching_token_positions(tokens: list[int], sequences: list[list[int]]):
    for sequence in sequences:
        seq_len = len(sequence)
        if seq_len == 0 or seq_len > len(tokens):
            continue
        for start in range(0, len(tokens) - seq_len + 1):
            if tokens[start : start + seq_len] == sequence:
                yield from range(start, start + seq_len)


def _mask_teacher_logprobs_with_student(args, sample: Sample, teacher_log_probs):
    sequences = _get_opd_mask_token_sequences(args)
    if not sequences or sample.response_length == 0:
        return teacher_log_probs

    student_log_probs = sample.rollout_log_probs
    if not student_log_probs:
        logger.warning(
            "Cannot mask OPD teacher log-probs for sample index=%s because rollout_log_probs is missing.",
            getattr(sample, "index", None),
        )
        return teacher_log_probs

    resp_len = sample.response_length
    prompt_len = len(sample.tokens) - resp_len
    response_tokens = sample.tokens[prompt_len:]
    student_log_probs = list(student_log_probs)[-resp_len:]
    masked_positions = sorted(set(_iter_matching_token_positions(response_tokens, sequences)))
    if not masked_positions:
        return teacher_log_probs

    if isinstance(teacher_log_probs, torch.Tensor):
        masked = teacher_log_probs.clone()
        for pos in masked_positions:
            if pos < masked.numel() and pos < len(student_log_probs):
                masked[pos] = float(student_log_probs[pos])
        return masked

    masked = list(teacher_log_probs)
    for pos in masked_positions:
        if pos < len(masked) and pos < len(student_log_probs):
            masked[pos] = float(student_log_probs[pos])
    return masked


async def reward_func(args, sample, **kwargs):
    if kwargs.get("evaluation"):
        from slime.rollout.rm_hub import async_rm as default_async_rm

        saved = args.custom_rm_path
        args.custom_rm_path = None
        try:
            return await default_async_rm(args, sample, **kwargs)
        finally:
            args.custom_rm_path = saved

    payload = {
        # "text": sample.prompt + sample.response,
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

    return await _post_teacher_json(args, args.rm_url, payload, sample, "OPD teacher request")


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
    raw_rewards = [
        (
            sample.reward
            if isinstance(sample.reward, dict) and sample.reward.get("_opd_teacher_fallback")
            else sample.get_reward_value(args)
        )
        for sample in samples
    ]
    response_lengths = [sample.response_length for sample in samples]

    for sample, reward in zip(samples, raw_rewards, strict=False):
        if isinstance(reward, dict) and reward.get("_opd_teacher_fallback"):
            _use_student_log_probs_for_teacher(sample, reward.get("_opd_teacher_fallback_reason", "teacher_failed"))

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
        if not (isinstance(reward, dict) and reward.get("_opd_teacher_fallback"))
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(
            teacher_log_probs,
            [
                length
                for reward, length in zip(raw_rewards, response_lengths, strict=False)
                if not (isinstance(reward, dict) and reward.get("_opd_teacher_fallback"))
            ],
            strict=False,
        )
    ]

    samples_with_teacher = [
        sample
        for sample, reward in zip(samples, raw_rewards, strict=False)
        if not (isinstance(reward, dict) and reward.get("_opd_teacher_fallback"))
    ]
    for sample, t_log_probs in zip(samples_with_teacher, teacher_log_probs, strict=False):
        t_log_probs = _mask_teacher_logprobs_with_student(args, sample, t_log_probs)
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards


def _load_cross_vocab_tokenizers(args):
    """Lazy-load teacher and student tokenizers, caching them on args."""
    if hasattr(args, "_cross_vocab_student_tok"):
        return

    teacher_path = getattr(args, "teacher_tokenizer_path", None)
    if not teacher_path:
        raise ValueError(
            "--teacher-tokenizer-path is required for cross-vocabulary OPD. "
            "It must point to the teacher model tokenizer."
        )

    from transformers import AutoTokenizer

    args._cross_vocab_student_tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    args._cross_vocab_teacher_tok = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)


def _get_nested_metadata(metadata: dict, key: str | None):
    if not key:
        return None
    current = metadata
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _render_teacher_prompt(args, sample: Sample, teacher_tok):
    """Render the rollout prompt with the teacher tokenizer when messages are available."""
    metadata = sample.metadata or {}
    messages_key = getattr(args, "opd_prompt_messages_key", None)
    prompt_messages = _get_nested_metadata(metadata, messages_key)
    if prompt_messages is None and isinstance(sample.prompt, list):
        prompt_messages = sample.prompt

    if prompt_messages is not None:
        if isinstance(prompt_messages, str):
            prompt_text = prompt_messages
            prompt_ids = teacher_tok.encode(prompt_text, add_special_tokens=False)
            return prompt_text, prompt_ids
        if not isinstance(prompt_messages, list):
            raise TypeError(
                f"OPD prompt messages from key {messages_key!r} must be a list of chat messages or a string, "
                f"got {type(prompt_messages).__name__}."
            )

        tools = metadata.get("tools")
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools is not None:
            template_kwargs["tools"] = tools
        try:
            prompt_text = teacher_tok.apply_chat_template(prompt_messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("tools", None)
            prompt_text = teacher_tok.apply_chat_template(prompt_messages, **template_kwargs)

        token_kwargs = dict(template_kwargs)
        token_kwargs["tokenize"] = True
        try:
            prompt_ids = teacher_tok.apply_chat_template(prompt_messages, **token_kwargs)
        except TypeError:
            token_kwargs.pop("tools", None)
            prompt_ids = teacher_tok.apply_chat_template(prompt_messages, **token_kwargs)
        return prompt_text, prompt_ids

    if not isinstance(sample.prompt, str):
        raise TypeError(
            "Cross-vocabulary OPD requires a string prompt or chat messages in sample.prompt/metadata. "
            f"Got sample.prompt={type(sample.prompt).__name__}."
        )
    return sample.prompt, teacher_tok.encode(sample.prompt, add_special_tokens=False)


def _decode_token_texts(tokenizer, token_ids: list[int]) -> list[str]:
    return [
        tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False) or ""
        for token_id in token_ids
    ]


def _align_common_tokens_1to1(teacher_texts, teacher_lps, student_texts, student_rollout_lps):
    """Align teacher logprobs to student positions using exact 1:1 decoded-token matches.

    Positions without an exact match keep the student rollout logprob. The downstream
    reverse-KL delta is therefore zero for tokenizer-boundary mismatches.
    """
    aligned = list(student_rollout_lps)

    i = j = 0
    teacher_prefix = ""
    student_prefix = ""
    num_matched = 0

    while i < len(teacher_texts) and j < len(student_texts):
        teacher_text = teacher_texts[i] or ""
        student_text = student_texts[j] or ""

        if teacher_prefix == student_prefix and teacher_text == student_text:
            aligned[j] = teacher_lps[i]
            num_matched += 1
            teacher_prefix += teacher_text
            student_prefix += student_text
            i += 1
            j += 1
        elif len(teacher_prefix) > len(student_prefix):
            student_prefix += student_text
            j += 1
        elif len(teacher_prefix) < len(student_prefix):
            teacher_prefix += teacher_text
            i += 1
        else:
            teacher_prefix += teacher_text
            student_prefix += student_text
            i += 1
            j += 1

    return aligned, num_matched


async def reward_func_cross_vocab(args, sample: Sample, **kwargs):
    """Ask an SGLang teacher for logprobs after rendering the prompt with teacher chat template."""
    if kwargs.get("evaluation"):
        from slime.rollout.rm_hub import async_rm as default_async_rm

        saved = args.custom_rm_path
        args.custom_rm_path = None
        try:
            return await default_async_rm(args, sample, **kwargs)
        finally:
            args.custom_rm_path = saved

    _load_cross_vocab_tokenizers(args)
    teacher_tok = args._cross_vocab_teacher_tok
    teacher_prompt, teacher_prompt_ids = _render_teacher_prompt(args, sample, teacher_tok)

    payload = {
        "text": teacher_prompt + sample.response,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    result = await _post_teacher_json(args, args.rm_url, payload, sample, "Cross-vocab OPD teacher request")
    if isinstance(result, dict) and result.get("_opd_teacher_fallback"):
        return result

    result["_cross_vocab_meta"] = {"teacher_prompt_len": len(teacher_prompt_ids)}
    return result


def post_process_rewards_cross_vocab(args, samples: list[Sample], **kwargs):
    """Store teacher logprobs on student response positions that share a 1:1 token boundary."""
    raw_rewards = [
        (
            sample.reward
            if isinstance(sample.reward, dict) and sample.reward.get("_opd_teacher_fallback")
            else sample.get_reward_value(args)
        )
        for sample in samples
    ]
    has_teacher_rewards = any(
        not (isinstance(reward, dict) and reward.get("_opd_teacher_fallback")) for reward in raw_rewards
    )
    if has_teacher_rewards:
        _load_cross_vocab_tokenizers(args)
        student_tok = args._cross_vocab_student_tok
        teacher_tok = args._cross_vocab_teacher_tok

    for sample, reward in zip(samples, raw_rewards, strict=True):
        resp_len = sample.response_length
        if resp_len == 0:
            sample.teacher_log_probs = torch.empty(0, dtype=torch.float32)
            continue

        if isinstance(reward, dict) and reward.get("_opd_teacher_fallback"):
            _use_student_log_probs_for_teacher(sample, reward.get("_opd_teacher_fallback_reason", "teacher_failed"))
            continue

        all_info = reward["meta_info"]["input_token_logprobs"]
        teacher_prompt_len = reward["_cross_vocab_meta"]["teacher_prompt_len"]

        teacher_resp_info = all_info[teacher_prompt_len:]
        teacher_lps = [item[0] for item in teacher_resp_info]
        teacher_ids = [item[1] for item in teacher_resp_info]

        prompt_len = len(sample.tokens) - resp_len
        student_resp_ids = sample.tokens[prompt_len:]
        student_texts = _decode_token_texts(student_tok, student_resp_ids)
        teacher_texts = _decode_token_texts(teacher_tok, teacher_ids)
        student_rollout_lps = list(sample.rollout_log_probs or [0.0] * resp_len)[-resp_len:]

        aligned_lps, num_matched = _align_common_tokens_1to1(
            teacher_texts, teacher_lps, student_texts, student_rollout_lps
        )
        aligned_lps = _mask_teacher_logprobs_with_student(args, sample, aligned_lps)

        sample.teacher_log_probs = torch.tensor(aligned_lps, dtype=torch.float32)
        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["cross_vocab_token_overlap"] = num_matched / resp_len if resp_len > 0 else 0.0

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
