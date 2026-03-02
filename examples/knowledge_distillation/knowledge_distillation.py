import asyncio
import json
import logging
import os

import aiohttp
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.async_utils import run
from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)

KD_TOP_K = int(os.environ.get("KD_TOP_K", "8"))
KD_SAVE_PATH = os.environ.get("KD_SAVE_PATH")
TOKENIZER = None


def _get_tokenizer(args):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    return TOKENIZER


def _build_sampling_params(args):
    return {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        "max_new_tokens": args.rollout_max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


async def _generate_sample(args, sample, sampling_params, tokenizer, session, semaphore):
    assert isinstance(sample.prompt, str), "KD rollout requires string prompts. Enable --apply-chat-template."

    prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if KD_TOP_K > 0:
        payload["top_logprobs_num"] = KD_TOP_K

    async with semaphore:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            output = await resp.json()

    meta = output["meta_info"]
    generated = meta["output_token_logprobs"]
    response_tokens = [int(item[1]) for item in generated]

    sample.tokens = prompt_ids + response_tokens
    sample.response = output.get("text", "")
    sample.response_length = len(response_tokens)
    sample.teacher_log_probs = [float(item[0]) for item in generated]
    sample.reward = 0.0

    if KD_TOP_K > 0:
        top_logprobs = meta["output_top_logprobs"]
        sample.train_metadata = {
            "teacher_top_k_ids": [[int(e[1]) for e in pos[:KD_TOP_K]] for pos in top_logprobs],
            "teacher_top_k_logprobs": [[float(e[0]) for e in pos[:KD_TOP_K]] for pos in top_logprobs],
        }

    return sample


def _save_to_jsonl(samples, save_path, rollout_id):
    save_path = save_path.format(rollout_id=rollout_id)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    with open(save_path, "w") as f:
        metadata = {"__metadata__": True, "distillation_type": "top_k" if KD_TOP_K > 0 else "sampled_kl"}
        if KD_TOP_K > 0:
            metadata["top_k"] = KD_TOP_K
        f.write(json.dumps(metadata) + "\n")

        for group in samples:
            for s in group:
                record = {
                    "prompt": s.prompt,
                    "tokens": s.tokens,
                    "response": s.response,
                    "response_length": s.response_length,
                    "teacher_log_probs": s.teacher_log_probs,
                }
                if KD_TOP_K > 0 and s.train_metadata:
                    record["teacher_top_k_ids"] = s.train_metadata["teacher_top_k_ids"]
                    record["teacher_top_k_logprobs"] = s.train_metadata["teacher_top_k_logprobs"]
                f.write(json.dumps(record) + "\n")

    logger.info(f"Saved {sum(len(g) for g in samples)} samples to {save_path}")


async def _generate_rollout_async(args, data_source):
    assert args.rollout_global_dataset

    tokenizer = _get_tokenizer(args)
    samples = data_source.get_samples(args.rollout_batch_size)
    sampling_params = _build_sampling_params(args)
    semaphore = asyncio.Semaphore(max(getattr(args, "sglang_server_concurrency", 64), 1))

    async with aiohttp.ClientSession() as session:
        generated_groups = await asyncio.gather(*(
            asyncio.gather(*(_generate_sample(args, s, sampling_params, tokenizer, session, semaphore) for s in group))
            for group in samples
        ))

    first = generated_groups[0][0]
    logger.info(f"KD rollout: prompt={first.prompt[:80]!r}, response={first.response[:80]!r}, len={first.response_length}")

    token_count = sum(s.response_length for g in generated_groups for s in g)
    return RolloutFnTrainOutput(samples=generated_groups, metrics={"kd/token_count": token_count})


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        return RolloutFnEvalOutput(data={})
    assert args.rm_url, "--rm-url must be set for KD rollout."

    result = run(_generate_rollout_async(args, data_source))

    # Store top-k data for loss function access
    if KD_TOP_K > 0:
        from examples.knowledge_distillation.kd_loss import store_topk_data
        store_topk_data(result.samples)

    if KD_SAVE_PATH:
        _save_to_jsonl(result.samples, KD_SAVE_PATH, rollout_id)
    return result
