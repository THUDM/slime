#!/usr/bin/env python3
"""Offline eval backfill: run missing evals on existing checkpoints and log to wandb.

Usage:
    python eval_backfill.py \
        --experiment-dir /path/to/experiment \
        --wandb-run-id <run_id> \
        --wandb-project slime-multidomain-v1 \
        --sglang-url http://localhost:30000 \
        --eval-data bfcl_v3_eval:/path/to/bfcl_v3_eval.normalized.jsonl \
        --eval-data mmlu_pro_eval:/path/to/mmlu_pro_eval.normalized.jsonl \
        --rollout-id <current checkpoint rollout_id>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

_tokenizer = None


def get_tokenizer(model_path: str):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info(f"Loaded tokenizer from {model_path}")
    return _tokenizer


def load_reward_func():
    import importlib.util

    reward_path = SCRIPT_DIR / "reward_multidomain_v1.py"
    if not reward_path.exists():
        reward_path = SCRIPT_DIR.parent / "multidomain_v1" / "reward_multidomain_v1.py"
    spec = importlib.util.spec_from_file_location("reward_multidomain_v1", reward_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.reward_func


def load_eval_data(path: str) -> list[dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(_normalize_eval_sample_for_backfill(json.loads(line)))
    return samples


def _normalize_eval_sample_for_backfill(sample: dict[str, Any]) -> dict[str, Any]:
    """Patch known stale eval metadata so old experiment caches can be rescored correctly."""

    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return sample

    dataset_name = str(metadata.get("dataset_name", "")).strip().lower()
    if dataset_name in {"bfcl_v3", "bfcl_v3_multi_turn_base"}:
        metadata = dict(metadata)
        metadata["reward_type"] = "tool_call_soft"
        sample = dict(sample)
        sample["metadata"] = metadata
    return sample


def apply_chat_template(tokenizer, sample: dict[str, Any]) -> str:
    """Apply tokenizer chat template to get the full prompt text (with tools baked in)."""
    messages = sample.get("prompt", [])
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    tools = sample.get("tools") or None
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def filter_long_prompts(
    tokenizer,
    eval_samples: list[dict[str, Any]],
    prompt_texts: list[str],
    max_prompt_tokens: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter out samples whose prompt exceeds max_prompt_tokens, matching training framework behavior."""
    input_ids_list = tokenizer(prompt_texts, add_special_tokens=False)["input_ids"]
    filtered_samples = []
    filtered_texts = []
    skipped = 0
    for sample, text, input_ids in zip(eval_samples, prompt_texts, input_ids_list):
        if len(input_ids) <= max_prompt_tokens:
            filtered_samples.append(sample)
            filtered_texts.append(text)
        else:
            skipped += 1
    if skipped:
        logger.info(f"  Filtered {skipped} samples exceeding {max_prompt_tokens} prompt tokens")
    return filtered_samples, filtered_texts


async def _async_generate_one(
    session,
    base_url: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    semaphore,
) -> str:
    import aiohttp

    payload = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    f"{base_url}/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                    result = await resp.json()
                    return result.get("text", "")
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"  Failed after 3 attempts: {e}")
                    return ""
                await asyncio.sleep(2)
    return ""


async def _async_generate_batch(
    prompt_texts: list[str],
    sglang_url: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    concurrency: int,
) -> list[str]:
    import aiohttp

    base_url = sglang_url.rstrip("/")
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            _async_generate_one(session, base_url, text, max_tokens, temperature, top_p, semaphore)
            for text in prompt_texts
        ]
        return await asyncio.gather(*tasks)


def generate_batch(
    sglang_url: str,
    prompt_texts: list[str],
    max_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 1.0,
    batch_size: int = 64,
) -> list[str]:
    """Generate responses using sglang /generate endpoint with async concurrency."""
    logger.info(f"  Generating {len(prompt_texts)} samples (concurrency={batch_size})")
    return asyncio.run(
        _async_generate_batch(prompt_texts, sglang_url, max_tokens, temperature, top_p, concurrency=batch_size)
    )


class MockSample:
    """Minimal sample object compatible with the reward function."""

    class Status:
        TRUNCATED = "truncated"
        COMPLETED = "completed"

    def __init__(self, response: str, metadata: dict, prompt: str = "", status: str = "completed"):
        self.response = response
        self.metadata = metadata
        self.prompt = prompt
        self.status = status
        self.reward = None
        self.effective_response_length = len(response)


async def compute_rewards(
    reward_func,
    eval_samples: list[dict[str, Any]],
    responses: list[str],
) -> list[float]:
    rewards = []
    for sample_data, response in zip(eval_samples, responses):
        metadata = sample_data.get("metadata", {})
        prompt = sample_data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = json.dumps(prompt)
        mock = MockSample(response=response, metadata=metadata, prompt=prompt)
        try:
            reward = await reward_func(None, mock)
        except Exception as e:
            logger.warning(f"  Reward computation failed: {e}")
            reward = 0.0
        rewards.append(float(reward))
    return rewards


def compute_eval_metrics(
    eval_name: str,
    eval_samples: list[dict[str, Any]],
    rewards: list[float],
    responses: list[str],
) -> dict[str, float]:
    """Compute eval metrics matching the training framework's format."""
    metrics = {}
    n = len(rewards)
    if n == 0:
        return metrics

    metrics[f"eval/{eval_name}"] = sum(rewards) / n

    # Per-dataset metrics
    from collections import defaultdict

    by_dataset = defaultdict(list)
    by_domain = defaultdict(list)
    for sample_data, reward, response in zip(eval_samples, rewards, responses):
        meta = sample_data.get("metadata", {})
        ds_name = meta.get("dataset_name", eval_name)
        domain = meta.get("domain", "unknown")
        by_dataset[ds_name].append((reward, response))
        by_domain[domain].append((reward, response))

    for ds_name, items in by_dataset.items():
        ds_rewards = [r for r, _ in items]
        ds_responses = [resp for _, resp in items]
        metrics[f"eval_by_source/{ds_name}/count"] = len(items)
        metrics[f"eval_by_source/{ds_name}/score"] = sum(ds_rewards) / len(ds_rewards)
        lengths = [len(r) for r in ds_responses]
        if lengths:
            metrics[f"eval_by_source/{ds_name}/response_len/mean"] = sum(lengths) / len(lengths)

    for domain_name, items in by_domain.items():
        dom_rewards = [r for r, _ in items]
        metrics[f"eval_by_domain/{domain_name}/count"] = len(items)
        metrics[f"eval_by_domain/{domain_name}/score"] = sum(dom_rewards) / len(dom_rewards)

    return metrics


def log_to_wandb(
    wandb_run_id: str,
    wandb_project: str,
    wandb_host: str,
    wandb_key: str,
    wandb_group: str,
    step: int,
    metrics: dict[str, float],
):
    """Resume wandb run and log metrics at the given step."""
    if wandb_key:
        wandb.login(key=wandb_key, host=wandb_host)

    run = wandb.init(
        id=wandb_run_id,
        project=wandb_project,
        resume="allow",
        reinit=True,
        settings=wandb.Settings(mode="shared"),
    )
    metrics["eval/step"] = step
    wandb.log(metrics, step=step)
    wandb.finish()
    logger.info(f"  Logged {len(metrics)} metrics to wandb run {wandb_run_id} at step {step}")


def wait_for_sglang(url: str, timeout: int = 300):
    """Wait for sglang server to be ready."""
    base_url = url.rstrip("/")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                logger.info("sglang server is ready.")
                return True
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"sglang server not ready after {timeout}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Eval backfill for multidomain_v2 checkpoints")
    parser.add_argument("--sglang-url", type=str, default="http://localhost:30000")
    parser.add_argument("--eval-data", action="append", required=True, help="name:path pairs for eval datasets")
    parser.add_argument("--rollout-id", type=int, required=True, help="Rollout ID (= checkpoint step) for wandb logging")
    parser.add_argument("--wandb-run-id", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, default="slime-multidomain-v2")
    parser.add_argument("--wandb-host", type=str, default="")
    parser.add_argument("--wandb-key", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--model-path", type=str, default=None, help="HF model path (for tokenizer chat template)")
    parser.add_argument("--max-context-len", type=int, default=32768, help="Max context length for sglang server (prompt tokens only, response excluded)")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    reward_func = load_reward_func()

    # Load tokenizer for applying chat template (bakes tools into prompt text)
    model_path = args.model_path
    if not model_path:
        # Fallback: try to detect from sglang /v1/models
        try:
            resp = requests.get(f"{args.sglang_url.rstrip('/')}/v1/models", timeout=10)
            model_id = resp.json()["data"][0]["id"]
            if Path(model_id).exists():
                model_path = model_id
        except Exception:
            pass
    if not model_path:
        raise RuntimeError("--model-path required (HF checkpoint path for tokenizer)")
    tokenizer = get_tokenizer(model_path)

    # Parse eval datasets
    eval_datasets = {}
    for spec in args.eval_data:
        name, path = spec.split(":", 1)
        eval_datasets[name] = path

    # Wait for sglang
    wait_for_sglang(args.sglang_url)

    all_metrics = {}
    for eval_name, eval_path in eval_datasets.items():
        logger.info(f"Running eval: {eval_name} from {eval_path}")
        eval_samples = load_eval_data(eval_path)
        logger.info(f"  Loaded {len(eval_samples)} samples")

        # Apply chat template: bakes tools + messages into a single prompt string
        prompt_texts = []
        for s in eval_samples:
            prompt_texts.append(apply_chat_template(tokenizer, s))

        # Filter out prompts that exceed context length (same as training framework)
        eval_samples, prompt_texts = filter_long_prompts(
            tokenizer, eval_samples, prompt_texts, args.max_context_len,
        )
        if not eval_samples:
            logger.warning(f"  All samples filtered out for {eval_name}, skipping")
            continue

        responses = generate_batch(
            sglang_url=args.sglang_url,
            prompt_texts=prompt_texts,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )

        rewards = asyncio.run(compute_rewards(reward_func, eval_samples, responses))
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info(f"  {eval_name}: avg reward = {avg_reward:.4f}")

        metrics = compute_eval_metrics(eval_name, eval_samples, rewards, responses)
        all_metrics.update(metrics)

    # Log to wandb
    if all_metrics:
        log_to_wandb(
            wandb_run_id=args.wandb_run_id,
            wandb_project=args.wandb_project,
            wandb_host=args.wandb_host,
            wandb_key=args.wandb_key,
            wandb_group=args.wandb_group,
            step=args.rollout_id,
            metrics=all_metrics,
        )

    logger.info("Eval backfill complete.")


if __name__ == "__main__":
    main()
