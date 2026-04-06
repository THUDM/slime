#!/usr/bin/env python3
"""Offline eval backfill for existing checkpoints.

This lightweight recovery version keeps the normal inference + logging path used by
MOPD step-0 teacher evaluation. Migration helpers are intentionally unsupported here.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR

_tokenizer = None


def get_tokenizer(model_path: str):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Loaded tokenizer from %s", model_path)
    return _tokenizer


def load_reward_func(module_path: str = "multidomain_shared.reward_func"):
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    parts = module_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"--reward-module must be 'module.attr', got: {module_path}")
    mod = importlib.import_module(parts[0])
    return getattr(mod, parts[1])


def _normalize_eval_sample_for_backfill(sample: dict[str, Any]) -> dict[str, Any]:
    return sample


def load_eval_data(path: str) -> list[dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(_normalize_eval_sample_for_backfill(json.loads(line)))
    return samples


def _load_bfcl_runner():
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    from bfcl_official_runner import (
        DEFAULT_BFCL_MODEL_NAME,
        generate_bfcl_multi_turn_outputs,
        run_bfcl_official_eval,
        summary_to_metrics,
    )

    return {
        "DEFAULT_BFCL_MODEL_NAME": DEFAULT_BFCL_MODEL_NAME,
        "generate_bfcl_multi_turn_outputs": generate_bfcl_multi_turn_outputs,
        "run_bfcl_official_eval": run_bfcl_official_eval,
        "summary_to_metrics": summary_to_metrics,
    }


def _is_bfcl_official_eval(eval_samples: list[dict[str, Any]]) -> bool:
    if not eval_samples:
        return False
    metadata = eval_samples[0].get("metadata", {}) or {}
    dataset_name = str(metadata.get("dataset_name", "")).strip()
    reward_type = str(metadata.get("reward_type", "")).strip()
    return dataset_name in {"bfcl_v3", "bfcl_v3_multi_turn_base"} or reward_type == "bfcl_official"


def _is_bfcl_multi_turn_eval(eval_samples: list[dict[str, Any]]) -> bool:
    if not eval_samples:
        return False
    metadata = eval_samples[0].get("metadata", {}) or {}
    return str(metadata.get("dataset_name", "")).strip() == "bfcl_v3_multi_turn_base"


def apply_chat_template(tokenizer, sample: dict[str, Any]) -> str:
    messages = sample.get("prompt", [])
    if not messages:
        question = str(sample.get("question") or "").strip()
        if question:
            messages = [{"role": "user", "content": question}]
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    tools = sample.get("tools") or None
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def filter_long_prompts(tokenizer, eval_samples, prompt_texts, max_prompt_tokens):
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
        logger.info("  Filtered %d samples exceeding %d prompt tokens", skipped, max_prompt_tokens)
    return filtered_samples, filtered_texts


async def _async_generate_one(session, base_url, prompt_text, max_tokens, temperature, top_p, semaphore):
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
            except Exception as exc:  # pragma: no cover - best effort retry path
                if attempt == 2:
                    logger.warning("  Failed after 3 attempts: %s", exc)
                    return ""
                await asyncio.sleep(2)
    return ""


async def _async_generate_batch(prompt_texts, sglang_url, max_tokens, temperature, top_p, concurrency):
    import aiohttp

    base_url = sglang_url.rstrip("/")
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            _async_generate_one(session, base_url, text, max_tokens, temperature, top_p, semaphore)
            for text in prompt_texts
        ]
        return await asyncio.gather(*tasks)


def generate_batch(sglang_url, prompt_texts, max_tokens=8192, temperature=0.7, top_p=1.0, batch_size=64):
    logger.info("  Generating %d samples (concurrency=%d)", len(prompt_texts), batch_size)
    return asyncio.run(
        _async_generate_batch(prompt_texts, sglang_url, max_tokens, temperature, top_p, concurrency=batch_size)
    )


def generate_one(sglang_url, prompt_text, max_tokens=8192, temperature=0.7, top_p=1.0):
    responses = generate_batch(
        sglang_url=sglang_url,
        prompt_texts=[prompt_text],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=1,
    )
    return responses[0] if responses else ""


class MockSample:
    class Status:
        TRUNCATED = "truncated"
        COMPLETED = "completed"
        ABORTED = "aborted"
        FAILED = "failed"

    def __init__(
        self,
        response: str,
        metadata: dict,
        prompt: str = "",
        label: Any = "",
        tools: list[dict[str, Any]] | None = None,
        status: str = "completed",
    ):
        self.response = response
        self.metadata = metadata
        self.prompt = prompt
        self.label = label
        self.tools = tools or []
        self.status = status
        self.reward = None
        self.effective_response_length = len(response)


async def compute_rewards(reward_func, eval_samples, responses):
    rewards = []
    for sample_data, response in zip(eval_samples, responses):
        metadata = sample_data.get("metadata", {})
        prompt = sample_data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = json.dumps(prompt)
        mock = MockSample(
            response=response,
            metadata=metadata,
            prompt=prompt,
            label=sample_data.get("label", ""),
            tools=sample_data.get("tools") or [],
        )
        try:
            reward = await reward_func(None, mock)
        except Exception as exc:
            logger.warning("  Reward computation failed: %s", exc)
            reward = 0.0
        rewards.append(float(reward))
    return rewards


def compute_eval_metrics(eval_name, eval_samples, rewards, responses):
    metrics = {}
    n = len(rewards)
    if n == 0:
        return metrics

    metrics[f"eval/{eval_name}"] = sum(rewards) / n

    from collections import defaultdict

    by_dataset = defaultdict(list)
    by_domain = defaultdict(list)
    for sample_data, reward, response in zip(eval_samples, rewards, responses):
        metadata = sample_data.get("metadata", {}) or {}
        dataset_name = str(metadata.get("dataset_name", "unknown")).strip() or "unknown"
        domain = str(metadata.get("domain", "unknown")).strip() or "unknown"
        by_dataset[dataset_name].append(reward)
        by_domain[domain].append(reward)

    for dataset_name, values in by_dataset.items():
        metrics[f"eval/{eval_name}/{dataset_name}"] = sum(values) / len(values)
    for domain, values in by_domain.items():
        metrics[f"eval/{eval_name}/domain/{domain}"] = sum(values) / len(values)
    metrics[f"eval/{eval_name}/response_len"] = sum(len(response) for response in responses) / n
    return metrics


def log_to_wandb(
    *,
    metrics: dict[str, float],
    rollout_id: int,
    run_id: str,
    project: str,
    entity: str = "",
    host: str = "",
    key: str = "",
    group: str = "",
    run_name: str = "",
):
    if not metrics:
        logger.info("No eval metrics to log to W&B for rollout %s", rollout_id)
        return False

    if host:
        os.environ["WANDB_BASE_URL"] = host
    if key:
        os.environ["WANDB_API_KEY"] = key

    try:
        wandb.init(
            project=project,
            entity=entity or None,
            id=run_id,
            resume="allow",
            reinit=True,
            group=group or None,
            name=run_name or None,
        )
        wandb.define_metric("eval/step", overwrite=True)
        wandb.define_metric("eval/*", step_metric="eval/step", overwrite=True)
        wandb.log({"eval/step": rollout_id, **metrics})
        return True
    except Exception as exc:
        logger.warning("Skipping W&B logging because initialization or upload failed: %s", exc)
        return False
    finally:
        try:
            if wandb.run is not None:
                wandb.finish()
        except Exception as exc:
            logger.warning("Failed to finish W&B run cleanly: %s", exc)


def wait_for_sglang(base_url: str, timeout_seconds: int = 300):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=5)
            if response.ok:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang server did not become ready within {timeout_seconds}s: {base_url}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reward-module", default="multidomain_shared.reward_func")
    parser.add_argument("--eval-data", action="append", default=[])
    parser.add_argument("--rollout-id", type=int, required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-host", default="")
    parser.add_argument("--wandb-key", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--runtime-data-dir", default="")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--max-context-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bfcl-model-name", default="")
    parser.add_argument("--migrate-full-run", action="store_true", default=False)
    parser.add_argument("--migrate-eval-history", action="store_true", default=False)
    parser.add_argument("--target-wandb-run-id", default="")
    parser.add_argument("--target-wandb-run-name", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    wait_for_sglang(args.sglang_url)
    tokenizer = get_tokenizer(args.model_path)
    reward_func = load_reward_func(args.reward_module)
    combined_metrics: dict[str, float] = {}

    for spec in args.eval_data:
        eval_name, eval_path = spec.split(":", 1)
        logger.info("Evaluating %s from %s", eval_name, eval_path)
        eval_samples = load_eval_data(eval_path)
        prompt_texts = [apply_chat_template(tokenizer, sample) for sample in eval_samples]
        eval_samples, prompt_texts = filter_long_prompts(tokenizer, eval_samples, prompt_texts, args.max_context_len)

        metrics = {}
        if _is_bfcl_official_eval(eval_samples):
            runner = _load_bfcl_runner()
            if _is_bfcl_multi_turn_eval(eval_samples):
                outputs = runner["generate_bfcl_multi_turn_outputs"](
                    rows=eval_samples,
                    tokenizer=tokenizer,
                    generate_one=lambda prompt: generate_one(
                        args.sglang_url,
                        prompt,
                        max_tokens=args.max_tokens,
                    ),
                    max_prompt_tokens=args.max_context_len,
                )
            else:
                outputs = generate_batch(
                    sglang_url=args.sglang_url,
                    prompt_texts=prompt_texts,
                    max_tokens=args.max_tokens,
                    batch_size=args.batch_size,
                )
            model_name = args.bfcl_model_name or runner["DEFAULT_BFCL_MODEL_NAME"]
            summary = runner["run_bfcl_official_eval"](eval_samples, outputs, model_name=model_name)
            metrics = runner["summary_to_metrics"](eval_name, summary)
        else:
            responses = generate_batch(
                sglang_url=args.sglang_url,
                prompt_texts=prompt_texts,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
            )
            rewards = asyncio.run(compute_rewards(reward_func, eval_samples, responses))
            metrics = compute_eval_metrics(eval_name, eval_samples, rewards, responses)

        if args.dry_run:
            logger.info("Dry run metrics for %s: %s", eval_name, metrics)
            continue

        overlap = set(combined_metrics).intersection(metrics)
        if overlap:
            logger.warning("Eval metrics for %s overwrite existing keys: %s", eval_name, sorted(overlap))
        combined_metrics.update(metrics)

    if args.dry_run:
        return

    log_to_wandb(
        metrics=combined_metrics,
        rollout_id=args.rollout_id,
        run_id=args.wandb_run_id,
        project=args.wandb_project,
        entity=args.wandb_entity,
        host=args.wandb_host,
        key=args.wandb_key,
        group=args.wandb_group,
        run_name=getattr(args, "wandb_run_name", ""),
    )


if __name__ == "__main__":
    main()
