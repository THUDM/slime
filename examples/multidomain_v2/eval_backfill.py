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
import sys
import time
from pathlib import Path
from typing import Any

import requests
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parents[1]

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

    def __init__(self, response: str, metadata: dict, prompt: str = "", status: str = "completed"):
        self.response = response
        self.metadata = metadata
        self.prompt = prompt
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
        mock = MockSample(response=response, metadata=metadata, prompt=prompt)
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
        meta = sample_data.get("metadata", {})
        ds_name = meta.get("dataset_name", eval_name)
        domain = meta.get("domain", "unknown")
        by_dataset[ds_name].append((reward, response))
        by_domain[domain].append((reward, response))

    for ds_name, items in by_dataset.items():
        ds_rewards = [r for r, _ in items]
        ds_responses = [resp for _, resp in items]
        metrics[f"eval/{eval_name}/by_source/{ds_name}/count"] = len(items)
        metrics[f"eval/{eval_name}/by_source/{ds_name}/score"] = sum(ds_rewards) / len(ds_rewards)
        lengths = [len(r) for r in ds_responses]
        if lengths:
            metrics[f"eval/{eval_name}/by_source/{ds_name}/response_len/mean"] = sum(lengths) / len(lengths)

    for domain_name, items in by_domain.items():
        dom_rewards = [r for r, _ in items]
        metrics[f"eval/{eval_name}/by_domain/{domain_name}/count"] = len(items)
        metrics[f"eval/{eval_name}/by_domain/{domain_name}/score"] = sum(dom_rewards) / len(dom_rewards)

    return metrics


def log_to_wandb(wandb_run_id, wandb_project, wandb_entity, wandb_host, wandb_key, wandb_group, wandb_run_name, step, metrics):
    if wandb_key:
        login_kwargs = {"key": wandb_key}
        if wandb_host:
            login_kwargs["host"] = wandb_host
        wandb.login(**login_kwargs)

    init_kwargs: dict[str, Any] = {
        "id": wandb_run_id,
        "project": wandb_project,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(mode="shared"),
    }
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    if wandb_group:
        init_kwargs["group"] = wandb_group
    if wandb_run_name:
        init_kwargs["name"] = wandb_run_name
    wandb.init(**init_kwargs)
    wandb.define_metric("eval/step", overwrite=True)
    wandb.define_metric("eval/*", step_metric="eval/step", overwrite=True)
    metrics["eval/step"] = step
    wandb.log(metrics)
    wandb.finish()
    logger.info("  Logged %d metrics to wandb run %s at step %s", len(metrics), wandb_run_id, step)


def _official_benchmark_dirs(runtime_data_dir: str, eval_path: str, eval_name: str, rollout_id: int):
    base = Path(runtime_data_dir) if runtime_data_dir else Path(eval_path).resolve().parent
    root = base / "official_benchmarks" / eval_name / f"step_{rollout_id}"
    return root / "result", root / "score"


def wait_for_sglang(url: str, timeout: int = 300):
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
    parser.add_argument("--migrate-eval-history", action="store_true")
    parser.add_argument("--migrate-full-run", action="store_true")
    parser.add_argument("--sglang-url", type=str, default="http://localhost:30000")
    parser.add_argument("--eval-data", action="append", default=[], help="name:path pairs for eval datasets")
    parser.add_argument("--rollout-id", type=int, default=None, help="Rollout ID (= checkpoint step) for wandb logging")
    parser.add_argument("--wandb-run-id", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, default="slime-multidomain-v2")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-host", type=str, default="")
    parser.add_argument("--wandb-key", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--target-wandb-run-id", type=str, default="")
    parser.add_argument("--target-wandb-run-name", type=str, default="")
    parser.add_argument("--runtime-data-dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-path", type=str, default=None, help="HF model path (for tokenizer chat template)")
    parser.add_argument("--max-context-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bfcl-model-name", type=str, default="")
    parser.add_argument("--reward-module", type=str, default="multidomain_shared.reward_func")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.migrate_eval_history or args.migrate_full_run:
        raise RuntimeError("Migration modes are not supported in this lightweight recovery version")

    if not args.eval_data:
        raise RuntimeError("--eval-data is required")
    if args.rollout_id is None:
        raise RuntimeError("--rollout-id is required")

    reward_func = load_reward_func(args.reward_module)
    bfcl_runner = None

    model_path = args.model_path
    if not model_path:
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
    bfcl_model_name = args.bfcl_model_name or "gorilla-openfunctions-v2"

    eval_datasets = {}
    for spec in args.eval_data:
        name, path = spec.split(":", 1)
        eval_datasets[name] = path

    wait_for_sglang(args.sglang_url)

    all_metrics = {}
    for eval_name, eval_path in eval_datasets.items():
        logger.info("Running eval: %s from %s", eval_name, eval_path)
        eval_samples = load_eval_data(eval_path)
        logger.info("  Loaded %d samples", len(eval_samples))

        if _is_bfcl_official_eval(eval_samples):
            if bfcl_runner is None:
                bfcl_runner = _load_bfcl_runner()
                bfcl_model_name = args.bfcl_model_name or bfcl_runner["DEFAULT_BFCL_MODEL_NAME"]
            if _is_bfcl_multi_turn_eval(eval_samples):
                outputs = bfcl_runner["generate_bfcl_multi_turn_outputs"](
                    eval_samples,
                    tokenizer=tokenizer,
                    generate_one=lambda prompt_text: generate_one(
                        sglang_url=args.sglang_url,
                        prompt_text=prompt_text,
                        max_tokens=args.max_tokens,
                    ),
                    max_prompt_tokens=args.max_context_len,
                )
            else:
                prompt_texts = [apply_chat_template(tokenizer, sample) for sample in eval_samples]
                eval_samples, prompt_texts = filter_long_prompts(
                    tokenizer,
                    eval_samples,
                    prompt_texts,
                    args.max_context_len,
                )
                if not eval_samples:
                    logger.warning("  All BFCL samples filtered out for %s, skipping", eval_name)
                    continue
                outputs = generate_batch(
                    sglang_url=args.sglang_url,
                    prompt_texts=prompt_texts,
                    max_tokens=args.max_tokens,
                    batch_size=args.batch_size,
                )

            result_dir, score_dir = _official_benchmark_dirs(
                args.runtime_data_dir,
                eval_path,
                eval_name,
                args.rollout_id,
            )
            summary = bfcl_runner["run_bfcl_official_eval"](
                eval_samples,
                outputs,
                model_name=bfcl_model_name,
                result_dir=result_dir,
                score_dir=score_dir,
            )
            logger.info("  %s: official BFCL accuracy = %.4f", eval_name, summary["overall_accuracy"])
            all_metrics.update(bfcl_runner["summary_to_metrics"](eval_name, summary))
            continue

        prompt_texts = [apply_chat_template(tokenizer, sample) for sample in eval_samples]
        eval_samples, prompt_texts = filter_long_prompts(tokenizer, eval_samples, prompt_texts, args.max_context_len)
        if not eval_samples:
            logger.warning("  All samples filtered out for %s, skipping", eval_name)
            continue

        responses = generate_batch(
            sglang_url=args.sglang_url,
            prompt_texts=prompt_texts,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
        rewards = asyncio.run(compute_rewards(reward_func, eval_samples, responses))
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info("  %s: avg reward = %.4f", eval_name, avg_reward)
        all_metrics.update(compute_eval_metrics(eval_name, eval_samples, rewards, responses))

    if all_metrics:
        log_to_wandb(
            wandb_run_id=args.wandb_run_id,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_host=args.wandb_host,
            wandb_key=args.wandb_key,
            wandb_group=args.wandb_group,
            wandb_run_name=args.wandb_run_name,
            step=args.rollout_id,
            metrics=all_metrics,
        )

    logger.info("Eval backfill complete.")


if __name__ == "__main__":
    main()
