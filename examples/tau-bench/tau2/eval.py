#!/usr/bin/env python3
"""Evaluate a checkpoint on tau2-bench with Pass@K sampling.

This script extends eval.py to support Pass@K evaluation:
- Samples k times per task with temperature > 0
- Selects the best attempt (prioritizing success)
- Logs all k attempts for analysis
- Reports both Pass@1 and Pass@K metrics

Example:
  python3 examples/tau-bench/tau2/eval_passk.py \
    --hf-checkpoint /models/Qwen3-4B-tau2-grpo-v1 \
    --sglang-url http://127.0.0.1:30000/generate \
    --domains airline,retail,telecom \
    --task-split test \
    --max-tasks-per-domain 25 \
    --num-samples 4 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --output /workspace/tau2_pass4_eval.json

Note: Default parameters follow Qwen3 best practices:
  temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

# Optional: Weave for tracing
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    weave = None  # type: ignore[assignment]

# Optional: WandB for metrics
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]

# Add script dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from actions import env_action_from_parsed_action, followup_messages_for_observation, parse_action
from env import compute_partial_score_from_reward_info, parse_reward_info
from prompting import build_tau2_agent_system_prompt

logger = logging.getLogger(__name__)

DEFAULT_DOMAINS = ("airline", "retail", "telecom")


def _parse_csv(value: str) -> list[str]:
    items = [x.strip() for x in value.split(",")]
    return [x for x in items if x]

def _get_user_llm_args(*, temperature: float) -> dict[str, Any]:
    """Build LiteLLM user simulator args.

    Supports local OpenAI-compatible servers via TAU2_USER_API_BASE.
    """
    user_llm_args: dict[str, Any] = {"temperature": temperature}

    user_api_base = os.environ.get("TAU2_USER_API_BASE", "").strip()
    if user_api_base:
        user_llm_args["api_base"] = user_api_base
        user_llm_args["api_key"] = "dummy-key-for-local-server"
        user_llm_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    return user_llm_args


@dataclass(frozen=True, slots=True)
class EvalResult:
    domain: str
    task_split: str
    task_index: int
    task_id: str
    success: bool
    reward: float
    partial_score: float
    partial_components: dict[str, float]
    steps: int
    status: str
    sample_idx: int = 0  # Which sample (0 to k-1)
    error: str | None = None
    reward_info: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PassKResult:
    """Aggregated Pass@K result for a task."""
    domain: str
    task_split: str
    task_index: int
    task_id: str
    num_samples: int
    best_success: bool
    best_reward: float
    best_partial_score: float
    best_sample_idx: int
    all_attempts: list[dict[str, Any]]  # All k attempts
    pass_at_1: float  # 1.0 if first attempt succeeded, else 0.0
    pass_at_k: float  # 1.0 if any attempt succeeded, else 0.0


class SGLangClient:
    def __init__(self, url: str, *, max_connections: int = 32, timeout_s: float = 300.0):
        self.url = url
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout_s),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(self, *, text: str, sampling_params: dict[str, Any], max_retries: int = 30) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = await self._client.post(self.url, json={"text": text, "sampling_params": sampling_params})
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(min(2.0, 0.25 * (2**attempt)))
        raise RuntimeError(f"SGLang request failed after {max_retries} retries: {last_exc}")


def _load_tasks(domain: str, task_split: str) -> list[str]:
    from tau2.registry import registry
    tasks = registry.get_tasks_loader(domain)(task_split)
    return [t.id for t in tasks]


async def _evaluate_task_single_attempt(
    *,
    client: SGLangClient,
    tokenizer,
    domain: str,
    task_split: str,
    task_index: int,
    task_id: str,
    sample_idx: int,
    sampling_params: dict[str, Any],
    max_steps: int,
    user_llm: str,
    user_llm_args: dict[str, Any],
) -> EvalResult:
    """Evaluate a single attempt of a task (one of k samples)."""
    from tau2.gym.gym_agent import AgentGymEnv

    env = AgentGymEnv(
        domain=domain,
        task_id=task_id,
        max_steps=max_steps,
        solo_mode=False,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        all_messages_as_observation=False,
    )

    observation, info = env.reset()
    tools = info.get("tools", [])
    tools_openai = [t if isinstance(t, dict) else t.openai_schema for t in tools]

    policy = info.get("policy", "")
    system_prompt = build_tau2_agent_system_prompt(domain=domain, policy=policy, tools_openai=tools_openai)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(
        followup_messages_for_observation(
            observation=observation,
            last_action_call="(reset)",
            last_action_was_tool=False,
            native_fc=True,
        )
    )

    assistant_texts: list[str] = []
    terminated = False
    reward = 0.0
    reward_info: dict[str, Any] = {}

    for step in range(max_steps):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        out = await client.generate(text=prompt_text, sampling_params=sampling_params)
        if out.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
            return EvalResult(
                domain=domain,
                task_split=task_split,
                task_index=task_index,
                task_id=task_id,
                sample_idx=sample_idx,
                success=False,
                reward=0.0,
                partial_score=0.0,
                partial_components={},
                steps=step,
                status="aborted",
                error="sglang_abort",
            )

        assistant_text = (out.get("text") or "").strip()
        if not assistant_text:
            return EvalResult(
                domain=domain,
                task_split=task_split,
                task_index=task_index,
                task_id=task_id,
                sample_idx=sample_idx,
                success=False,
                reward=0.0,
                partial_score=0.0,
                partial_components={},
                steps=step,
                status="empty_generation",
                error="empty_generation",
            )

        parsed = None
        parse_err: str | None = None
        for parse_attempt in range(2):
            try:
                parsed = parse_action(assistant_text)
                break
            except Exception as exc:
                parse_err = f"{type(exc).__name__}: {exc}"
                if parse_attempt == 1:
                    break
                messages.append({"role": "assistant", "content": assistant_text})
                messages.append(
                    {
                        "role": "user",
                        "content": "FORMAT ERROR. Re-output EXACTLY in the required <tool_call> format: "
                        '<tool_call>{"name": "...", "arguments": {...}}</tool_call>. One action only.',
                    }
                )
                repair_params = {**sampling_params, "temperature": 0.0}
                out = await client.generate(text=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), sampling_params=repair_params)
                assistant_text = (out.get("text") or "").strip()

        messages.append({"role": "assistant", "content": assistant_text})
        assistant_texts.append(assistant_text)

        if parsed is None:
            partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
            return EvalResult(
                domain=domain,
                task_split=task_split,
                task_index=task_index,
                task_id=task_id,
                sample_idx=sample_idx,
                success=False,
                reward=float(reward),
                partial_score=partial_score,
                partial_components=partial_components,
                steps=step + 1,
                status="parse_error",
                error=parse_err,
                reward_info=reward_info,
            )

        env_action = env_action_from_parsed_action(parsed)
        observation, reward, terminated, _truncated, info = env.step(env_action)

        if terminated:
            reward_info = parse_reward_info(info)
            break

        messages.extend(
            followup_messages_for_observation(
                observation=observation,
                last_action_call=parsed.raw_action_call,
                last_action_was_tool=(parsed.name != "respond"),
                native_fc=True,
            )
        )

    if not terminated:
        reward_info = parse_reward_info(info)

    partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
    success = float(reward) >= 1.0 - 1e-9
    status = "completed" if terminated else "truncated"
    return EvalResult(
        domain=domain,
        task_split=task_split,
        task_index=task_index,
        task_id=task_id,
        sample_idx=sample_idx,
        success=success,
        reward=float(reward),
        partial_score=partial_score,
        partial_components=partial_components,
        steps=len(assistant_texts),
        status=status,
        reward_info=reward_info,
    )


async def evaluate_task_passk(
    *,
    client: SGLangClient,
    tokenizer,
    domain: str,
    task_split: str,
    task_index: int,
    task_id: str,
    num_samples: int,
    sampling_params: dict[str, Any],
    max_steps: int,
    user_llm: str,
    user_llm_args: dict[str, Any],
) -> PassKResult:
    """Evaluate a task k times and return best result."""

    # Run k attempts
    attempts: list[EvalResult] = []
    for sample_idx in range(num_samples):
        logger.info(f"Task {task_id} (idx={task_index}): sample {sample_idx+1}/{num_samples}")
        result = await _evaluate_task_single_attempt(
            client=client,
            tokenizer=tokenizer,
            domain=domain,
            task_split=task_split,
            task_index=task_index,
            task_id=task_id,
            sample_idx=sample_idx,
            sampling_params=sampling_params,
            max_steps=max_steps,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
        )
        attempts.append(result)

        # Early stopping: if we get a perfect success, we can stop
        if result.success and result.reward >= 1.0:
            logger.info(f"Task {task_id}: Early success on sample {sample_idx+1}, skipping remaining samples")
            # Fill remaining with None to maintain count
            for remaining_idx in range(sample_idx + 1, num_samples):
                attempts.append(EvalResult(
                    domain=domain,
                    task_split=task_split,
                    task_index=task_index,
                    task_id=task_id,
                    sample_idx=remaining_idx,
                    success=False,
                    reward=0.0,
                    partial_score=0.0,
                    partial_components={},
                    steps=0,
                    status="skipped",
                    error="early_stopping",
                ))
            break

    # Select best attempt (prioritize success, then reward, then partial score)
    best_idx = 0
    best = attempts[0]
    for idx, attempt in enumerate(attempts[1:], start=1):
        if attempt.status == "skipped":
            continue
        if attempt.success and not best.success:
            best_idx = idx
            best = attempt
        elif attempt.success == best.success:
            if attempt.reward > best.reward:
                best_idx = idx
                best = attempt
            elif attempt.reward == best.reward and attempt.partial_score > best.partial_score:
                best_idx = idx
                best = attempt

    # Compute Pass@1 and Pass@K
    pass_at_1 = 1.0 if attempts[0].success else 0.0
    pass_at_k = 1.0 if any(a.success for a in attempts if a.status != "skipped") else 0.0

    return PassKResult(
        domain=domain,
        task_split=task_split,
        task_index=task_index,
        task_id=task_id,
        num_samples=num_samples,
        best_success=best.success,
        best_reward=best.reward,
        best_partial_score=best.partial_score,
        best_sample_idx=best_idx,
        all_attempts=[asdict(a) for a in attempts],
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
    )


# Apply Weave tracing if available
if WEAVE_AVAILABLE and weave is not None:
    evaluate_task_passk = weave.op(evaluate_task_passk)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tau2-bench with Pass@K sampling")
    parser.add_argument("--hf-checkpoint", required=True, help="HF checkpoint path")
    parser.add_argument("--sglang-url", required=True, help="SGLang HTTP /generate URL")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS), help="Comma-separated domains")
    parser.add_argument("--task-split", default="base", choices=("train", "test", "base"))
    parser.add_argument("--max-tasks-per-domain", type=int, default=None)
    parser.add_argument("--start-task-index", type=int, default=0, help="Skip tasks before this index (for resuming)")
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("TAU2_MAX_STEPS", "100")))
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples per task (k in Pass@K)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (Qwen3 recommended: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling (Qwen3 recommended: 0.8)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling (Qwen3 recommended: 20)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (Qwen3 recommended: 1.0)")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--user-model", default=os.environ.get("TAU2_USER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--user-temperature", type=float, default=float(os.environ.get("TAU2_USER_TEMPERATURE", "0.7")))
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default="eval-passk")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    # Initialize WandB + Weave
    wandb_run = None
    if args.wandb_project and WANDB_AVAILABLE:
        checkpoint_name = Path(args.hf_checkpoint).name
        run_name = args.wandb_name or f"eval-pass{args.num_samples}-{args.task_split}-{checkpoint_name}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            config={
                "hf_checkpoint": args.hf_checkpoint,
                "sglang_url": args.sglang_url,
                "task_split": args.task_split,
                "domains": args.domains,
                "max_tasks_per_domain": args.max_tasks_per_domain,
                "max_steps": args.max_steps,
                "num_samples": args.num_samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "user_model": args.user_model,
            },
        )
        if WEAVE_AVAILABLE:
            weave.init(args.wandb_project)

    domains = _parse_csv(args.domains)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "stop": ["</tool_call>"],
        # Keep the stop sequence in the returned text so we can parse `</tool_call>`.
        "no_stop_trim": True,
    }

    client = SGLangClient(args.sglang_url)

    user_llm_args = _get_user_llm_args(temperature=args.user_temperature)
    all_results: list[PassKResult] = []

    for domain in domains:
        task_ids = _load_tasks(domain, args.task_split)
        if args.max_tasks_per_domain:
            task_ids = task_ids[:args.max_tasks_per_domain]

        # Apply start index for resuming
        start_idx = args.start_task_index
        tasks_to_eval = task_ids[start_idx:]

        logger.info(f"Evaluating domain={domain} split={args.task_split} tasks={len(tasks_to_eval)}/{len(task_ids)} samples_per_task={args.num_samples}")
        if start_idx > 0:
            logger.info(f"Resuming from task index {start_idx}, skipping first {start_idx} tasks")

        for offset, task_id in enumerate(tasks_to_eval):
            task_index = start_idx + offset  # Preserve original task index
            result = await evaluate_task_passk(
                client=client,
                tokenizer=tokenizer,
                domain=domain,
                task_split=args.task_split,
                task_index=task_index,
                task_id=task_id,
                num_samples=args.num_samples,
                sampling_params=sampling_params,
                max_steps=args.max_steps,
                user_llm=args.user_model,
                user_llm_args=user_llm_args,
            )
            all_results.append(result)

            # Log to WandB
            if wandb_run:
                wandb.log({
                    f"{domain}/pass@1": result.pass_at_1,
                    f"{domain}/pass@{args.num_samples}": result.pass_at_k,
                    f"{domain}/best_reward": result.best_reward,
                    f"{domain}/best_partial": result.best_partial_score,
                })

    await client.close()

    # Compute aggregate metrics
    domain_metrics = {}
    for domain in domains:
        domain_results = [r for r in all_results if r.domain == domain]
        if domain_results:
            pass1 = sum(r.pass_at_1 for r in domain_results) / len(domain_results)
            passk = sum(r.pass_at_k for r in domain_results) / len(domain_results)
            domain_metrics[domain] = {
                "pass@1": pass1,
                f"pass@{args.num_samples}": passk,
                "num_tasks": len(domain_results),
            }

    overall_pass1 = sum(r.pass_at_1 for r in all_results) / len(all_results) if all_results else 0.0
    overall_passk = sum(r.pass_at_k for r in all_results) / len(all_results) if all_results else 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"Pass@1: {overall_pass1:.1%}")
    logger.info(f"Pass@{args.num_samples}: {overall_passk:.1%}")
    logger.info(f"{'='*60}")
    for domain, metrics in domain_metrics.items():
        logger.info(f"{domain}: Pass@1={metrics['pass@1']:.1%}, Pass@{args.num_samples}={metrics[f'pass@{args.num_samples}']:.1%}")

    # Save results
    output_data = {
        "config": {
            "hf_checkpoint": args.hf_checkpoint,
            "task_split": args.task_split,
            "domains": domains,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
        "summary": {
            "overall_pass@1": overall_pass1,
            f"overall_pass@{args.num_samples}": overall_passk,
            "domain_metrics": domain_metrics,
        },
        "results": [asdict(r) for r in all_results],
    }

    Path(args.output).write_text(json.dumps(output_data, indent=2))
    logger.info(f"Results saved to {args.output}")

    if wandb_run:
        wandb.log({
            "overall/pass@1": overall_pass1,
            f"overall/pass@{args.num_samples}": overall_passk,
        })
        wandb.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
