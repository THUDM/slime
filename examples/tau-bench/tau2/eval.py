#!/usr/bin/env python3
"""Pass@K evaluation for tau2-bench using an SGLang-served policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from actions import env_action_from_parsed_action, followup_messages_for_observation, parse_action
from env import compute_partial_score_from_reward_info, parse_reward_info
from prompting import build_tau2_agent_system_prompt

logger = logging.getLogger(__name__)

DEFAULT_DOMAINS = ("airline", "retail", "telecom")
PASS_AT_K_NOTE = "pass@k = any success among k attempts (not pass^k leaderboard estimate)"


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _get_user_llm_args(*, temperature: float) -> dict[str, Any]:
    args: dict[str, Any] = {"temperature": temperature}
    api_base = os.environ.get("TAU2_USER_API_BASE", "").strip()
    if api_base:
        args["api_base"] = api_base
        args["api_key"] = "dummy-key-for-local-server"
        args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return args


@dataclass(frozen=True, slots=True)
class AttemptResult:
    success: bool
    reward: float
    partial_score: float
    partial_components: dict[str, float]
    steps: int
    status: str
    error: str | None = None
    reward_info: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PassKResult:
    domain: str
    task_split: str
    task_index: int
    task_id: str
    num_samples: int
    best_success: bool
    best_reward: float
    best_partial_score: float
    best_sample_idx: int
    attempts: list[dict[str, Any]]
    pass_at_1: float
    pass_at_k: float


class SGLangClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(self, *, text: str, sampling_params: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.post(self.url, json={"text": text, "sampling_params": sampling_params})
        resp.raise_for_status()
        return resp.json()


def _load_tasks(domain: str, task_split: str) -> list[str]:
    from tau2.registry import registry

    return [t.id for t in registry.get_tasks_loader(domain)(task_split)]


async def _run_one_attempt(
    *,
    client: SGLangClient,
    tokenizer,
    domain: str,
    task_id: str,
    sampling_params: dict[str, Any],
    max_steps: int,
    user_llm: str,
    user_llm_args: dict[str, Any],
) -> AttemptResult:
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
        )
    )

    reward = 0.0
    reward_info: dict[str, Any] = {}

    for step in range(max_steps):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = await client.generate(text=prompt_text, sampling_params=sampling_params)

        if out.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
            return AttemptResult(
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
            return AttemptResult(
                success=False,
                reward=0.0,
                partial_score=0.0,
                partial_components={},
                steps=step,
                status="empty_generation",
                error="empty_generation",
            )

        try:
            parsed = parse_action(assistant_text)
        except Exception as exc:
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": "FORMAT ERROR. Re-output EXACTLY in the required <tool_call> format: "
                    '<tool_call>{"name": "...", "arguments": {...}}</tool_call>. One action only.',
                }
            )
            repair_params = {**sampling_params, "temperature": 0.0}
            out = await client.generate(
                text=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                sampling_params=repair_params,
            )
            assistant_text = (out.get("text") or "").strip()
            try:
                parsed = parse_action(assistant_text)
            except Exception:
                partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
                return AttemptResult(
                    success=False,
                    reward=float(reward),
                    partial_score=partial_score,
                    partial_components=partial_components,
                    steps=step + 1,
                    status="parse_error",
                    error=str(exc),
                    reward_info=reward_info,
                )

        messages.append({"role": "assistant", "content": assistant_text})

        env_action = env_action_from_parsed_action(parsed)
        observation, reward, terminated, _truncated, info = env.step(env_action)

        if terminated:
            reward_info = parse_reward_info(info)
            partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
            return AttemptResult(
                success=float(reward) >= 1.0,
                reward=float(reward),
                partial_score=partial_score,
                partial_components=partial_components,
                steps=step + 1,
                status="completed",
                reward_info=reward_info,
            )

        messages.extend(
            followup_messages_for_observation(
                observation=observation,
                last_action_call=parsed.raw_action_call,
                last_action_was_tool=(parsed.name != "respond"),
            )
        )

    partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
    return AttemptResult(
        success=False,
        reward=float(reward),
        partial_score=partial_score,
        partial_components=partial_components,
        steps=max_steps,
        status="truncated",
        reward_info=reward_info,
    )


def _summarize(results: list[PassKResult], *, k: int) -> dict[str, Any]:
    total = len(results)
    pass1 = sum(r.pass_at_1 for r in results) / total if total else 0.0
    passk = sum(r.pass_at_k for r in results) / total if total else 0.0
    return {"total": total, "pass_at_1": pass1, f"pass_at_{k}": passk}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate tau2-bench with Pass@K sampling")
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--task-split", default="test", choices=("train", "test", "base"))
    parser.add_argument("--max-tasks-per-domain", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("TAU2_MAX_STEPS", "100")))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--user-model", default=os.environ.get("TAU2_USER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--user-temperature", type=float, default=float(os.environ.get("TAU2_USER_TEMPERATURE", "0.7")))
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")


async def main_async() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    domains = _parse_csv(args.domains)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    sampling_params: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "stop": ["</tool_call>"],
        "no_stop_trim": True,
    }
    if args.top_k > 0:
        sampling_params["top_k"] = args.top_k

    client = SGLangClient(args.sglang_url)
    try:
        user_llm_args = _get_user_llm_args(temperature=args.user_temperature)
        all_results: list[PassKResult] = []

        for domain in domains:
            task_ids = _load_tasks(domain, args.task_split)
            if args.max_tasks_per_domain is not None:
                task_ids = task_ids[: args.max_tasks_per_domain]

            logger.info(f"Evaluating domain={domain} split={args.task_split} tasks={len(task_ids)} k={args.num_samples}")
            for i, task_id in enumerate(task_ids):
                attempts: list[AttemptResult] = []
                for _ in range(args.num_samples):
                    attempts.append(
                        await _run_one_attempt(
                            client=client,
                            tokenizer=tokenizer,
                            domain=domain,
                            task_id=task_id,
                            sampling_params=sampling_params,
                            max_steps=args.max_steps,
                            user_llm=args.user_model,
                            user_llm_args=user_llm_args,
                        )
                    )

                best_idx = 0
                for j in range(1, len(attempts)):
                    a = attempts[j]
                    b = attempts[best_idx]
                    if a.success and not b.success:
                        best_idx = j
                    elif a.success == b.success and a.partial_score > b.partial_score:
                        best_idx = j

                pass_at_1 = 1.0 if attempts and attempts[0].success else 0.0
                pass_at_k = 1.0 if any(a.success for a in attempts) else 0.0
                best = attempts[best_idx]
                all_results.append(
                    PassKResult(
                        domain=domain,
                        task_split=args.task_split,
                        task_index=i,
                        task_id=task_id,
                        num_samples=args.num_samples,
                        best_success=best.success,
                        best_reward=best.reward,
                        best_partial_score=best.partial_score,
                        best_sample_idx=best_idx,
                        attempts=[asdict(a) for a in attempts],
                        pass_at_1=pass_at_1,
                        pass_at_k=pass_at_k,
                    )
                )

        by_domain: dict[str, list[PassKResult]] = {}
        for r in all_results:
            by_domain.setdefault(r.domain, []).append(r)

        report = {
            "hf_checkpoint": args.hf_checkpoint,
            "sglang_url": args.sglang_url,
            "task_split": args.task_split,
            "domains": domains,
            "k": args.num_samples,
            "metric_note": PASS_AT_K_NOTE,
            "summary": _summarize(all_results, k=args.num_samples),
            "by_domain": {d: _summarize(rs, k=args.num_samples) for d, rs in sorted(by_domain.items())},
            "results": [asdict(r) for r in all_results],
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Wrote {len(all_results)} results to {args.output}")
        logger.info(f"Overall: {report['summary']}")
        for d, s in report["by_domain"].items():
            logger.info(f"{d}: {s}")
    finally:
        await client.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
