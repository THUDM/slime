"""Run real coding-agent rollouts and graders against an existing SGLang server.

Uses the same generate() entry point as training. No Ray trainers are started;
captured tokens, masks and log-probabilities are saved for inspection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import torch

from slime.utils.types import Sample


async def run(options):
    from .generate import _AdapterService, generate

    server = urlsplit(options.sglang_url)
    if server.scheme != "http" or not server.hostname or server.path not in ("", "/"):
        raise ValueError("--sglang-url must be an http://host:port URL")
    output = options.output
    output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(
        hf_checkpoint=options.hf_checkpoint,
        sglang_router_ip=server.hostname,
        sglang_router_port=server.port or 80,
        sglang_tool_call_parser="qwen3_coder",
        sglang_reasoning_parser="qwen3",
        rollout_max_context_len=options.max_context_tokens,
        apply_chat_template_kwargs=options.apply_chat_template_kwargs,
    )
    sampling = {
        "temperature": options.temperature,
        "top_p": 0.95,
        "max_new_tokens": options.max_response_tokens,
        "stop_token_ids": [248046, 248044],
    }
    rows = [json.loads(line) for line in options.data.read_text().splitlines() if line.strip()]
    semaphore = asyncio.Semaphore(options.concurrency)
    service = _AdapterService(args)
    results = []

    async def one(group_index, row, sample_index):
        index = group_index * options.samples_per_prompt + sample_index
        sample = Sample(
            index=index,
            group_index=group_index,
            prompt=row["prompt"],
            label=row.get("label"),
            metadata=row.get("metadata", {}),
        )
        async with semaphore:
            start = time.monotonic()
            branches = await generate(args, sample, sampling)
            elapsed = time.monotonic() - start
        for branch in branches:
            assert len(branch.loss_mask) == len(branch.rollout_log_probs) == branch.response_length
            assert all(math.isfinite(value) for value in branch.rollout_log_probs)
        torch.save(
            {"rollout_id": 0, "samples": [branch.to_dict() for branch in branches]}, output / f"sample-{index}.pt"
        )
        result = {
            "index": index,
            "instance_id": row.get("metadata", {}).get("instance_id", row.get("label")),
            "reward": branches[0].reward if branches else None,
            "branches": len(branches),
            "trainable_tokens": sum(sum(branch.loss_mask) for branch in branches),
            "agent_exit_codes": [branch.metadata.get("agent_exit_code") for branch in branches],
            "elapsed_seconds": elapsed,
        }
        results.append(result)
        (output / "results.json").write_text(
            json.dumps(sorted(results, key=lambda item: item["index"]), indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)

    try:
        await asyncio.gather(
            *(
                one(group, row, sample)
                for group, row in enumerate(rows)
                for sample in range(options.samples_per_prompt)
            )
        )
    finally:
        await asyncio.to_thread(service.app_handle.stop)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=65536)
    parser.add_argument("--max-response-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default={})
    options = parser.parse_args()
    if options.concurrency < 1 or options.samples_per_prompt < 1:
        parser.error("concurrency and samples-per-prompt must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = asyncio.run(run(options))
    if not any(result["reward"] == 1 for result in results):
        raise SystemExit("No reward=1 rollout; see the saved trajectories and fresh-sandbox grader logs.")


if __name__ == "__main__":
    main()
