#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import uuid
from pathlib import Path

SLIME_ROOT = Path(__file__).resolve().parents[2]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))


def command_short_sha1(args: argparse.Namespace) -> int:
    print(hashlib.sha1(args.text.encode("utf-8")).hexdigest()[:8])
    return 0


def command_filter_jsonl_by_prompt_budget(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    path = Path(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    kept = 0
    skipped = 0
    worst_tokens = -1
    worst_record = ""

    with path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)
            prompt_messages = sample.get("prompt")
            tools = sample.get("tools")
            if isinstance(prompt_messages, list):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tools=tools if tools else None,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif isinstance(prompt_messages, str):
                prompt_text = prompt_messages
            else:
                prompt_text = json.dumps(prompt_messages, ensure_ascii=False)
            prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            if prompt_tokens <= args.max_prompt_tokens:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
                continue
            skipped += 1
            if prompt_tokens > worst_tokens:
                metadata = sample.get("metadata") or {}
                worst_tokens = prompt_tokens
                worst_record = str(metadata.get("dataset_name") or metadata.get("record_id") or "")

    if kept == 0:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Filtered every sample from {args.label} for exceeding {args.max_prompt_tokens} prompt tokens"
        )

    tmp_path.replace(path)
    print(
        f"Filtered {skipped} samples exceeding {args.max_prompt_tokens} prompt tokens for {args.label}: "
        f"kept={kept} worst_tokens={worst_tokens} worst_record={worst_record}"
    )
    return 0


def command_ray_cluster_ready(args: argparse.Namespace) -> int:
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{args.dashboard_port}/api/cluster_status",
            timeout=args.timeout_seconds,
        ) as response:
            payload = json.load(response)
    except Exception:
        return 1

    report = payload.get("data", {}).get("clusterStatus", {}).get("autoscalerReport", {})
    active_nodes = report.get("activeNodes") or {}
    usage = payload.get("data", {}).get("clusterStatus", {}).get("loadMetricsReport", {}).get("usage", {})
    gpu_total = (usage.get("GPU") or [0.0, 0.0])[1]

    if len(active_nodes) >= args.expected_nodes and gpu_total >= args.expected_gpus:
        return 0
    return 1


def command_resolve_hostname_to_ip(args: argparse.Namespace) -> int:
    try:
        print(socket.gethostbyname(args.host))
    except OSError:
        return 1
    return 0


def command_is_ip_address(args: argparse.Namespace) -> int:
    import ipaddress

    try:
        ipaddress.ip_address(args.candidate)
    except ValueError:
        return 1
    return 0


def command_extract_ray_head_addr(args: argparse.Namespace) -> int:
    import re

    patterns = [
        r"--address='([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+):[0-9]+'",
        r"ray\.init\(_node_ip_address='([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)'\)",
        r"http://([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+):8265",
        r"Local node IP.*?([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, args.text)
        if match:
            print(match.group(1))
            return 0
    return 1


def command_random_run_id(args: argparse.Namespace) -> int:
    del args
    print(uuid.uuid4().hex[:8])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    short_sha1 = subparsers.add_parser("short-sha1")
    short_sha1.add_argument("--text", required=True)
    short_sha1.set_defaults(func=command_short_sha1)

    filter_jsonl = subparsers.add_parser("filter-jsonl-by-prompt-budget")
    filter_jsonl.add_argument("--input", required=True)
    filter_jsonl.add_argument("--label", required=True)
    filter_jsonl.add_argument("--model-dir", required=True)
    filter_jsonl.add_argument("--max-prompt-tokens", required=True, type=int)
    filter_jsonl.set_defaults(func=command_filter_jsonl_by_prompt_budget)

    cluster_ready = subparsers.add_parser("ray-cluster-ready")
    cluster_ready.add_argument("--dashboard-port", required=True, type=int)
    cluster_ready.add_argument("--timeout-seconds", required=True, type=int)
    cluster_ready.add_argument("--expected-gpus", required=True, type=int)
    cluster_ready.add_argument("--expected-nodes", required=True, type=int)
    cluster_ready.set_defaults(func=command_ray_cluster_ready)

    resolve_host = subparsers.add_parser("resolve-hostname-to-ip")
    resolve_host.add_argument("--host", required=True)
    resolve_host.set_defaults(func=command_resolve_hostname_to_ip)

    is_ip = subparsers.add_parser("is-ip-address")
    is_ip.add_argument("--candidate", required=True)
    is_ip.set_defaults(func=command_is_ip_address)

    ray_addr = subparsers.add_parser("extract-ray-head-addr")
    ray_addr.add_argument("--text", required=True)
    ray_addr.set_defaults(func=command_extract_ray_head_addr)

    random_run_id = subparsers.add_parser("random-run-id")
    random_run_id.set_defaults(func=command_random_run_id)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
