#!/usr/bin/env python3
"""Build OSWorld training task registry and replay buffer.

This script merges all OSWorld-related HF datasets into:
  - osworld_tasks_all.parquet: all base task_ids (with missing config flagged)
  - osworld_tasks_train.parquet: tasks with full task_config and replay overlap (Ubuntu)
  - osworld_replay_train.jsonl: expanded replay buffer with normalized system prompt
  - osworld_train_stats.json: summary stats for the train subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

from examples.osworld.env import OSWORLD_SYSTEM_PROMPT, OSWorldEnvConfig, OSWorldEnvWrapper


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    quality_rank: int


def _base_task_id(task_id: str | None) -> str | None:
    if not task_id:
        return None
    if "_chunk" in task_id:
        return task_id.rsplit("_chunk", 1)[0]
    return task_id


def _extract_task_id(record: dict[str, Any]) -> str | None:
    if "task_id" in record:
        return record.get("task_id")
    if "id" in record:
        return record.get("id")
    meta = record.get("metadata")
    if isinstance(meta, dict):
        if "task_id" in meta:
            return meta.get("task_id")
        if "id" in meta:
            return meta.get("id")
    return None


def _normalize_messages(messages: list[dict]) -> list[dict]:
    if not messages:
        return []
    normalized = []
    if messages[0].get("role") == "system":
        normalized.append({"role": "system", "content": OSWORLD_SYSTEM_PROMPT})
        normalized.extend(messages[1:])
        return normalized
    return [{"role": "system", "content": OSWORLD_SYSTEM_PROMPT}] + messages


def _messages_hash(messages: list[dict]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _assistant_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _passes_action_parse(env: OSWorldEnvWrapper, messages: list[dict]) -> bool:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = _assistant_text(msg.get("content"))
        parsed = env._parse_action(content)
        if parsed.get("action_type") == "unknown" and not parsed.get("fallback_used"):
            return False
    return True


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _load_osworld_task_configs(osworld_repo: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for json_path in osworld_repo.rglob("*.json"):
        if json_path.name == "test_all.json":
            continue
        try:
            obj = json.loads(json_path.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        task_id = obj.get("id")
        instruction = obj.get("instruction")
        if not task_id or not instruction:
            continue
        base_id = _base_task_id(task_id)
        if base_id is None:
            continue
        configs[base_id] = obj
    return configs


def _detect_platform(config: dict[str, Any]) -> str:
    snapshot = str(config.get("snapshot", "")).lower()
    if snapshot in {"excel", "word", "powerpoint"}:
        return "windows"
    text_blob = json.dumps(config, ensure_ascii=True)
    if "C:\\\\" in text_blob or "C:/" in text_blob or "\\\\Users\\\\" in text_blob:
        return "windows"
    return "ubuntu"


def _load_distill_task_configs(parquet_paths: list[Path]) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for path in parquet_paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "task_config" not in df.columns:
            continue
        for _, row in df.iterrows():
            task_id = _base_task_id(row.get("task_id"))
            task_config = row.get("task_config")
            if not task_id or not isinstance(task_config, str):
                continue
            try:
                config_obj = json.loads(task_config)
            except json.JSONDecodeError:
                continue
            if task_id not in configs:
                configs[task_id] = config_obj
    return configs


def build_union_tasks(
    task_ids: set[str],
    distill_configs: dict[str, dict[str, Any]],
    osworld_configs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for task_id in sorted(task_ids):
        config = None
        source = "missing"
        if task_id in distill_configs:
            config = distill_configs[task_id]
            source = "distill_data"
        elif task_id in osworld_configs:
            config = osworld_configs[task_id]
            source = "osworld_repo"

        has_config = config is not None
        instruction = ""
        domain = ""
        config_payload = "{}"
        platform = "unknown"
        if config:
            config = dict(config)
            config["id"] = task_id
            instruction = config.get("instruction", "")
            domain = config.get("domain", "")
            if not domain and isinstance(config.get("related_apps"), list) and config["related_apps"]:
                domain = config["related_apps"][0]
            platform = _detect_platform(config)
            config_payload = json.dumps(config, ensure_ascii=False)

        rows.append(
            {
                "task_id": task_id,
                "prompt": task_id,
                "domain": domain,
                "instruction": instruction,
                "task_config": config_payload,
                "task_config_source": source,
                "has_task_config": has_config,
                "platform": platform,
            }
        )

    return pd.DataFrame(rows)


def build_replay_buffer(
    datasets: list[DatasetSpec],
    env: OSWorldEnvWrapper,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    stats = {
        "input_samples": 0,
        "kept_samples": 0,
        "dropped_parse": 0,
        "deduped": 0,
    }

    for ds in datasets:
        records = _load_jsonl(ds.path)
        for record in records:
            stats["input_samples"] += 1
            task_id = _base_task_id(_extract_task_id(record))
            if not task_id:
                continue
            messages = record.get("messages") or record.get("conversations") or []
            if not messages:
                continue
            normalized = _normalize_messages(messages)
            if not _passes_action_parse(env, normalized):
                stats["dropped_parse"] += 1
                continue
            msg_hash = _messages_hash(normalized)
            key = (task_id, msg_hash)
            payload = {
                "messages": normalized,
                "images": record.get("images", []),
                "reward": 1.0,
                "task_id": task_id,
                "num_steps": sum(1 for m in normalized if m.get("role") == "assistant"),
                "original_task_id": _extract_task_id(record),
                "source_dataset": ds.name,
                "quality_rank": ds.quality_rank,
            }

            if key not in dedup:
                dedup[key] = payload
                continue
            if ds.quality_rank > dedup[key]["quality_rank"]:
                dedup[key] = payload
                stats["deduped"] += 1

    merged = list(dedup.values())
    stats["kept_samples"] = len(merged)
    stats["per_dataset"] = dict(Counter(sample["source_dataset"] for sample in merged))
    return merged, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OSWorld union datasets")
    parser.add_argument(
        "--hf-root",
        type=str,
        default="/tmp/hf_verify",
        help="Root directory where HF datasets are downloaded",
    )
    parser.add_argument(
        "--osworld-repo",
        type=str,
        default="/tmp/OSWorld",
        help="Local OSWorld repo path for task_config backfill",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/osworld_union",
        help="Output directory for union artifacts",
    )
    args = parser.parse_args()

    hf_root = Path(args.hf_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_specs = [
        DatasetSpec(
            name="sft_curated_final",
            path=hf_root / "osworld_datasets" / "osworld-sft-curated-final" / "osworld_sft_curated_final.jsonl",
            quality_rank=5,
        ),
        DatasetSpec(
            name="sft_curated_v3",
            path=hf_root / "osworld_datasets" / "osworld-sft-curated-v3" / "osworld_sft_curated.jsonl",
            quality_rank=4,
        ),
        DatasetSpec(
            name="sft_hybrid",
            path=hf_root / "osworld_datasets" / "osworld-sft-hybrid" / "osworld_sft_hybrid.jsonl",
            quality_rank=3,
        ),
        DatasetSpec(
            name="combined_sft",
            path=hf_root / "osworld_datasets" / "osworld-combined-sft" / "osworld_combined_sft.jsonl",
            quality_rank=2,
        ),
        DatasetSpec(
            name="sft_clean",
            path=hf_root / "osworld_datasets" / "osworld-sft-clean" / "osworld_sft_clean.jsonl",
            quality_rank=1,
        ),
        DatasetSpec(
            name="bc_sft",
            path=hf_root / "osworld_datasets" / "osworld-bc-sft" / "bc_sft_train_vlm_split_fixed.jsonl",
            quality_rank=0,
        ),
    ]

    replay_sources = jsonl_specs
    exploratory_path = hf_root / "osworld_datasets" / "osworld-exploratory-rollouts" / "all_rollouts.jsonl"

    parquet_paths = [
        hf_root / "osworld_datasets" / "osworld-vlm-distill-data" / "train.parquet",
        hf_root / "osworld_datasets" / "osworld-vlm-distill-data" / "test.parquet",
        hf_root / "osworld_datasets" / "osworld-variance-subset" / "train_variance_subset.parquet",
        hf_root / "osworld-merged-replay" / "osworld_gspo_tasks.parquet",
        hf_root / "osworld-merged-replay" / "osworld_gspo_tasks_aligned.parquet",
    ]

    union_task_ids: set[str] = set()
    for spec in replay_sources:
        records = _load_jsonl(spec.path)
        for record in records:
            task_id = _base_task_id(_extract_task_id(record))
            if task_id:
                union_task_ids.add(task_id)
    if exploratory_path.exists():
        records = _load_jsonl(exploratory_path)
        for record in records:
            task_id = _base_task_id(_extract_task_id(record))
            if task_id:
                union_task_ids.add(task_id)

    for path in parquet_paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "task_id" in df.columns:
            ids = df["task_id"].tolist()
        elif "id" in df.columns:
            ids = df["id"].tolist()
        else:
            ids = []
        for tid in ids:
            base_id = _base_task_id(tid)
            if base_id:
                union_task_ids.add(base_id)

    osworld_configs = _load_osworld_task_configs(Path(args.osworld_repo))
    distill_configs = _load_distill_task_configs(
        [
            hf_root / "osworld_datasets" / "osworld-vlm-distill-data" / "train.parquet",
            hf_root / "osworld_datasets" / "osworld-vlm-distill-data" / "test.parquet",
            hf_root / "osworld_datasets" / "osworld-variance-subset" / "train_variance_subset.parquet",
        ]
    )

    union_df = build_union_tasks(union_task_ids, distill_configs, osworld_configs)
    union_all_path = output_dir / "osworld_tasks_all.parquet"
    union_df.to_parquet(union_all_path, index=False)

    env = OSWorldEnvWrapper(OSWorldEnvConfig(), task_config={})
    replay_samples, replay_stats = build_replay_buffer(replay_sources, env)
    replay_path = output_dir / "osworld_replay_train.jsonl"
    with replay_path.open("w") as f:
        for sample in replay_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    ready_df = union_df[(union_df["has_task_config"]) & (union_df["platform"] != "windows")].reset_index(drop=True)
    replay_task_ids = {sample["task_id"] for sample in replay_samples}
    replay_ready_df = ready_df[ready_df["task_id"].isin(replay_task_ids)].reset_index(drop=True)

    union_ready_path = output_dir / "osworld_tasks_train.parquet"
    replay_ready_df.to_parquet(union_ready_path, index=False)

    platform_counts = dict(Counter(union_df["platform"].tolist()))
    stats = {
        "union_task_ids": len(union_task_ids),
        "task_config_distill": len(distill_configs),
        "task_config_osworld": len(osworld_configs),
        "union_all_rows": len(union_df),
        "union_ready_rows": len(replay_ready_df),
        "replay_ready_rows": len(replay_ready_df),
        "platform_counts": platform_counts,
        "replay_stats": replay_stats,
    }
    stats_path = output_dir / "osworld_train_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"wrote {union_all_path}")
    print(f"wrote {union_ready_path}")
    print(f"wrote {replay_path}")
    print(f"wrote {stats_path}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
