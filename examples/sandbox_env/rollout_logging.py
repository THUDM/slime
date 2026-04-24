from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def build_batch_log_dir(log_root: str | None) -> Path | None:
    if not log_root:
        return None
    return Path(log_root) / "current_batch"


def prepare_batch_log_dir(log_root: str | None) -> Path | None:
    batch_log_dir = build_batch_log_dir(log_root)
    if batch_log_dir is None:
        return None
    shutil.rmtree(batch_log_dir, ignore_errors=True)
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    return batch_log_dir


def build_sample_log_dir(log_root: str | None, *, sample_idx: int) -> Path | None:
    batch_log_dir = build_batch_log_dir(log_root)
    if batch_log_dir is None:
        return None
    return batch_log_dir / f"sample_{sample_idx}"


def build_live_sandbox_log_path(log_root: str | None, *, sample_idx: int) -> Path | None:
    sample_log_dir = build_sample_log_dir(log_root, sample_idx=sample_idx)
    if sample_log_dir is None:
        return None
    sandbox_log_dir = sample_log_dir / "sandbox"
    sandbox_log_dir.mkdir(parents=True, exist_ok=True)
    return sandbox_log_dir / "agent_output.log"


def build_eval_log_path(log_root: str | None, *, sample_idx: int) -> Path | None:
    sample_log_dir = build_sample_log_dir(log_root, sample_idx=sample_idx)
    if sample_log_dir is None:
        return None
    sandbox_log_dir = sample_log_dir / "sandbox"
    sandbox_log_dir.mkdir(parents=True, exist_ok=True)
    return sandbox_log_dir / "eval_output.log"


def truncate_text(value: str | None, limit: int) -> str:
    text = value or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit]


def write_sample_artifacts_snapshot(
    *,
    log_root: str | None,
    rollout_id: int,
    sample_idx: int,
    sample_prompt: str,
    metadata: dict[str, Any],
    extra_metadata: dict[str, Any],
    turn_responses: list[str],
    trajectory: list[dict[str, Any]],
    final_messages: list[dict[str, Any]] | None,
    last_response_payload: str | None,
) -> str | None:
    sample_log_dir = build_sample_log_dir(log_root, sample_idx=sample_idx)
    if sample_log_dir is None:
        return None

    sample_log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rollout_id": rollout_id,
        "sample_idx": sample_idx,
        "sandbox_id": extra_metadata.get("sandbox_id"),
        "instance_id": metadata.get("instance_id"),
        "repo": metadata.get("repo"),
        "local_image_name": metadata.get("local_image_name"),
        "prompt": sample_prompt,
        "extra_metadata": extra_metadata,
        "turn_responses": turn_responses,
        "trajectory": trajectory,
        "final_messages": final_messages,
        "last_response_payload": last_response_payload,
    }
    (sample_log_dir / "sample_artifacts.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(sample_log_dir)
