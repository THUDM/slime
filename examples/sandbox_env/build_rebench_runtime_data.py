#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TASKS_JSON = AVALANCHE_ROOT / "data" / "raw_data" / "single" / "swe_rebench_v2" / "data" / "train-00000-of-00001.json"
DEFAULT_TEMPLATE_MANIFEST = (
    AVALANCHE_ROOT / "data" / "raw_data" / "single" / "swe_rebench_v2" / "data" / "prefetch_image_template_success.jsonl"
)
def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [row for row in data if isinstance(row, dict)]


def _load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    manifest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = str(row.get("instance_id") or "").strip()
            if instance_id:
                manifest[instance_id] = row
    return manifest


def _normalize_optional_path(value: str) -> Path | None:
    value = (value or "").strip()
    if not value:
        return None
    return Path(value)


def _has_resolved_image_user(entry: dict[str, Any] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    return bool(str(entry.get("docker_image_default_user") or "").strip())


def _has_resolved_image_env(entry: dict[str, Any] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    env_value = entry.get("docker_image_env")
    return isinstance(env_value, dict) and len(env_value) > 0


def _repo_workdir(repo: str) -> str:
    parts = str(repo or "").split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Invalid repo value: {repo!r}")
    return f"/{parts[1]}"


def _build_prompt(row: dict[str, Any]) -> str:
    parts = [str(row.get("problem_statement") or "").strip()]

    pr_description = str(row.get("pr_description") or "").strip()
    if pr_description:
        parts.extend(["", "Additional Context:", pr_description])

    interface = str(row.get("interface") or "").strip()
    if interface and interface != "No new interfaces are introduced.":
        parts.extend(["", "Interface Notes:", interface])

    parts.extend(
        [
            "",
            "Evaluation Constraints:",
            (
                "Do not modify test files unless absolutely necessary. The evaluation pipeline will apply a test patch "
                "after your code changes to check correctness."
            ),
            (
                "If you temporarily edit any test file for local investigation, restore it to its original state before "
                "you finish."
            ),
            (
                "Prefer implementing the fix in production code only. Leave issue-specific test additions or updates to "
                "the evaluation patch."
            ),
        ]
    )

    return "\n".join(part for part in parts if part is not None).strip()


def _build_runtime_row(
    row: dict[str, Any],
    *,
    conversation_prompt: bool,
    consumable_template_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt_text = _build_prompt(row)
    if conversation_prompt:
        prompt: Any = [{"role": "user", "content": prompt_text}]
    else:
        prompt = prompt_text

    image_name = str(row.get("image_name") or "").strip()
    template_alias = ""
    default_user = ""
    default_user_raw = ""
    default_user_checked_at = ""
    default_env: dict[str, str] = {}
    default_env_raw = ""
    default_env_checked_at = ""
    if consumable_template_entry is not None:
        template_alias = str(
            consumable_template_entry.get("inspire_template")
            or consumable_template_entry.get("template_alias")
            or ""
        ).strip()
        default_user = str(consumable_template_entry.get("docker_image_default_user") or "").strip()
        default_user_raw = str(consumable_template_entry.get("docker_image_default_user_raw") or "")
        default_user_checked_at = str(
            consumable_template_entry.get("docker_image_default_user_checked_at") or ""
        ).strip()
        env_value = consumable_template_entry.get("docker_image_env")
        if isinstance(env_value, dict):
            default_env = {str(k): str(v) for k, v in env_value.items() if str(k).strip()}
        default_env_raw = str(consumable_template_entry.get("docker_image_env_raw") or "")
        default_env_checked_at = str(
            consumable_template_entry.get("docker_image_env_checked_at") or ""
        ).strip()

    metadata = {
        "domain": "swe",
        "source_name": "swe_rebench_v2",
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "repo_workdir": _repo_workdir(str(row["repo"])),
        "base_commit": row["base_commit"],
        "image_name": image_name,
        "local_image_name": image_name,
        "inspire_template": template_alias,
        "docker_image_default_user": default_user,
        "docker_image_default_user_raw": default_user_raw,
        "docker_image_default_user_checked_at": default_user_checked_at,
        "docker_image_env": default_env,
        "docker_image_env_raw": default_env_raw,
        "docker_image_env_checked_at": default_env_checked_at,
        "problem_statement": row.get("problem_statement", "") or "",
        "pr_description": row.get("pr_description", "") or "",
        "interface": row.get("interface", "") or "",
        "language": row.get("language", "") or "",
        "license": row.get("license", "") or "",
        "patch": row.get("patch", "") or "",
        "test_patch": row.get("test_patch", "") or "",
        "install_config": row.get("install_config") or {},
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS") or [],
        "PASS_TO_PASS": row.get("PASS_TO_PASS") or [],
        "eval_mode": "rebench_v2",
    }
    return {
        "prompt": prompt,
        "label": row.get("patch", "") or "",
        "metadata": metadata,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime jsonl files for SWE-rebench training.")
    parser.add_argument("--tasks-json", default=str(DEFAULT_TASKS_JSON))
    parser.add_argument("--train-dest", required=True)
    parser.add_argument("--val-dest", default="")
    parser.add_argument("--train-max", type=int, default=1024)
    parser.add_argument("--val-max", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--consumable-template-manifest", default=str(DEFAULT_TEMPLATE_MANIFEST))
    parser.add_argument("--require-consumable-templates", action="store_true")
    parser.add_argument("--conversation-prompt", action="store_true")
    args = parser.parse_args()

    rows = _load_rows(Path(args.tasks_json))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    train_rows = rows if args.train_max < 0 else rows[: args.train_max]
    val_start = len(train_rows)
    if args.val_max < 0:
        val_rows = rows[val_start:]
    elif args.val_max > 0:
        val_rows = rows[val_start : val_start + args.val_max]
    else:
        val_rows = []

    consumable_manifest = _load_manifest(_normalize_optional_path(args.consumable_template_manifest))
    if args.require_consumable_templates:
        before_train = len(train_rows)
        before_val = len(val_rows)
        train_rows = [
            row
            for row in train_rows
            if _has_resolved_image_user(consumable_manifest.get(row["instance_id"]))
            and _has_resolved_image_env(consumable_manifest.get(row["instance_id"]))
        ]
        val_rows = [
            row
            for row in val_rows
            if _has_resolved_image_user(consumable_manifest.get(row["instance_id"]))
            and _has_resolved_image_env(consumable_manifest.get(row["instance_id"]))
        ]
        print(
            f"filtered by consumable template manifest with docker_image_default_user+docker_image_env: "
            f"train {before_train}->{len(train_rows)} "
            f"val {before_val}->{len(val_rows)}"
        )

    built_train = [
        _build_runtime_row(
            row,
            conversation_prompt=args.conversation_prompt,
            consumable_template_entry=consumable_manifest.get(row["instance_id"]),
        )
        for row in train_rows
    ]
    built_val = [
        _build_runtime_row(
            row,
            conversation_prompt=args.conversation_prompt,
            consumable_template_entry=consumable_manifest.get(row["instance_id"]),
        )
        for row in val_rows
    ]

    _write_jsonl(Path(args.train_dest), built_train)
    print(f"train -> {args.train_dest} ({len(built_train)} rows)")

    if args.val_dest:
        _write_jsonl(Path(args.val_dest), built_val)
        print(f"val -> {args.val_dest} ({len(built_val)} rows)")


if __name__ == "__main__":
    main()
