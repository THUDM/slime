#!/usr/bin/env python3
"""Populate effective docker runtime users into prefetch success manifest.

This script resolves the user by actually entering each docker image the same
way official eval does: `docker run ... /bin/bash -lc ...`.

It writes the resolved user back into the success manifest in-place, supporting:
- parallel execution
- resume / skip already-written rows
- force rewrite of all existing user fields
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = AVALANCHE_ROOT / "data" / "raw_data" / "single" / "swe_rebench_v2" / "data"
DEFAULT_SUCCESS = DEFAULT_OUT_DIR / "prefetch_image_template_success.jsonl"
DEFAULT_FAILURE = DEFAULT_OUT_DIR / "prefetch_image_template_failure.jsonl"
DEFAULT_DOCKER_USER_FIELD = "docker_image_default_user"
DEFAULT_DOCKER_USER_RAW_FIELD = "docker_image_default_user_raw"
DEFAULT_DOCKER_USER_CHECKED_AT_FIELD = "docker_image_default_user_checked_at"
DEFAULT_DOCKER_ENV_FIELD = "docker_image_env"
DEFAULT_DOCKER_ENV_RAW_FIELD = "docker_image_env_raw"
DEFAULT_DOCKER_ENV_CHECKED_AT_FIELD = "docker_image_env_checked_at"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _rewrite_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _append_jsonl_row(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ensure_image_local(image_name: str) -> None:
    inspect = subprocess.run(
        ["docker", "image", "inspect", image_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        return

    pull = subprocess.run(
        ["docker", "pull", image_name],
        check=False,
        capture_output=True,
        text=True,
    )
    if pull.returncode != 0:
        raise RuntimeError(
            f"docker pull failed for {image_name!r}: {(pull.stderr or pull.stdout or '').strip()}"
        )


def _run_image_user_probe(image_name: str) -> tuple[str, str]:
    _ensure_image_local(image_name)

    probe = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            image_name,
            "/bin/bash",
            "-lc",
            "whoami; id -un; id -u; id -g",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"docker run user probe failed for {image_name!r}: {(probe.stderr or probe.stdout or '').strip()}"
        )

    lines = [line.strip() for line in (probe.stdout or "").splitlines() if line.strip()]
    if len(lines) < 4:
        pull = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Config.User}}", image_name],
            check=False,
            capture_output=True,
            text=True,
        )
        raw_cfg = (pull.stdout or "").strip() if pull.returncode == 0 else ""
        raise RuntimeError(
            f"unexpected docker run user probe output for {image_name!r}: "
            f"stdout={(probe.stdout or '').strip()!r} stderr={(probe.stderr or '').strip()!r} "
            f"config_user={raw_cfg!r}"
        )

    whoami_name, id_un_name, uid_value, gid_value = lines[:4]
    effective_user = whoami_name or id_un_name or "root"
    raw_user = json.dumps(
        {
            "whoami": whoami_name,
            "id_un": id_un_name,
            "uid": uid_value,
            "gid": gid_value,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return effective_user, raw_user


def _run_image_env_probe(image_name: str) -> tuple[dict[str, str], str]:
    _ensure_image_local(image_name)

    probe = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            image_name,
            "/bin/bash",
            "-lc",
            "env -0",
        ],
        check=False,
        capture_output=True,
        text=False,
    )
    if probe.returncode != 0:
        stderr = probe.stderr.decode("utf-8", errors="replace") if probe.stderr else ""
        stdout = probe.stdout.decode("utf-8", errors="replace") if probe.stdout else ""
        raise RuntimeError(
            f"docker run env probe failed for {image_name!r}: {(stderr or stdout).strip()}"
        )

    raw_bytes = probe.stdout or b""
    env_map: dict[str, str] = {}
    for entry in raw_bytes.split(b"\x00"):
        if not entry or b"=" not in entry:
            continue
        key_raw, value_raw = entry.split(b"=", 1)
        key = key_raw.decode("utf-8", errors="replace").strip()
        if not key:
            continue
        env_map[key] = value_raw.decode("utf-8", errors="replace")
    return env_map, json.dumps(env_map, ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate docker image default users into success manifest.")
    parser.add_argument("--success-manifest", default=str(DEFAULT_SUCCESS))
    parser.add_argument("--failure-manifest", default=str(DEFAULT_FAILURE))
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--max-instances", type=int, default=-1)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    success_manifest = Path(args.success_manifest)
    failure_manifest = Path(args.failure_manifest)
    failure_manifest.parent.mkdir(parents=True, exist_ok=True)

    success_rows = _load_jsonl_rows(success_manifest)
    if not success_rows:
        raise SystemExit(f"No success rows found in {success_manifest}")

    requested_ids = {value.strip() for value in args.instance_id if value.strip()}
    manifest_rows = list(success_rows)
    instance_to_index = {
        str(row.get("instance_id") or "").strip(): idx
        for idx, row in enumerate(manifest_rows)
        if str(row.get("instance_id") or "").strip()
    }

    if args.force:
        for row in manifest_rows:
            row.pop(DEFAULT_DOCKER_USER_FIELD, None)
            row.pop(DEFAULT_DOCKER_USER_RAW_FIELD, None)
            row.pop(DEFAULT_DOCKER_USER_CHECKED_AT_FIELD, None)
            row.pop(DEFAULT_DOCKER_ENV_FIELD, None)
            row.pop(DEFAULT_DOCKER_ENV_RAW_FIELD, None)
            row.pop(DEFAULT_DOCKER_ENV_CHECKED_AT_FIELD, None)
        _rewrite_jsonl_rows(success_manifest, manifest_rows)

    candidates: list[dict[str, Any]] = []
    for row in manifest_rows:
        instance_id = str(row.get("instance_id") or "").strip()
        if not instance_id:
            continue
        if requested_ids and instance_id not in requested_ids:
            continue
        has_user = str(row.get(DEFAULT_DOCKER_USER_FIELD) or "").strip()
        has_env = row.get(DEFAULT_DOCKER_ENV_FIELD)
        if not args.force and has_user and isinstance(has_env, dict) and len(has_env) > 0:
            _emit_json(
                {
                    "status": "skip_existing",
                    "instance_id": instance_id,
                    "image_name": row.get("image_name"),
                    "default_user": row.get(DEFAULT_DOCKER_USER_FIELD),
                    "env_keys": len(has_env),
                }
            )
            continue
        candidates.append(row)

    if args.max_instances >= 0:
        candidates = candidates[: args.max_instances]

    if not candidates:
        _emit_json({"status": "done", "message": "No candidates need inspection."})
        return

    success_lock = threading.Lock()
    failure_lock = threading.Lock()

    def _inspect_one(row: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(row.get("instance_id") or "").strip()
        image_name = str(row.get("image_name") or "").strip()
        if not image_name:
            raise RuntimeError(f"missing image_name for instance_id={instance_id!r}")
        default_user, raw_user = _run_image_user_probe(image_name)
        env_map, raw_env = _run_image_env_probe(image_name)
        checked_at = _utc_now_iso()
        return {
            "instance_id": instance_id,
            "image_name": image_name,
            DEFAULT_DOCKER_USER_FIELD: default_user,
            DEFAULT_DOCKER_USER_RAW_FIELD: raw_user,
            DEFAULT_DOCKER_USER_CHECKED_AT_FIELD: checked_at,
            DEFAULT_DOCKER_ENV_FIELD: env_map,
            DEFAULT_DOCKER_ENV_RAW_FIELD: raw_env,
            DEFAULT_DOCKER_ENV_CHECKED_AT_FIELD: checked_at,
        }

    _emit_json(
        {
            "status": "start",
            "success_manifest": str(success_manifest),
            "failure_manifest": str(failure_manifest),
            "candidates": len(candidates),
            "parallelism": max(1, args.parallelism),
        }
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as executor:
        future_to_instance = {
            executor.submit(_inspect_one, row): str(row.get("instance_id") or "").strip()
            for row in candidates
        }
        for future in concurrent.futures.as_completed(future_to_instance):
            instance_id = future_to_instance[future]
            try:
                result = future.result()
            except Exception as exc:
                failure_row = {
                    "instance_id": instance_id,
                    "source_name": "swe_rebench_v2",
                    "status": "default_user_error",
                    "reason": str(exc),
                    "checked_at": _utc_now_iso(),
                }
                _append_jsonl_row(failure_manifest, failure_row, failure_lock)
                _emit_json({"status": "failed", "instance_id": instance_id, "error": str(exc)})
                continue

            row_index = instance_to_index.get(instance_id)
            if row_index is None:
                failure_row = {
                    "instance_id": instance_id,
                    "source_name": "swe_rebench_v2",
                    "status": "default_user_error",
                    "reason": "instance_id disappeared from success manifest before write-back",
                    "checked_at": _utc_now_iso(),
                }
                _append_jsonl_row(failure_manifest, failure_row, failure_lock)
                _emit_json({"status": "failed", "instance_id": instance_id, "error": failure_row["reason"]})
                continue

            with success_lock:
                manifest_rows[row_index][DEFAULT_DOCKER_USER_FIELD] = result[DEFAULT_DOCKER_USER_FIELD]
                manifest_rows[row_index][DEFAULT_DOCKER_USER_RAW_FIELD] = result[DEFAULT_DOCKER_USER_RAW_FIELD]
                manifest_rows[row_index][DEFAULT_DOCKER_USER_CHECKED_AT_FIELD] = result[
                    DEFAULT_DOCKER_USER_CHECKED_AT_FIELD
                ]
                manifest_rows[row_index][DEFAULT_DOCKER_ENV_FIELD] = result[DEFAULT_DOCKER_ENV_FIELD]
                manifest_rows[row_index][DEFAULT_DOCKER_ENV_RAW_FIELD] = result[DEFAULT_DOCKER_ENV_RAW_FIELD]
                manifest_rows[row_index][DEFAULT_DOCKER_ENV_CHECKED_AT_FIELD] = result[
                    DEFAULT_DOCKER_ENV_CHECKED_AT_FIELD
                ]
                _rewrite_jsonl_rows(success_manifest, manifest_rows)

            _emit_json(
                {
                    "status": "ok",
                    "instance_id": instance_id,
                    "image_name": result["image_name"],
                    "default_user": result[DEFAULT_DOCKER_USER_FIELD],
                    "raw_user": result[DEFAULT_DOCKER_USER_RAW_FIELD],
                    "env_keys": len(result[DEFAULT_DOCKER_ENV_FIELD]),
                }
            )


if __name__ == "__main__":
    main()
