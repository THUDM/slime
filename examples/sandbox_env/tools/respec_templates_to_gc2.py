#!/usr/bin/env python3
"""Batch re-spec G_C4 templates to G_C2 via from_template().

For each row in the source manifest, build a new template aliased
`swe-rebench-gc2-{instance_id_slug}` that descends from the existing G_C4 alias
in the row's `inspire_template` field. NO additional install layers — pure
spec re-stamp (~13s per template, cache-hit on parent).

Idempotent: rows already in the success manifest (or whose new alias is
already `ready` in Inspire) are skipped. Failures go to the failure manifest;
re-runs will retry them.
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(
    0,
    "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/czxs253130081/.local/lib/python3.10/site-packages",
)

from dotenv import load_dotenv
load_dotenv()

from inspire_sandbox import SandboxSpecCode, Template  # noqa: E402
from inspire_sandbox.api.client.api.templates import (  # noqa: E402
    get_v1_templates_aliases_alias,
    get_v_1_templates_template_id,
)
from inspire_sandbox.api.client_sync import get_api_client  # noqa: E402
from inspire_sandbox.connection_config import ConnectionConfig  # noqa: E402


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "unknown"


class JsonlAppender:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def alias_is_ready(api_client, alias: str) -> bool:
    """Return True iff the alias exists AND its latest build is `ready`."""
    try:
        r = get_v1_templates_aliases_alias.sync_detailed(alias, client=api_client)
        if int(r.status_code) != 200:
            return False
        template_id = str(getattr(r.parsed, "template_id", "") or "")
        if not template_id:
            return False
        t = get_v_1_templates_template_id.sync_detailed(template_id, client=api_client, limit=5)
        if int(t.status_code) != 200:
            return False
        builds = list(getattr(t.parsed, "builds", []) or [])
        if not builds:
            return False
        latest = max(
            builds,
            key=lambda b: getattr(b, "updated_at", None) or getattr(b, "created_at", None),
        )
        status = getattr(getattr(latest, "status", ""), "value", str(getattr(latest, "status", "")))
        return status == "ready"
    except Exception:
        return False


def respec_one(row: dict, api_client, alias_format: str,
               success: JsonlAppender, failure: JsonlAppender) -> tuple[str, str]:
    iid = row.get("instance_id") or ""
    parent = row.get("inspire_template") or row.get("template_alias") or ""
    new_alias = alias_format.format(instance_id_slug=slugify(iid))
    if not iid or not parent:
        failure.append({"instance_id": iid, "reason": "missing instance_id or parent alias", "new_alias": new_alias})
        return iid, "fail_missing_parent"

    if alias_is_ready(api_client, new_alias):
        out = dict(row)
        out["inspire_template"] = new_alias
        out["template_alias"] = new_alias
        out["respec_status"] = "skipped_already_ready"
        out["respec_parent"] = parent
        out["respec_spec"] = "G_C2"
        success.append(out)
        return iid, "skip_ready"

    t0 = time.monotonic()
    try:
        builder = Template().from_template(parent)
        template = builder.set_start_cmd("echo swe_rebench_template_smoke_ok", ":")
        Template.build(
            template,
            new_alias,
            spec_code=SandboxSpecCode.G_C2,
            skip_cache=False,
        )
        out = dict(row)
        out["inspire_template"] = new_alias
        out["template_alias"] = new_alias
        out["respec_build_seconds"] = round(time.monotonic() - t0, 1)
        out["respec_parent"] = parent
        out["respec_spec"] = "G_C2"
        success.append(out)
        return iid, f"ok_{out['respec_build_seconds']}s"
    except Exception as e:
        failure.append({
            "instance_id": iid,
            "parent_alias": parent,
            "new_alias": new_alias,
            "reason": f"{type(e).__name__}: {e}",
            "respec_build_seconds": round(time.monotonic() - t0, 1),
        })
        return iid, f"fail_{type(e).__name__}"


def load_processed(success_path: Path, failure_path: Path) -> set[str]:
    """Return instance_ids already terminal (success-recorded or recently failed)."""
    done = set()
    for path in (success_path,):
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                iid = d.get("instance_id")
                if iid:
                    done.add(iid)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-spec G_C4 templates to G_C2 in batch.")
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--success-manifest", required=True)
    ap.add_argument("--failure-manifest", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument(
        "--alias-format",
        default="swe-rebench-gc2-{instance_id_slug}",
        help="Output alias format (uses {instance_id_slug}).",
    )
    ap.add_argument(
        "--require-status",
        default="ready",
        help="If a row has a 'status' field, only process when it equals this value "
        "(use empty string to disable). Rows without 'status' are always considered.",
    )
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()

    success_path = Path(args.success_manifest)
    failure_path = Path(args.failure_manifest)

    done = load_processed(success_path, failure_path)

    rows: list[dict] = []
    seen: set[str] = set()
    with open(args.source_manifest) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            iid = d.get("instance_id")
            if not iid or iid in seen:
                continue
            seen.add(iid)
            if iid in done:
                continue
            if not (d.get("inspire_template") or d.get("template_alias")):
                continue
            if args.require_status:
                row_status = d.get("status")
                if row_status and row_status != args.require_status:
                    continue
            rows.append(d)
    if args.limit > 0:
        rows = rows[: args.limit]

    print(
        f"source={args.source_manifest}\n"
        f"already done={len(done)} pending={len(rows)} workers={args.workers}\n"
        f"alias_format={args.alias_format}\n"
        f"success_out={args.success_manifest}\n"
        f"failure_out={args.failure_manifest}",
        flush=True,
    )
    if not rows:
        print("nothing to do")
        return

    api_client = get_api_client(ConnectionConfig(), require_api_key=True, require_access_token=False)
    success = JsonlAppender(success_path)
    failure = JsonlAppender(failure_path)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    counts_lock = threading.Lock()
    t_start = time.monotonic()

    def categorize(status: str) -> str:
        if status.startswith("ok"):
            return "ok"
        if status.startswith("skip"):
            return "skip"
        return "fail"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(respec_one, r, api_client, args.alias_format, success, failure) for r in rows]
        for fut in as_completed(futures):
            try:
                _, status = fut.result()
            except Exception as e:
                print(f"WORKER_EXC: {type(e).__name__}: {e}", flush=True)
                with counts_lock:
                    counts["fail"] += 1
                    done_n = sum(counts.values())
            else:
                cat = categorize(status)
                with counts_lock:
                    counts[cat] += 1
                    done_n = sum(counts.values())
            if done_n % args.progress_every == 0 or done_n == len(rows):
                elapsed = time.monotonic() - t_start
                rate = done_n / max(elapsed, 1.0)
                eta_min = (len(rows) - done_n) / max(rate, 0.01) / 60.0
                print(
                    f"[{done_n}/{len(rows)}] ok={counts['ok']} skip={counts['skip']} "
                    f"fail={counts['fail']} rate={rate:.2f}/s elapsed={elapsed/60:.1f}m "
                    f"eta={eta_min:.1f}m",
                    flush=True,
                )

    print(f"DONE: ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}", flush=True)


if __name__ == "__main__":
    main()
