#!/usr/bin/env python3
"""End-to-end mini-rollout smoke test against a real Inspire sandbox.

This is the direct Inspire safety net for the SWE sandbox path.  It starts an
existing template with ``inspire_sandbox``, optionally checks the
selected scaffold and sandbox-local ``wstunnel`` binary, applies the canonical
gold patch, then runs the rebench evaluator.

What's covered:

1. ``sandbox_runtime.decode_swe_rollout_config`` reads SWE_* env vars.
2. ``Sandbox.create`` starts the manifest template directly.
3. optional ``--scaffold-preflight`` checks absolute runtime paths.
4. gold patch + rebench eval produce a positive reward.

Pass criterion: ``reward > 0.0`` (gold patch should solve the task).

Run with::

    python tools/mini_rollout_test.py [--instance-id <id>]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

_TOOLS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _TOOLS_DIR.parent
_SLIME_ROOT = _PACKAGE_DIR.parents[1]
sys.path.insert(0, str(_SLIME_ROOT))

AVALANCHE_ROOT = Path("/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche")
INSPIRE_SDK = AVALANCHE_ROOT / ".local" / "share" / "inspire_sandbox_site_packages"
for p in (str(INSPIRE_SDK),):
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# Default test data: rebench v2 manifest + sample_5 task json.
DEFAULT_PREFETCH = (
    AVALANCHE_ROOT / "zf_workspace" / "eval" / "data" / "swe_rebench_v2" / "data"
    / "prefetch_image_template_success.jsonl"
)
DEFAULT_TASKS = (
    AVALANCHE_ROOT / "zf_workspace" / "eval" / "data" / "swe_rebench_v2" / "data"
    / "train-00000-of-00001.sample_5.json"
)


def _load_jsonl_or_json(path: Path) -> list[dict]:
    """Load both list-JSON and JSONL formats transparently."""
    text = path.read_text(encoding="utf-8")
    head = text.lstrip()[:1]
    if head == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _select_sample(prefetch_path: Path, tasks_path: Path, instance_id: str | None) -> dict:
    """Pick one task that has both a built template AND populated user/env."""
    prefetch_rows = _load_jsonl_or_json(prefetch_path)
    tasks = _load_jsonl_or_json(tasks_path)

    prefetch_by_id = {
        str(r.get("instance_id") or ""): r for r in prefetch_rows
        if r.get("status") == "ready"
        and r.get("docker_image_default_user")
        and isinstance(r.get("docker_image_env"), dict)
    }
    tasks_by_id = {str(t.get("instance_id") or ""): t for t in tasks}

    common = sorted(set(prefetch_by_id) & set(tasks_by_id))
    if not common:
        raise RuntimeError("no task overlaps between prefetch manifest and task data")

    chosen = instance_id if instance_id else common[0]
    if chosen not in prefetch_by_id:
        raise RuntimeError(
            f"instance_id={chosen!r} not in prefetch manifest (or missing user/env). "
            f"Available overlap: {common[:5]}..."
        )
    if chosen not in tasks_by_id:
        raise RuntimeError(f"instance_id={chosen!r} not in tasks data")

    # Merge: prefetch wins for build/template fields; task fields fill in patch/tests.
    merged = {**tasks_by_id[chosen], **prefetch_by_id[chosen]}
    merged.setdefault("local_image_name", merged.get("image_name"))
    repo = str(merged.get("repo") or "")
    if repo and not merged.get("repo_workdir"):
        merged["repo_workdir"] = f"/{repo.split('/', 1)[1]}"
    return merged


async def _apply_gold_patch(sandbox, metadata: dict, *, user: str | None, wait_timeout: int, log) -> None:
    """Write the canonical ``patch`` field into the sandbox workdir and apply it."""
    from examples.sandbox_env.rebench import resolve_rebench_workdir
    from examples.sandbox_env.sandbox_runtime import run_sandbox_command, write_sandbox_file

    workdir = resolve_rebench_workdir(metadata)
    gold_patch = str(metadata.get("patch") or "")
    if not gold_patch.strip():
        raise RuntimeError("manifest has no 'patch' field — can't simulate a gold rollout")
    if not gold_patch.endswith("\n"):
        # ``git apply`` rejects patches without a trailing newline as "corrupt".
        gold_patch += "\n"

    remote = "/tmp/gold.patch"
    await write_sandbox_file(sandbox, remote, gold_patch.encode("utf-8"), user=user)
    apply_script = (
        f"cd {shlex.quote(workdir)} && "
        f"git apply --whitespace=nowarn --recount {shlex.quote(remote)}"
    )
    result = await run_sandbox_command(
        sandbox,
        f"bash -lc {shlex.quote(apply_script)}",
        timeout=wait_timeout,
        user=user,
        cwd=workdir,
        log=log,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"git apply gold patch failed: {result.output.strip()[:500]}")


async def main() -> int:
    """CLI entry point: runs a gold-patch mini rollout against one real Inspire sandbox."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefetch-manifest", default=str(DEFAULT_PREFETCH))
    parser.add_argument("--tasks-data", default=str(DEFAULT_TASKS))
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--keep-sandbox", action="store_true", help="Skip stop() at end (for debugging).")
    parser.add_argument(
        "--scaffold-preflight",
        action="store_true",
        help="Also verify selected scaffold and sandbox-local wstunnel readiness.",
    )
    args = parser.parse_args()

    metadata = _select_sample(Path(args.prefetch_manifest), Path(args.tasks_data), args.instance_id)
    instance_id = metadata["instance_id"]
    print(f"[setup] instance_id={instance_id}")
    print(f"[setup] image={metadata['image_name']}")
    print(f"[setup] template={metadata['inspire_template']}")
    print(f"[setup] gold_patch={len(str(metadata.get('patch') or ''))}B")
    print(f"[setup] test_patch={len(str(metadata.get('test_patch') or ''))}B")
    print(f"[setup] FAIL_TO_PASS={len(metadata.get('FAIL_TO_PASS') or [])}")

    from agentic_protocol.command_factory.registry import resolve_agent_command_factory
    from examples.sandbox_env.rebench import (
        prepare_workspace,
        resolve_template_alias,
        run_rebench_eval,
        sandbox_default_user,
    )
    from examples.sandbox_env.sandbox_runtime import (
        LiveLog,
        build_eval_log_path,
        build_live_sandbox_log_path,
        build_sandbox_envs,
        create_sandbox_with_retry,
        decode_swe_rollout_config,
        preflight_scaffold,
    )

    log_root = Path("/tmp/mini_rollout_logs")
    log_root.mkdir(exist_ok=True)
    cfg = decode_swe_rollout_config(SimpleNamespace(swe_log_root=str(log_root)))
    print(f"[setup] harness={cfg.agent_harness} protocol_root={cfg.protocol_root}")
    print(f"[setup] wstunnel={cfg.wstunnel_bin}")

    agent_log = build_live_sandbox_log_path(cfg.log_root, sample_idx=0)
    eval_log = build_eval_log_path(cfg.log_root, sample_idx=0)
    print(f"[setup] agent_log={agent_log}\n[setup] eval_log={eval_log}")

    prompt = str(metadata.get("problem_statement") or "")
    live_log = LiveLog(agent_log)
    user = sandbox_default_user(metadata)

    sandbox = None
    rc = 0
    try:
        print("\n=== start sandbox ===")
        sandbox = await create_sandbox_with_retry(
            template_alias=resolve_template_alias(metadata),
            timeout=cfg.startup_timeout,
            envs=build_sandbox_envs(metadata, cfg, prompt),
            retry_times=cfg.sandbox_start_retry_times,
            retry_interval=cfg.sandbox_start_retry_interval,
        )
        print(f"  sandbox_id={getattr(sandbox, 'sandbox_id', '')}")
        assert sandbox is not None and sandbox.sandbox_id

        if args.scaffold_preflight:
            print("\n=== scaffold preflight ===")
            factory = resolve_agent_command_factory(cfg.agent_harness, protocol_root=cfg.protocol_root)
            await preflight_scaffold(
                sandbox,
                wstunnel_bin=cfg.wstunnel_bin,
                readiness_command=factory.readiness_command(),
                user=user,
                log=live_log,
                harness_label=cfg.agent_harness,
                protocol_root=cfg.protocol_root,
            )
            print("  ✓ scaffold and wstunnel are ready")

        print("\n=== prepare workspace ===")
        workspace_output = await prepare_workspace(
            sandbox,
            metadata,
            user=user,
            wait_timeout=cfg.wait_timeout,
            log=live_log,
        )
        print("  ✓ workspace prepared")
        if workspace_output.strip():
            print("  " + workspace_output.strip().splitlines()[-1])

        print("\n=== apply gold patch (skip agent turns) ===")
        await _apply_gold_patch(sandbox, metadata, user=user, wait_timeout=cfg.wait_timeout, log=live_log)
        print("  ✓ gold patch applied")

        print("\n=== run_rebench_eval ===")
        reward, eval_extras = await run_rebench_eval(
            sandbox,
            metadata,
            user=user,
            eval_log_path=eval_log,
            wait_timeout=cfg.wait_timeout,
            preview_limit=cfg.preview_limit,
            reached_turn_limit=False,
            last_generation_finish_reason="stop",
        )
        print(f"  reward={reward}")
        print(f"  solved={eval_extras.get('solved')}  parser={eval_extras.get('log_parser')}")
        print(f"  passed_actual={eval_extras.get('passed_actual')}")
        print(f"  failed_actual={eval_extras.get('failed_actual')}")
        print(f"  from_fail_to_pass={eval_extras.get('from_fail_to_pass')}")
        print(f"  fail_to_pass_expected={eval_extras.get('fail_to_pass_expected')}")
        print(f"  eval_exit_code={eval_extras.get('eval_exit_code')}")

        if reward <= 0.0:
            preview = eval_extras.get("eval_output_preview", "")
            print("\n  [FAIL] gold patch produced reward <= 0; eval preview (last 1500 chars):")
            print("  " + preview[-1500:].replace("\n", "\n  "))
            rc = 1
        else:
            print(f"\n  ✓ gold rollout scores reward={reward}")
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        rc = 1
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        rc = 2
    finally:
        if sandbox is not None and not args.keep_sandbox:
            print("\n=== teardown ===")
            try:
                await asyncio.to_thread(sandbox.kill)
                print("  ✓ sandbox killed")
            except Exception as e:
                print(f"  [warn] stop failed: {e}")
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
