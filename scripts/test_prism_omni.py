#!/usr/bin/env python3
"""Probe real Prism Omni replicas and their Python router on reserved GPUs.

Run from the repository with python -m scripts.test_prism_omni. Static mode
loads the HF artifact without publishing trainer weights. Publication mode
calls a trainer hook before allowing the initially paused replicas to serve.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing
import signal
import socket
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", "--hf", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("examples/prism/student.yaml"))
    parser.add_argument("--python", default=sys.executable, help="Interpreter for Omni workers and router")
    parser.add_argument("--artifact-dir", "--artifactdir", type=Path, required=True)
    parser.add_argument("--gpus", default="0", help="One or two physical GPU ids, one per pipeline replica")
    parser.add_argument("--mode", choices=("static", "publication"), default="static")
    parser.add_argument("--publication-hook", help="module:function(engines, context) returning a report dict")
    parser.add_argument(
        "--post-generation-hook", help="module:function(context, response_body, sample) returning a report dict"
    )
    parser.add_argument("--publication-args-json", type=Path, help="Additional trainer hook configuration")
    parser.add_argument("--native-checkpoint", type=Path)
    parser.add_argument("--expected-version", help="Require the resulting policy version to match")
    parser.add_argument("--text", default="你好，欢迎参加语音训练。")
    parser.add_argument(
        "--task-type", default="TTS", help="Checkpoint prompt protocol task; TTS uses no reference or instruction"
    )
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--return-audio", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--cleanup-timeout", type=float, default=30.0)
    parser.add_argument("--admin-key", default=None)
    args = parser.parse_args()
    try:
        args.gpu_ids = [int(part.strip()) for part in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must contain integer GPU ids")
    if len(args.gpu_ids) not in (1, 2) or len(set(args.gpu_ids)) != len(args.gpu_ids) or min(args.gpu_ids) < 0:
        parser.error("--gpus must name one or two distinct nonnegative GPU ids")
    if args.max_frames < 1 or args.seed < 0 or not args.text.strip():
        parser.error("Require positive --max-frames, nonnegative --seed, and nonempty --text")
    for name in ("startup_timeout", "request_timeout", "cleanup_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if (args.mode == "publication") != bool(args.publication_hook):
        parser.error("--mode publication requires --publication-hook; static mode must omit it")
    for name in ("hf_checkpoint", "config", "native_checkpoint", "publication_args_json"):
        value = getattr(args, name)
        if value is not None:
            value = value.resolve(strict=True)
            setattr(args, name, value)
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    return args


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def worker_states(session: requests.Session, router_url: str, engines: list, *, disabled: bool) -> list[dict]:
    states = []
    for engine in engines:
        response = session.get(f"{router_url}/workers/{quote(engine.get_url(), safe='')}", timeout=5)
        response.raise_for_status()
        state = response.json()
        if state.get("disabled") is not disabled:
            raise RuntimeError(f"Router worker admission is inconsistent: {state}")
        states.append(state)
    return states


def wait_router(session: requests.Session, process: multiprocessing.Process, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"Omni router exited with code {process.exitcode}")
        try:
            response = session.get(f"{url}/live", timeout=1)
            response.raise_for_status()
            return
        except requests.RequestException:
            time.sleep(0.1)
    raise TimeoutError("Omni router startup timed out")


def run(args: argparse.Namespace) -> dict:
    from slime.backends.sglang_utils.deployment import launch_omni_router
    from slime.backends.sglang_utils.omni_engine import OMNI_WORKER_CAPABILITIES, OmniEngine
    from slime.utils.types import Sample
    from slime_plugins.omni_rollout import PrismConfig, build_generate_payload, parse_rollout

    config = PrismConfig.from_path(args.hf_checkpoint)
    engine_args = argparse.Namespace(
        omni_config=str(args.config),
        omni_python=args.python,
        omni_admin_key=args.admin_key,
        hf_checkpoint=str(args.hf_checkpoint),
        colocate=False,
        offload_rollout=False,
        rollout_external=False,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=max(args.gpu_ids) + 1,
        distributed_timeout_minutes=args.startup_timeout / 60,
        rollout_health_check_timeout=min(10.0, args.request_timeout),
        rollout_cleanup_timeout=args.cleanup_timeout,
    )
    publication = args.mode == "publication"
    report = {
        "mode": args.mode,
        "static_checkpoint_probe": not publication,
        "trainer_weight_publication_verified": False,
        "hf_checkpoint": str(args.hf_checkpoint),
        "config": str(args.config),
        "gpu_ids": args.gpu_ids,
    }
    engines = []
    router = None
    cleanup_errors = []
    session = requests.Session()
    session.trust_env = False
    if args.admin_key:
        session.headers["Authorization"] = f"Bearer {args.admin_key}"
    try:
        for rank, gpu_id in enumerate(args.gpu_ids):
            engine = OmniEngine(
                engine_args,
                rank,
                base_gpu_id=gpu_id,
                gpu_ids=[gpu_id],
                num_gpus_per_engine=1,
                sglang_overrides={"update_weights": publication},
            )
            engines.append(engine)
            port = available_port()
            print(f"Starting Omni replica {rank} on GPU {gpu_id}, port {port}", flush=True)
            engine.init(f"127.0.0.1:{available_port()}", port, available_port(), host="127.0.0.1")
            if publication and (not engine.paused or engine.get_weight_version() is not None):
                raise RuntimeError("Student replica must remain paused before its first publication")
        router_port = available_port()
        router_url = f"http://127.0.0.1:{router_port}"
        router_config = {
            "host": "127.0.0.1",
            "port": router_port,
            "workers": [{"url": engine.get_url(), "capabilities": OMNI_WORKER_CAPABILITIES} for engine in engines],
            "policy": "round_robin",
            "request_timeout_secs": math.ceil(args.request_timeout),
        }
        router = multiprocessing.get_context("spawn").Process(
            target=launch_omni_router,
            args=(args.python, router_config, publication, args.admin_key),
        )
        router.start()
        wait_router(session, router, router_url, args.startup_timeout)
        for engine in engines:
            engine.attach_router("127.0.0.1", router_port)
        initial_states = worker_states(session, router_url, engines, disabled=publication)
        write_json(args.artifact_dir / "router-workers-before.json", initial_states)
        context = {
            **vars(args),
            "router_url": router_url,
            "omni_urls": [engine.get_url() for engine in engines],
            "hf_checkpoint": str(args.hf_checkpoint),
            "config": str(args.config),
            "artifact_dir": str(args.artifact_dir),
            "native_checkpoint": str(args.native_checkpoint) if args.native_checkpoint else None,
            "publication_args_json": str(args.publication_args_json) if args.publication_args_json else None,
        }
        context["publication_args"] = (
            json.loads(args.publication_args_json.read_text()) if args.publication_args_json else {}
        )
        write_json(
            args.artifact_dir / "deployment.json", {key: value for key, value in context.items() if key != "admin_key"}
        )
        if publication:
            module_name, separator, function_name = args.publication_hook.partition(":")
            if not separator or not module_name or not function_name:
                raise ValueError("--publication-hook must be module:function")
            hook = getattr(importlib.import_module(module_name), function_name)
            hook_report = hook(engines, context)
            if not isinstance(hook_report, dict) or not isinstance(hook_report.get("weight_version"), str):
                raise ValueError("Publication hook must return a report with a string weight_version")
            version = hook_report["weight_version"]
            if not version or args.expected_version is not None and version != args.expected_version:
                raise RuntimeError("Publication hook returned an unexpected policy version")
            for engine in engines:
                if not engine.paused or engine.publication_failed or engine.pending_version != version:
                    raise RuntimeError("Hook must finish all OmniEngine weight buckets and leave every replica paused")
                info = engine.request_admin("model_info")
                if str(info.get("weight_version")) != version:
                    raise RuntimeError("Publication did not install the requested policy version")
            worker_states(session, router_url, engines, disabled=True)
            write_json(args.artifact_dir / "publication.json", hook_report)
            context["publication_report"] = hook_report
            for engine in engines:
                engine.continue_generation()
            report["trainer_weight_publication_verified"] = True
        else:
            versions = {engine.get_weight_version() for engine in engines}
            if len(versions) != 1 or None in versions:
                raise RuntimeError("Static replicas must expose the same nonempty artifact weight version")
            version = versions.pop()
            if args.expected_version is not None and version != args.expected_version:
                raise RuntimeError(f"Static artifact version {version} != {args.expected_version}")
        final_states = worker_states(session, router_url, engines, disabled=False)
        write_json(args.artifact_dir / "router-workers-after.json", final_states)
        infos = [engine.request_admin("model_info") for engine in engines]
        write_json(args.artifact_dir / "worker-info.json", infos)
        for info in infos:
            if (
                str(info.get("weight_version")) != version
                or info["results"][0]["data"].get("engine_paused") is not False
            ):
                raise RuntimeError("Replica version or serving state differs after activation")
        sample = Sample(
            index=0,
            group_index=0,
            prompt=args.text,
            metadata={"task_type": args.task_type, "show_language": False},
        )
        payload = build_generate_payload(
            argparse.Namespace(omni_return_audio=args.return_audio),
            sample,
            {"max_new_tokens": args.max_frames, "seed": args.seed},
        )
        context["weight_version"] = version
        context["request"] = payload
        write_json(
            args.artifact_dir / "deployment.json", {key: value for key, value in context.items() if key != "admin_key"}
        )
        write_json(args.artifact_dir / "request.json", payload)
        print(f"Generating a real Prism trajectory through {router_url}", flush=True)
        response = session.post(f"{router_url}/generate", json=payload, timeout=args.request_timeout)
        response.raise_for_status()
        response_body = response.json()
        write_json(args.artifact_dir / "response.json", response_body)
        parse_rollout(response_body, sample, config, max_frames=args.max_frames, expected_weight_version=version)
        if args.post_generation_hook:
            module_name, separator, function_name = args.post_generation_hook.partition(":")
            if not separator or not module_name or not function_name:
                raise ValueError("--post-generation-hook must be module:function")
            hook = getattr(importlib.import_module(module_name), function_name)
            post_generation_report = hook(context, response_body, sample)
            if not isinstance(post_generation_report, dict):
                raise ValueError("Post-generation hook must return a report dict")
            report["post_generation"] = post_generation_report
        report.update(
            weight_version=version,
            response_json=str(args.artifact_dir / "response.json"),
            frames=sample.response_length,
            prompt_rows=sample.metadata["omni_prompt_rows"],
            finish_reason=sample.metadata["omni_finish_reason"],
            total_action_count=sample.metadata["omni_rollout"]["total_action_count"],
            trajectory_contract_verified=True,
            router_url=router_url,
            omni_urls=[engine.get_url() for engine in engines],
        )
        return report
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error), trajectory_contract_verified=False)
        raise
    finally:
        for engine in reversed(engines):
            try:
                engine.shutdown()
            except Exception as error:
                cleanup_errors.append(f"Engine {engine.rank}: {type(error).__name__}: {error}")
                if engine.process is not None and engine.process.is_alive():
                    engine.process.terminate()
                    engine.process.join(timeout=args.cleanup_timeout)
                    if engine.process.is_alive():
                        engine.process.kill()
                        engine.process.join(timeout=args.cleanup_timeout)
        if router is not None and router.pid is not None:
            if router.is_alive():
                router.terminate()
                router.join(timeout=args.cleanup_timeout)
                if router.is_alive():
                    router.kill()
                    router.join(timeout=args.cleanup_timeout)
            else:
                router.join(timeout=args.cleanup_timeout)
            if router.is_alive():
                cleanup_errors.append("Router process did not exit")
        session.close()
        report["cleanup_errors"] = cleanup_errors
        report["processes_stopped"] = all(
            engine.process is None or not engine.process.is_alive() for engine in engines
        ) and (router is None or not router.is_alive())
        write_json(args.artifact_dir / "summary.json", report)
        if cleanup_errors and "error" not in report:
            raise RuntimeError(f"Omni cleanup failed: {cleanup_errors}")


def handle_sigterm(signum: int, frame: object) -> None:
    raise KeyboardInterrupt("Prism Omni probe received SIGTERM")


def main() -> None:
    args = parse_args()
    previous_handler = signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        report = run(args)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
