#!/usr/bin/env python3
"""
Simple HTTP server that proxies Slime evaluation requests to the `tb run`
command shipped with Terminal Bench.

Usage:
    python examples/eval/terminal_bench/tb_server.py \
        --host 0.0.0.0 --port 9050 \
        --output-root /opt/tb-eval

Slime (or Slime-compatible runners) should POST the payload described in
`EvalRequestPayload` to http://<host>:<port>/evaluate. The server blocks until
`tb run` finishes, then returns aggregated metrics along with paths to the
generated artifacts (logs + raw metrics).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

logger = logging.getLogger("terminal_bench_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Request payload helpers
# ---------------------------------------------------------------------------


@dataclass
class EvalRequestPayload:
    model_name: str = ""
    api_base: str = ""
    n_tasks: int | None = None
    n_concurrent: int | None = None
    dataset_path: str | None = None
    task_ids: list[str] | None = None
    task_id: str | None = None
    n_attempts: int | None = None
    metric_prefix: str | None = None


@dataclass
class JobRecord:
    job_id: str
    status: str
    run_id: str
    command: str
    output_dir: str
    log_path: str
    raw_metrics: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "run_id": self.run_id,
            "command": self.command,
            "output_dir": self.output_dir,
            "log_path": self.log_path,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.raw_metrics is not None:
            payload["raw_metrics"] = self.raw_metrics
        if self.error:
            payload["error"] = self.error
        return payload


# ---------------------------------------------------------------------------
# Configuration + command helpers
# ---------------------------------------------------------------------------


def _normalize_model_name(model_name: str) -> str:
    name = (model_name or "").strip()
    if not name:
        return ""
    if "/" in name:
        return name
    return f"openai/{name}"


@dataclass
class ServerConfig:
    output_root: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ServerConfig":
        return cls(output_root=Path(args.output_root).expanduser().resolve())


class TerminalBenchEvaluator:
    def __init__(self, config: ServerConfig):
        self._config = config
        self._lock = threading.Lock()
        self._jobs_lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._config.output_root.mkdir(parents=True, exist_ok=True)
        self._log_root = REPO_ROOT / "tb_eval_logs"
        self._log_root.mkdir(parents=True, exist_ok=True)

    def evaluate(self, payload: EvalRequestPayload) -> dict[str, Any]:
        if not payload.model_name:
            raise ValueError("Missing `model_name` in request payload.")
        if not payload.api_base:
            raise ValueError("Missing `api_base` in request payload.")

        job_id = uuid.uuid4().hex
        run_id = f"{int(time.time())}-{job_id[:8]}"
        run_dir = self._config.output_root / run_id

        command = self._build_command(payload, run_id)
        command_str = " ".join(shlex.quote(part) for part in command)
        log_path = self._log_root / f"{run_id}.log"

        record = JobRecord(
            job_id=job_id,
            status="queued",
            run_id=run_id,
            command=command_str,
            output_dir=str(run_dir),
            log_path=str(log_path),
        )
        with self._jobs_lock:
            self._jobs[job_id] = record

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, payload, run_dir, command, log_path),
            daemon=True,
        )
        thread.start()

        return {
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/status/{job_id}",
            "run_id": run_id,
            "command": command_str,
            "output_dir": str(run_dir),
            "log_path": str(log_path),
        }

    def _run_job(
        self,
        job_id: str,
        payload: EvalRequestPayload,
        run_dir: Path,
        command: list[str],
        log_path: Path,
    ) -> None:
        with self._jobs_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "running"
            record.started_at = time.time()

        env = self._build_env()
        logger.info("Starting Terminal Bench run: %s", " ".join(shlex.quote(part) for part in command))
        try:
            with self._lock:
                self._run_command(command, env=env, log_path=log_path)
            metrics = self._collect_metrics(run_dir)
            if payload.metric_prefix:
                metrics = {payload.metric_prefix: metrics}
            with self._jobs_lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.status = "completed"
                record.raw_metrics = metrics
                record.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with self._jobs_lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.status = "failed"
                record.error = str(exc)
                record.finished_at = time.time()

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        with self._jobs_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return record.to_dict()

    def _build_command(self, payload: EvalRequestPayload, run_id: str) -> list[str]:
        # 1. Normalize model name (add openai/ prefix)
        model_name = _normalize_model_name(payload.model_name)

        cmd = [
            "tb",
            "run",
            "-a",
            "terminus-2",  # Added Agent flag
            "--output-path",
            str(self._config.output_root),
            "--run-id",
            run_id,
        ]

        # 2. Add model
        if model_name:
            cmd.extend(["--model", model_name])

        # 3. Add Agent kwargs (Use api_base exactly like the CLI command)
        if payload.api_base:
            cmd.extend(["--agent-kwarg", f"api_base={payload.api_base}"])

        # 4. Add n_tasks if present
        task_ids = []
        if payload.task_ids:
            task_ids.extend([str(item) for item in payload.task_ids if item])
        if payload.task_id:
            task_ids.append(str(payload.task_id))

        if payload.dataset_path:
            cmd.extend(["--dataset-path", payload.dataset_path])
        
        if payload.n_attempts is not None:
            cmd.extend(["--n-attempts", str(payload.n_attempts)])

        if task_ids:
            for task_id in task_ids:
                cmd.extend(["--task-id", task_id])
        elif payload.n_tasks is not None:
            cmd.extend(["--n-tasks", str(payload.n_tasks)])

        # 5. Add concurrency
        n_concurrent = payload.n_concurrent
        if n_concurrent is None:
            n_concurrent = 1
        cmd.extend(["--n-concurrent", str(n_concurrent)])

        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # Inject env var to simulate "OPENAI_API_KEY=EMPTY"
        env["OPENAI_API_KEY"] = "EMPTY"
        return env

    @staticmethod
    def _run_command(cmd: list[str], *, env: dict[str, str], log_path: Path):
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            retcode = process.wait()
        if retcode != 0:
            with open(log_path, encoding="utf-8", errors="ignore") as log_file:
                tail = "".join(log_file.readlines()[-200:])
            raise RuntimeError(f"`tb run` failed with exit code {retcode}. See {log_path}\n{tail}")

    @staticmethod
    def _collect_metrics(run_dir: Path) -> dict[str, Any]:
        metrics_path = run_dir / "results.json"
        if not metrics_path.exists():
            logger.warning("Results file missing at %s", metrics_path)
            return {}

        metrics = TerminalBenchEvaluator._extract_metrics(metrics_path)
        if not metrics:
            logger.warning("No accuracy/n_resolved metrics found in %s", metrics_path)
        return metrics

    @staticmethod
    def _extract_metrics(metrics_path: Path) -> dict[str, Any]:
        try:
            with open(metrics_path, encoding="utf-8") as fp:
                metrics_data = json.load(fp)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse %s: %s", metrics_path, exc)
            return {}

        metrics: dict[str, Any] = {}
        
        # core metrics
        accuracy = metrics_data.get("accuracy")
        if isinstance(accuracy, (int, float)):
            metrics["accuracy"] = float(accuracy)

        n_resolved = metrics_data.get("n_resolved")
        if isinstance(n_resolved, (int, float)):
            metrics["n_resolved"] = int(n_resolved)

        n_unresolved = metrics_data.get("n_unresolved")
        if isinstance(n_unresolved, (int, float)):
            metrics["n_unresolved"] = int(n_unresolved)

        # pass@k flatten
        pass_at_k = metrics_data.get("pass_at_k")
        if isinstance(pass_at_k, dict):
            for k, v in pass_at_k.items():
                if isinstance(v, (int, float)):
                    metrics[f"pass_at_k/{k}"] = float(v)

        # token stats from per-task results
        results = metrics_data.get("results")
        if isinstance(results, list):
            input_tokens = [
                r.get("total_input_tokens")
                for r in results
                if isinstance(r, dict) and isinstance(r.get("total_input_tokens"), (int, float))
            ]
            output_tokens = [
                r.get("total_output_tokens")
                for r in results
                if isinstance(r, dict) and isinstance(r.get("total_output_tokens"), (int, float))
            ]

            if input_tokens:
                metrics["total_input_tokens_mean"] = float(statistics.mean(input_tokens))
                metrics["total_input_tokens_median"] = float(statistics.median(input_tokens))
            if output_tokens:
                metrics["total_output_tokens_mean"] = float(statistics.mean(output_tokens))
                metrics["total_output_tokens_median"] = float(statistics.median(output_tokens))

        return metrics


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def build_app(evaluator: TerminalBenchEvaluator) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/evaluate")
    def evaluate_endpoint():
        try:
            raw_payload = request.get_json(force=True, silent=False)
            cfg = OmegaConf.merge(
                OmegaConf.structured(EvalRequestPayload),
                OmegaConf.create(raw_payload or {}),
            )
            payload = OmegaConf.to_object(cfg)
            result = evaluator.evaluate(payload)
            return jsonify(result)
        except OmegaConfBaseException as exc:
            logger.exception("Invalid request payload")
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("Evaluation failed")
            return jsonify({"error": str(exc)}), 500

    @app.get("/status/<job_id>")
    def status_endpoint(job_id: str):
        status = evaluator.get_job_status(job_id)
        if status is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(status)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Terminal Bench evaluation HTTP server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9050)
    parser.add_argument(
        "--output-root",
        type=str,
        default="./terminal-bench-output",
        help="Directory to store `tb run` outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ServerConfig.from_args(args)
    evaluator = TerminalBenchEvaluator(config)
    app = build_app(evaluator)
    logger.info(
        "Starting Terminal Bench evaluation server on %s:%s (output root=%s)",
        args.host,
        args.port,
        config.output_root,
    )
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
