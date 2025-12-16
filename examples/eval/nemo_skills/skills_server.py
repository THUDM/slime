#!/usr/bin/env python3
"""
Simple HTTP server that proxies Slime evaluation requests to the `ns eval`
command shipped with NeMo Skills.

Usage:
    python examples/eval/nemo_skills/skills_server.py \
        --host 0.0.0.0 --port 9050 \
        --output-root /opt/skills-eval \
        --config-dir examples/eval/nemo_skills/config \
        --cluster local_cluster \
        --max-concurrent-requests 512 \
        --openai-model-name slime-openai-model

Slime (or Slime-compatible runners) should POST the payload described in
`EvalRequestPayload` to http://<host>:<port>/evaluate. The server blocks until
`ns eval` finishes, then returns aggregated metrics along with paths to the
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.eval.nemo_skills.skills_config import SkillsEvalEnvDatasetConfig
from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

logger = logging.getLogger("skills_eval_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Request payload helpers
# ---------------------------------------------------------------------------


@dataclass
class EvalRequestPayload:
    rollout_id: int
    router_url: str
    defaults: dict[str, Any] = field(default_factory=dict)
    benchmarks: list[SkillsEvalEnvDatasetConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration + command helpers
# ---------------------------------------------------------------------------


HYDRA_OVERRIDE_MAP = {
    "temperature": "inference.temperature",
    "top_p": "inference.top_p",
    "top_k": "inference.top_k",
    "max_response_len": "inference.tokens_to_generate",
}


def _openai_api_base(router_url: str) -> str:
    return router_url.rstrip("/") + "/v1"


def _hydra_overrides_from_benchmark(
    defaults: Mapping[str, Any],
    benchmark_cfg: SkillsEvalEnvDatasetConfig,
    router_url: str,
    openai_model_name: str,
    max_concurrent_requests: int,
) -> list[str]:
    overrides: list[str] = []
    for key, hydra_key in HYDRA_OVERRIDE_MAP.items():
        value = getattr(benchmark_cfg, key, None)
        if value is None:
            value = defaults.get(key)
        if value is not None:
            overrides.append(f"++{hydra_key}={value}")

    api_base = _openai_api_base(router_url)
    overrides.extend(
        [
            "++server.server_type=openai",
            f"++server.base_url={api_base}",
            f"++server.model={openai_model_name}",
            "++server.api_key=EMPTY",
            f"++max_concurrent_requests={max_concurrent_requests}",
        ]
    )
    return overrides


@dataclass
class ServerConfig:
    output_root: Path
    cluster: str | None
    config_dir: str | None
    openai_model_name: str | None = None
    max_concurrent_requests: int = 512

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ServerConfig:
        return cls(
            output_root=Path(args.output_root).expanduser().resolve(),
            cluster=args.cluster,
            config_dir=args.config_dir,
            openai_model_name=args.openai_model_name,
            max_concurrent_requests=args.max_concurrent_requests,
        )


class SkillsEvaluator:
    def __init__(self, config: ServerConfig):
        self._config = config
        self._lock = threading.Lock()
        self._config.output_root.mkdir(parents=True, exist_ok=True)

    def evaluate(self, payload: EvalRequestPayload) -> dict[str, Any]:
        if not payload.benchmarks:
            warning_msg = "No benchmarks specified in delegate config; skipping NeMo Skills evaluation."
            logger.warning(warning_msg)
            return {
                "job_id": uuid.uuid4().hex,
                "command": None,
                "output_dir": None,
                "log_path": None,
                "warning": warning_msg,
                "raw_metrics": {},
            }

        job_id = uuid.uuid4().hex
        exp_name = f"rollout-{payload.rollout_id}"
        router_url = payload.router_url.rstrip("/")
        run_dir = self._config.output_root / f"{int(time.time())}-{exp_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        runs: list[dict[str, Any]] = []
        raw_metrics: dict[str, Any] = {}
        with self._lock:
            for benchmark in payload.benchmarks:
                result = self._run_single_benchmark(
                    payload,
                    benchmark,
                    exp_name,
                    router_url,
                    run_dir,
                )
                runs.append(result["run_info"])
                raw_metrics.update(result["metrics"])

        command_summary = "\n".join(run["command"] for run in runs) if runs else None
        log_path = runs[-1]["log_path"] if runs else None

        return {
            "job_id": job_id,
            "command": command_summary,
            "output_dir": str(run_dir),
            "log_path": log_path,
            "raw_metrics": raw_metrics,
            "runs": runs,
        }

    def _run_single_benchmark(
        self,
        payload: EvalRequestPayload,
        benchmark: SkillsEvalEnvDatasetConfig,
        exp_name: str,
        router_url: str,
        run_dir: Path,
    ) -> dict[str, Any]:
        name = benchmark.name
        benchmark_run_dir = run_dir / name
        benchmark_run_dir.mkdir(parents=True, exist_ok=True)
        bench_exp_name = f"{exp_name}-{name}"
        log_path = benchmark_run_dir / "skills_eval.log"

        command = self._build_command(
            benchmark=benchmark.runtime_name,
            exp_name=bench_exp_name,
            router_url=router_url,
            run_dir=benchmark_run_dir,
            defaults=payload.defaults,
            benchmark_cfg=benchmark,
        )
        env = self._build_env()
        logger.info("Starting NeMo Skills eval for %s: %s", name, " ".join(shlex.quote(part) for part in command))
        self._run_command(command, env=env, log_path=log_path)

        metrics = self._collect_metrics(benchmark_run_dir, benchmark.runtime_name)
        return {
            "run_info": {
                "benchmark": name,
                "command": " ".join(shlex.quote(part) for part in command),
                "output_dir": str(benchmark_run_dir),
                "log_path": str(log_path),
            },
            "metrics": metrics,
        }

    def _build_command(
        self,
        benchmark: str,
        exp_name: str,
        router_url: str,
        run_dir: Path,
        defaults: Mapping[str, Any],
        benchmark_cfg: SkillsEvalEnvDatasetConfig,
    ) -> list[str]:
        base_cmd = [
            "ns",
            "eval",
            "--output_dir",
            str(run_dir),
            "--benchmarks",
            benchmark,
            "--server_type",
            "openai",
            "--server_address",
            _openai_api_base(router_url),
            "--expname",
            exp_name,
        ]
        if self._config.cluster:
            base_cmd.extend(["--cluster", self._config.cluster])
        if self._config.config_dir:
            base_cmd.extend(["--config_dir", self._config.config_dir])

        openai_model_name = self._config.openai_model_name or "slime-openai-model"
        hydra_overrides = _hydra_overrides_from_benchmark(
            defaults,
            benchmark_cfg,
            router_url,
            openai_model_name,
            self._config.max_concurrent_requests,
        )
        return base_cmd + hydra_overrides

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        return env

    @staticmethod
    def _run_command(cmd: list[str], *, env: dict[str, str], log_path: Path):
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            retcode = process.wait()
        if retcode != 0:
            with open(log_path, encoding="utf-8", errors="ignore") as log_file:
                tail = "".join(log_file.readlines()[-200:])
            raise RuntimeError(f"`ns eval` failed with exit code {retcode}. See {log_path}\n{tail}")

    @staticmethod
    def _collect_metrics(run_dir: Path, benchmark: str) -> dict[str, Any]:
        benchmark_name = benchmark.split(":")[0]
        metrics_path = run_dir / "eval-results" / benchmark_name / "metrics.json"
        if not metrics_path.exists():
            logger.warning("Metrics file missing for %s at %s", benchmark_name, metrics_path)
            return {}
        try:
            with open(metrics_path, encoding="utf-8") as fp:
                metrics_data = json.load(fp)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse %s: %s", metrics_path, exc)
            return {}
        return {benchmark_name: metrics_data.get(benchmark_name, metrics_data)}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def build_app(evaluator: SkillsEvaluator) -> Flask:
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

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NeMo Skills evaluation HTTP server.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9050)
    parser.add_argument(
        "--output-root",
        type=str,
        default="./skills-eval-output",
        help="Directory to store `ns eval` outputs.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Cluster profile passed to `ns eval --cluster`.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Config directory passed to `ns eval --config_dir`.",
    )
    parser.add_argument(
        "--openai-model-name",
        type=str,
        default=None,
        help="Model identifier to pass when using the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=512,
        help="Maximum concurrent requests per benchmark run forwarded to NeMo Skills.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ServerConfig.from_args(args)
    evaluator = SkillsEvaluator(config)
    app = build_app(evaluator)
    logger.info(
        "Starting Skills evaluation server on %s:%s (output root=%s)",
        args.host,
        args.port,
        config.output_root,
    )
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
