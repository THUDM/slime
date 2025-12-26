from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from examples.eval.eval_delegate import EvalEnvConfig


@dataclass
class TerminalBenchConfig(EvalEnvConfig):
    """Environment configuration shared by the Terminal Bench client/server."""

    model_name: str = "qwen3-8b"
    api_base: str = "http://127.0.1.1:30001/v1"
    dataset_path: str | None = None
    n_tasks: int | None = None
    task_id: str | None = None
    task_ids: list[str] = field(default_factory=list)
    n_attempts: int | None = None
    n_concurrent: int = 8


    @classmethod
    def parse(cls, args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]) -> TerminalBenchConfig:
        clean_raw = dict(raw_env_config or {})
        clean_raw.pop("type", None)
        base_cfg: TerminalBenchConfig = super().parse(clean_raw, defaults)
        model_name = clean_raw.get("model_name")
        if model_name is not None:
            base_cfg.model_name = str(model_name)
        api_base = clean_raw.get("api_base")
        if api_base is not None:
            base_cfg.api_base = str(api_base)
        n_attempts = clean_raw.get("n_attempts")
        if n_attempts is not None:
            base_cfg.n_attempts = int(n_attempts)
        n_tasks = clean_raw.get("n_tasks")
        if n_tasks is not None:
            base_cfg.n_tasks = int(n_tasks)
        n_concurrent = clean_raw.get("n_concurrent")
        if n_concurrent is not None:
            base_cfg.n_concurrent = int(n_concurrent)
        dataset_path = clean_raw.get("dataset_path")
        if dataset_path is not None:
            base_cfg.dataset_path = str(dataset_path)
        task_id = clean_raw.get("task_id")
        if task_id is not None:
            base_cfg.task_id = str(task_id)
        task_ids = clean_raw.get("task_ids")
        if task_ids is None:
            task_ids = task_id
        if task_ids is not None:
            if isinstance(task_ids, (list, tuple)):
                base_cfg.task_ids = [str(item) for item in task_ids if item]
            else:
                base_cfg.task_ids = [str(task_ids)]
        return base_cfg


def build_terminal_bench_config(args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]):
    return TerminalBenchConfig.parse(args, raw_env_config, defaults)
