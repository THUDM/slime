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
    task_ids: list[str] = field(default_factory=list)
    n_attempts: int | None = None
    n_concurrent: int = 8

    @classmethod
    def parse(cls, args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]) -> TerminalBenchConfig:
        clean_raw = dict(raw_env_config or {})
        clean_raw.pop("type", None)
        base_cfg: TerminalBenchConfig = super().parse(clean_raw, defaults)

        field_casts = {
            "model_name": str,
            "api_base": str,
            "n_attempts": int,
            "n_tasks": int,
            "n_concurrent": int,
            "dataset_path": str,
        }

        for key, caster in field_casts.items():
            value = clean_raw.get(key)
            if value is not None:
                setattr(base_cfg, key, caster(value))

        task_ids = clean_raw.get("task_ids")
        if isinstance(task_ids, (list, tuple)):
            base_cfg.task_ids = [str(item) for item in task_ids if item]
        elif task_ids is not None:
            raise ValueError("task_ids must be a list")

        return base_cfg



def build_terminal_bench_config(args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]):
    return TerminalBenchConfig.parse(args, raw_env_config, defaults)
