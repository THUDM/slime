from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from examples.eval.eval_delegate import EvalEnvConfig, EvalEnvDatasetConfig


@dataclass
class SkillsEvalEnvDatasetConfig(EvalEnvDatasetConfig):
    """Dataset configuration shared by the Skills client/server."""

    def __post_init__(self):
        name = (self.name or "").strip()
        self.name = name
        if not name:
            raise ValueError("Each Skills dataset entry must include a non-empty `name`.")
        if ":" in name:
            raise ValueError(
                "Colon in dataset name is not allowed; use `n_samples_per_eval_prompt` to configure samples per prompt."
            )

    @property
    def runtime_name(self) -> str:
        if self.n_samples_per_eval_prompt is None:
            return self.name
        return f"{self.name}:{self.n_samples_per_eval_prompt}"

    @classmethod
    def parse(cls, args, dataset_cfg: Mapping[str, Any], defaults: Mapping[str, Any]):
        return super().parse(args, dataset_cfg, defaults)


@dataclass
class SkillsEvalEnvConfig(EvalEnvConfig):
    """Environment configuration shared by the Skills client/server."""

    datasets: list[SkillsEvalEnvDatasetConfig] = field(default_factory=list)

    @classmethod
    def parse(cls, args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]) -> SkillsEvalEnvConfig:
        base_cfg: SkillsEvalEnvConfig = super().parse(raw_env_config, defaults)
        datasets = raw_env_config.get("datasets") or []
        base_cfg.datasets = [
            SkillsEvalEnvDatasetConfig.parse(args, dataset_cfg, base_cfg.defaults) for dataset_cfg in datasets
        ]
        return base_cfg


def build_skills_eval_env_config(args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]):
    return SkillsEvalEnvConfig.parse(args, raw_env_config, defaults)
