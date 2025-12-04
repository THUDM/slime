import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from omegaconf import OmegaConf
from slime.utils.eval_config import DATASET_RUNTIME_SPECS, _apply_dataset_field_overrides

logger = logging.getLogger(__name__)


@dataclass
class EvalEnvDatasetConfig:
    """Dataset-level generation parameters shared across delegate clients."""

    name: str = ""
    n_samples_per_eval_prompt: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_response_len: int | None = None

    FIELD_NAMES = ("n_samples_per_eval_prompt", "temperature", "top_p", "top_k", "max_response_len")
    FIELD_SPECS = {field: DATASET_RUNTIME_SPECS[field] for field in FIELD_NAMES}

    @classmethod
    def parse(cls, args, dataset_cfg: Mapping[str, Any], defaults: Mapping[str, Any]) -> "EvalEnvDatasetConfig":
        """Merge dataset overrides with defaults/CLI settings and coerce types via OmegaConf."""
        defaults = defaults or {}
        name = str(dataset_cfg.get("name", "")).strip()
        if not name:
            raise ValueError("Each delegate dataset entry must include a non-empty `name`.")
        if ":" in name:
            raise ValueError(
                "Colon in dataset name is not allowed; use `n_samples_per_eval_prompt` to configure samples per prompt."
            )

        _apply_dataset_field_overrides(args, dataset_cfg, defaults, cls.FIELD_SPECS)

        cfg = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(dataset_cfg))
        obj = OmegaConf.to_object(cfg)
        if not isinstance(obj, cls):
            obj = cls(**obj)
        return obj

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for this dataset configuration."""
        payload: dict[str, Any] = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if value is None:
                continue
            payload[field_info.name] = value
        return payload


@dataclass
class EvalEnvConfig:
    """Environment definition shared across delegate implementations."""

    name: str = ""
    url: str | None = None
    timeout_secs: int = 3600
    max_retries: int = 1
    headers: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Mapping[str, Any], defaults: Mapping[str, Any]) -> "EvalEnvConfig":
        cfg = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(raw or {}))
        obj = OmegaConf.to_object(cfg)
        if not isinstance(obj, cls):
            obj = cls(**obj)

        return obj


def _rebuild_delegate_config(
    args, raw_delegate_config: Sequence[Mapping[str, Any]] | None, defaults: Mapping[str, Any] | None
) -> list[EvalEnvConfig]:
    envs: list[EvalEnvConfig] = []
    defaults = defaults or {}
    for env in raw_delegate_config or []:
        env_name = str(env.get("name", "")).strip().lower()
        if not env_name:
            logger.warning("Each delegate entry must include a non-empty `name`.")
            continue
        if env_name == "skills":
            from examples.eval.nemo_skills.skills_config import build_skills_eval_env_config

            env_cfg = build_skills_eval_env_config(args, env, defaults)
            if env_cfg is not None:
                envs.append(env_cfg)
        else:
            raise ValueError(f"Unknown delegate environment: {env_name}")
    return envs


class EvalDelegateError(RuntimeError):
    """Raised when the external evaluation server returns an error."""


class EvalClient:
    name: str = ""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, args, rollout_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")


def _flatten(result: dict[str, Any], prefix: str | None = None) -> dict[str, Any]:
    """Flatten nested metric dicts into slash separated keys."""
    flattened: dict[str, Any] = {}
    for key, value in (result or {}).items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


class EvalDelegateClient:
    """Aggregate multiple environment-specific delegate clients."""

    def __init__(self, delegates: Sequence[EvalClient]):
        self._delegates = list(delegates)

    @classmethod
    def maybe_create(cls, args, env_configs: Sequence[EvalEnvConfig] | None = None) -> Optional["EvalDelegateClient"]:
        env_configs = list(env_configs) if env_configs is not None else getattr(args, "eval_delegate_config", None)
        if not env_configs:
            return None

        router_addr = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        delegates: list[EvalClient] = []
        for env_cfg in env_configs:
            delegate = cls._create_delegate(env_cfg, router_addr)
            if delegate is not None:
                delegates.append(delegate)
        if not delegates:
            return None
        return cls(delegates)

    @staticmethod
    def _create_delegate(env_cfg: EvalEnvConfig, router_addr: str):
        env_name = env_cfg.name
        if env_name == "skills":
            from examples.eval.nemo_skills.skills_client import SkillsEvalClient

            return SkillsEvalClient.from_config(env_cfg, router_addr)
        logger.warning("No delegate client registered for environment: %s", env_name)
        return None

    def evaluate(self, args, rollout_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        aggregated_metrics: dict[str, Any] = {}
        raw_responses: dict[str, Any] = {}
        for delegate in self._delegates:
            metrics, response = delegate.evaluate(args, rollout_id)
            if metrics:
                aggregated_metrics.update(metrics)
            raw_responses[delegate.name] = response
        return aggregated_metrics, raw_responses
