from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_EMPTY_VALUES = (None, [], {})


def _ensure_metadata_overrides(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata_overrides must be a mapping.")
    return value


class EvalDatasetConfig(BaseModel):
    """Configuration for a single evaluation dataset."""

    name: str
    path: str
    rm_type: str | None = None

    # Dataset-specific overrides
    prompt_key: str | None = None
    label_key: str | None = None
    tool_key: str | None = None
    metadata_key: str | None = None

    n_samples_per_eval_prompt: int | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_response_len: int | None = None
    min_new_tokens: int | None = None

    stop: Sequence[str] | None = None
    stop_token_ids: Sequence[int] | None = None

    metadata_overrides: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @field_validator("metadata_overrides", mode="before")
    def _validate_metadata_overrides(cls, value: Any) -> dict[str, Any]:
        return _ensure_metadata_overrides(value)

    def apply_defaults(self, defaults: dict[str, Any]) -> None:
        for key, value in defaults.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if current in _EMPTY_VALUES:
                if isinstance(value, (dict, list)):
                    setattr(self, key, copy.deepcopy(value))
                else:
                    setattr(self, key, value)

    @property
    def cache_key(self) -> tuple[Any, ...]:
        """Return a tuple uniquely identifying dataset config for caching."""
        return (
            self.name,
            self.path,
            self.prompt_key,
            self.label_key,
            self.tool_key,
            self.metadata_key,
        )

    def inject_metadata(self, sample_metadata: Any) -> dict[str, Any]:
        """Return updated metadata merging overrides."""
        if not isinstance(sample_metadata, dict):
            metadata = {}
        else:
            metadata = dict(sample_metadata)

        if self.rm_type is not None:
            metadata["rm_type"] = self.rm_type

        for key, value in self.metadata_overrides.items():
            metadata[key] = value

        return metadata


def ensure_dataset_list(config: Any) -> list[dict[str, Any]]:
    """
    Normalize OmegaConf containers into a list of dicts.
    Accepts either a list or dictionary keyed by dataset name.
    """
    if config is None:
        return []

    if isinstance(config, dict):
        datasets = []
        for name, cfg in config.items():
            dataset = dict(cfg or {})
            dataset.setdefault("name", name)
            datasets.append(dataset)
        return datasets

    if isinstance(config, (list, tuple)):
        datasets = []
        for item in config:
            dataset = dict(item or {})
            if "name" not in dataset:
                raise ValueError("Each evaluation dataset entry must include a `name` field.")
            datasets.append(dataset)
        return datasets

    raise TypeError("eval.datasets must be either a list or a mapping.")


def build_eval_dataset_configs(
    raw_config: Iterable[dict[str, Any]], defaults: dict[str, Any]
) -> list[EvalDatasetConfig]:
    datasets: list[EvalDatasetConfig] = []
    for cfg in raw_config:
        dataset = EvalDatasetConfig(**cfg)
        dataset.apply_defaults(defaults)
        datasets.append(dataset)
    return datasets
