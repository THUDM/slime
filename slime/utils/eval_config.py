from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_MISSING = object()

# TODO: This is ugly, temporarily leave this. We should unify all the config name for dataset, default, and args. (advice from Tom.)
DATASET_RUNTIME_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "n_samples_per_eval_prompt": {
        "dataset_keys": ("n_samples_per_eval_prompt",),
        "default_keys": ("n_samples_per_eval_prompt",),
        "arg_attrs": ("n_samples_per_eval_prompt", "n_samples_per_prompt"),
    },
    "temperature": {
        "dataset_keys": ("temperature",),
        "default_keys": ("temperature",),
        "arg_attrs": ("eval_temperature", "rollout_temperature"),
    },
    "top_p": {
        "dataset_keys": ("top_p",),
        "default_keys": ("top_p",),
        "arg_attrs": ("eval_top_p", "rollout_top_p"),
    },
    "top_k": {
        "dataset_keys": ("top_k",),
        "default_keys": ("top_k",),
        "arg_attrs": ("eval_top_k", "rollout_top_k"),
    },
    "max_response_len": {
        "dataset_keys": ("max_response_len",),
        "default_keys": ("max_response_len",),
        "arg_attrs": ("eval_max_response_len", "rollout_max_response_len"),
    },
}

DATASET_SAMPLE_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "input_key": {
        "dataset_keys": ("input_key",),
        "default_keys": ("input_key",),
        "arg_attrs": ("eval_input_key", "input_key"),
    },
    "label_key": {
        "dataset_keys": ("label_key",),
        "default_keys": ("label_key",),
        "arg_attrs": ("eval_label_key", "label_key"),
    },
    "tool_key": {
        "dataset_keys": ("tool_key",),
        "default_keys": ("tool_key",),
        "arg_attrs": ("eval_tool_key", "tool_key"),
    },
    "metadata_key": {
        "dataset_keys": ("metadata_key",),
        "default_keys": ("metadata_key",),
        "arg_attrs": ("metadata_key",),
    },
}


def _first_not_missing(*values: Any) -> Any:
    for value in values:
        if value is not _MISSING:
            return value
    return _MISSING


def _pick_from_mapping(data: dict[str, Any], key_names: tuple[str, ...] | None) -> Any:
    if key_names is None:
        return _MISSING
    for key_name in key_names:
        if key_name in data:
            return data[key_name]
    return _MISSING


def pick_from_args(args: Any, attrs: tuple[str, ...]) -> Any:
    for attr in attrs:
        value = getattr(args, attr, None)
        if value is not None:
            return value
    return None


def _ensure_metadata_overrides(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata_overrides must be a mapping.")
    return value


@dataclass
class EvalDatasetConfig:
    """Configuration for a single evaluation dataset."""

    name: str
    path: str
    rm_type: str | None = None

    # Dataset-specific overrides
    input_key: str | None = None
    label_key: str | None = None
    tool_key: str | None = None
    metadata_key: str | None = None

    n_samples_per_eval_prompt: int | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_response_len: int | None = None
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    min_new_tokens: int | None = None

    # per-dataset custom generate function (e.g., for tool calling)
    custom_generate_function_path: str | None = None

    metadata_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata_overrides = _ensure_metadata_overrides(self.metadata_overrides)

    @property
    def cache_key(self) -> tuple[Any, ...]:
        """Return a tuple uniquely identifying dataset config for caching."""
        return (
            self.name,
            self.path,
            self.input_key,
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


def _apply_dataset_field_overrides(
    args: Any, dataset_cfg: dict[str, Any], defaults: dict[str, Any], spec_names: dict[str, Any]
) -> None:
    for field_name, spec in spec_names.items():
        dataset_value = _pick_from_mapping(dataset_cfg, spec["dataset_keys"])
        default_value = _pick_from_mapping(defaults, spec["default_keys"])
        resolved_value = _first_not_missing(dataset_value, default_value)
        if resolved_value is not _MISSING:
            dataset_cfg[field_name] = resolved_value
            continue
        dataset_cfg[field_name] = pick_from_args(args, spec["arg_attrs"])


def build_eval_dataset_configs(
    args: Any,
    raw_config: Iterable[dict[str, Any]],
    defaults: dict[str, Any],
) -> list[EvalDatasetConfig]:
    defaults = defaults or {}
    datasets: list[EvalDatasetConfig] = []
    for cfg in raw_config:
        cfg_dict = dict(cfg or {})
        combined_specs = {**DATASET_RUNTIME_SPECS, **DATASET_SAMPLE_SPECS}
        _apply_dataset_field_overrides(args, cfg_dict, defaults, combined_specs)
        dataset = EvalDatasetConfig(**cfg_dict)
        datasets.append(dataset)
    return datasets
