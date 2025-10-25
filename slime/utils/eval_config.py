from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_EMPTY_VALUES = (None, [], {})


@dataclass
class EvalDatasetConfig:
    """Configuration for a single evaluation dataset."""

    name: str
    path: str
    rm_type: Optional[str] = None

    # Dataset-specific overrides
    prompt_key: Optional[str] = None
    label_key: Optional[str] = None
    tool_key: Optional[str] = None
    metadata_key: Optional[str] = None

    n_samples_per_eval_prompt: Optional[int] = None

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_response_len: Optional[int] = None
    min_new_tokens: Optional[int] = None

    stop: Optional[Sequence[str]] = None
    stop_token_ids: Optional[Sequence[int]] = None

    metadata_overrides: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata_overrides is None:
            self.metadata_overrides = {}

    def apply_defaults(self, defaults: Dict[str, Any]) -> None:
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
    def cache_key(self) -> Tuple[Any, ...]:
        """Return a tuple uniquely identifying dataset config for caching."""
        return (
            self.name,
            self.path,
            self.prompt_key,
            self.label_key,
            self.tool_key,
            self.metadata_key,
        )

    def inject_metadata(self, sample_metadata: Any) -> Dict[str, Any]:
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


def ensure_dataset_list(config: Any) -> List[Dict[str, Any]]:
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
    raw_config: Iterable[Dict[str, Any]], defaults: Dict[str, Any]
) -> List[EvalDatasetConfig]:
    datasets: List[EvalDatasetConfig] = []
    for cfg in raw_config:
        dataset = EvalDatasetConfig(**cfg)
        dataset.apply_defaults(defaults)
        datasets.append(dataset)
    return datasets
