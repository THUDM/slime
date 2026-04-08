from __future__ import annotations

from pathlib import Path
from typing import Sequence

from examples.common.dataset_registry import DEFAULT_TRAIN_DATASETS_BY_DOMAIN, DEFAULT_TRAIN_DATASETS_BY_GROUP
from examples.common.dataset_selection import (
    discover_canonical_train_sources as discover_canonical_train_sources_shared,
    resolve_named_datasets as resolve_named_datasets_shared,
    train_domains_for_datasets as train_domains_for_datasets_shared,
    train_groups_for_datasets as train_groups_for_datasets_shared,
)


def default_train_datasets_for_domain(domain: str) -> tuple[str, ...]:
    try:
        return DEFAULT_TRAIN_DATASETS_BY_DOMAIN[domain]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_TRAIN_DATASETS_BY_DOMAIN))
        raise ValueError(f"Unsupported train domain '{domain}'. Supported domains: {supported}") from exc


def default_train_datasets_for_group(group: str) -> tuple[str, ...]:
    try:
        return DEFAULT_TRAIN_DATASETS_BY_GROUP[group]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_TRAIN_DATASETS_BY_GROUP))
        raise ValueError(f"Unsupported train group '{group}'. Supported groups: {supported}") from exc


def train_domains_for_datasets(dataset_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(train_domains_for_datasets_shared(list(dataset_names)))


def train_groups_for_datasets(dataset_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(train_groups_for_datasets_shared(list(dataset_names)))


def domain_signature(domains: Sequence[str]) -> str:
    ordered = tuple(dict.fromkeys(domains))
    if not ordered:
        return "unknown"
    if len(ordered) == 1:
        return ordered[0]
    return "mixed-" + "+".join(ordered)


def group_signature(groups: Sequence[str]) -> str:
    ordered = tuple(dict.fromkeys(groups))
    if not ordered:
        return "unknown"
    formatted = tuple(group.replace("_", "-") for group in ordered)
    if len(formatted) == 1:
        return formatted[0]
    return "mixed-" + "+".join(formatted)


def domain_signature_for_train_datasets(dataset_names: Sequence[str]) -> str:
    return domain_signature(train_domains_for_datasets(dataset_names))


def group_signature_for_train_datasets(dataset_names: Sequence[str]) -> str:
    return group_signature(train_groups_for_datasets(dataset_names))


def resolve_named_datasets(pool_root: Path, dataset_names: Sequence[str]) -> list[Path]:
    return resolve_named_datasets_shared(pool_root, list(dataset_names))


def discover_canonical_train_sources(pool_root: Path) -> list[Path]:
    return discover_canonical_train_sources_shared(pool_root)
