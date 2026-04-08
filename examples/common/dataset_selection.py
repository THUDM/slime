from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from examples.common.dataset_registry import (
    TRAIN_DATASET_DOMAIN_MAP,
    TRAIN_DATASET_GROUP_MAP,
    TRAIN_DATASET_SOURCE_MAP,
    default_eval_datasets_for_profile,
    default_train_datasets_for_profile,
    eval_dataset_spec,
)
from examples.common.pool_data_utils import transform_jsonl, write_source_manifest
from examples.common.pool_runtime_semantics import materialize_runtime_pool_row


@dataclass(frozen=True)
class ResolvedEvalDataset:
    name: str
    source: Path
    runtime_mode: str
    n_samples_per_eval_prompt: int = 1
    official: bool = False


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(Path(normalized))
    return resolved


def _read_manifest(path: Path) -> list[Path]:
    sources: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                sources.append(Path(text))
    return sources


def _split_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_named_datasets(pool_root: Path, dataset_names: Sequence[str]) -> list[Path]:
    if not pool_root.exists():
        raise FileNotFoundError(f"Pool root does not exist: {pool_root}")
    sources: list[Path] = []
    for dataset_name in dataset_names:
        relpaths = TRAIN_DATASET_SOURCE_MAP.get(dataset_name)
        if relpaths is None:
            supported = ", ".join(sorted(TRAIN_DATASET_SOURCE_MAP))
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")
        for relpath in relpaths:
            candidate = pool_root / relpath
            if not candidate.is_file():
                raise FileNotFoundError(f"Dataset source does not exist for '{dataset_name}': {candidate}")
            sources.append(candidate)
    return _dedupe_paths(sources)


def _is_discoverable_source(path: Path, train_dir: Path) -> bool:
    try:
        relative_parts = path.relative_to(train_dir).parts
    except ValueError:
        return False
    return all(part and not part.startswith(".") for part in relative_parts)


def discover_canonical_train_sources(pool_root: Path) -> list[Path]:
    sources: list[Path] = []
    for relpaths in TRAIN_DATASET_SOURCE_MAP.values():
        for relpath in relpaths:
            candidate = pool_root / relpath
            if candidate.is_file():
                sources.append(candidate)
    return sorted(_dedupe_paths(sources))


def discover_train_sources(pool_root: Path) -> list[Path]:
    if not pool_root.exists():
        raise FileNotFoundError(f"Pool root does not exist: {pool_root}")
    canonical_sources = discover_canonical_train_sources(pool_root)
    if canonical_sources:
        return canonical_sources
    sources: list[Path] = []
    for subdir in ("structured", "stem", "tool"):
        train_dir = pool_root / subdir / "train"
        if not train_dir.is_dir():
            continue
        for path in train_dir.rglob("*.jsonl"):
            if not path.is_file():
                continue
            if not _is_discoverable_source(path, train_dir):
                continue
            if "_data_" in path.name:
                continue
            sources.append(path)
    return sorted(_dedupe_paths(sources))


def resolve_dataset_names(
    *,
    profile: str | None = None,
    datasets: Sequence[str] | None = None,
    dataset_extras: Sequence[str] | None = None,
    default_dataset_names: Sequence[str] | None = None,
    default_from_profile,
) -> tuple[str, ...]:
    if datasets:
        names = list(datasets)
    elif default_dataset_names is not None:
        names = list(default_dataset_names)
    elif profile:
        names = list(default_from_profile(profile))
    else:
        names = []
    if dataset_extras:
        names.extend(dataset_extras)
    ordered: list[str] = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def legacy_train_sources(
    pool_root: Path,
    include_domains: Sequence[str],
    exclude_patterns: Sequence[str],
) -> list[Path]:
    sources: list[Path] = []
    for domain in include_domains:
        domain_root = pool_root / domain
        if not domain_root.exists():
            continue
        if domain in {"tool", "stem", "structured"}:
            candidates = sorted((domain_root / "train").glob("*.jsonl"))
        else:
            candidates = sorted(domain_root.glob("*.jsonl"))
        for src_path in candidates:
            if src_path.name.startswith("._"):
                continue
            rel = src_path.relative_to(pool_root).as_posix()
            if any(pattern in rel for pattern in exclude_patterns):
                continue
            sources.append(src_path)
    return _dedupe_paths(sources)


def resolve_train_sources(
    *,
    pool_root: Path,
    profile: str | None = None,
    datasets: Sequence[str] | None = None,
    dataset_extras: Sequence[str] | None = None,
    paths: Sequence[str | Path] | None = None,
    path_extras: Sequence[str | Path] | None = None,
    manifest: str | Path | None = None,
    default_dataset_names: Sequence[str] | None = None,
    include_domains: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> list[Path]:
    dataset_names = resolve_dataset_names(
        profile=profile,
        datasets=datasets,
        dataset_extras=dataset_extras,
        default_dataset_names=default_dataset_names,
        default_from_profile=default_train_datasets_for_profile,
    )
    explicit_paths = [Path(path) for path in paths or []]
    extra_paths = [Path(path) for path in path_extras or []]
    manifest_paths = _read_manifest(Path(manifest)) if manifest else []

    sources: list[Path] = []
    if dataset_names:
        sources.extend(resolve_named_datasets(pool_root, dataset_names))
    if explicit_paths:
        sources.extend(explicit_paths)
    if manifest_paths:
        sources.extend(manifest_paths)
    if extra_paths:
        sources.extend(extra_paths)
    if sources:
        return _dedupe_paths(sources)
    if include_domains:
        return legacy_train_sources(pool_root, include_domains, exclude_patterns or [])
    return discover_train_sources(pool_root)


def materialize_file(src: Path, dst: Path) -> int:
    return transform_jsonl(
        src,
        dst,
        row_builder=materialize_runtime_pool_row,
        skip_invalid_json=True,
        open_errors="replace",
    )


def materialize_training_sources(
    *,
    sources: Sequence[Path],
    pool_root: Path,
    cache_dir: Path,
) -> list[Path]:
    materialized_paths: list[Path] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for src_path in sources:
        src_path = src_path.resolve()
        try:
            rel = src_path.relative_to(pool_root)
        except ValueError:
            rel = Path(src_path.name)
        dst_path = cache_dir / rel
        materialize_file(src_path, dst_path)
        materialized_paths.append(dst_path)
    return materialized_paths


def write_training_manifest(dest: Path, sources: Sequence[Path]) -> int:
    return write_source_manifest(sources, dest)


def infer_runtime_mode_from_path(pool_root: Path, source: Path) -> str:
    try:
        rel = source.resolve().relative_to(pool_root.resolve()).as_posix()
    except ValueError:
        rel = source.name
    if rel.startswith("code/"):
        return "code"
    if rel.startswith("math/"):
        return "math"
    return "pool"


def resolve_eval_datasets(
    *,
    pool_root: Path,
    profile: str | None = None,
    datasets: Sequence[str] | None = None,
    dataset_extras: Sequence[str] | None = None,
    paths: Sequence[str | Path] | None = None,
    path_extras: Sequence[str | Path] | None = None,
    default_dataset_names: Sequence[str] | None = None,
) -> list[ResolvedEvalDataset]:
    dataset_names = resolve_dataset_names(
        profile=profile,
        datasets=datasets,
        dataset_extras=dataset_extras,
        default_dataset_names=default_dataset_names,
        default_from_profile=default_eval_datasets_for_profile,
    )
    resolved: list[ResolvedEvalDataset] = []
    seen: set[str] = set()
    for dataset_name in dataset_names:
        spec = eval_dataset_spec(dataset_name)
        source = (pool_root / spec.relpath).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Eval dataset source does not exist for '{dataset_name}': {source}")
        normalized = str(source)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(
            ResolvedEvalDataset(
                name=spec.name,
                source=source,
                runtime_mode=spec.runtime_mode,
                n_samples_per_eval_prompt=spec.n_samples_per_eval_prompt,
                official=spec.official,
            )
        )
    for raw_path in [Path(item) for item in paths or []] + [Path(item) for item in path_extras or []]:
        source = raw_path.resolve()
        normalized = str(source)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(
            ResolvedEvalDataset(
                name=raw_path.stem,
                source=source,
                runtime_mode=infer_runtime_mode_from_path(pool_root, source),
                n_samples_per_eval_prompt=1,
                official=False,
            )
        )
    return resolved


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


def train_domains_for_datasets(dataset_names: Sequence[str]) -> tuple[str, ...]:
    domains: list[str] = []
    for dataset_name in dataset_names:
        try:
            domain = TRAIN_DATASET_DOMAIN_MAP[dataset_name]
        except KeyError as exc:
            supported = ", ".join(sorted(TRAIN_DATASET_DOMAIN_MAP))
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}") from exc
        if domain not in domains:
            domains.append(domain)
    return tuple(domains)


def train_groups_for_datasets(dataset_names: Sequence[str]) -> tuple[str, ...]:
    groups: list[str] = []
    for dataset_name in dataset_names:
        try:
            group = TRAIN_DATASET_GROUP_MAP[dataset_name]
        except KeyError as exc:
            supported = ", ".join(sorted(TRAIN_DATASET_GROUP_MAP))
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}") from exc
        if group not in groups:
            groups.append(group)
    return tuple(groups)


def domain_signature_for_train_datasets(dataset_names: Sequence[str]) -> str:
    return domain_signature(train_domains_for_datasets(dataset_names))


def group_signature_for_train_datasets(dataset_names: Sequence[str]) -> str:
    return group_signature(train_groups_for_datasets(dataset_names))
