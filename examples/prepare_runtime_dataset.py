#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

SLIME_ROOT = Path(__file__).resolve().parents[1]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

from examples.MOPD.prepare_mopd_runtime import (
    DEFAULT_CODE_TRAIN_DATASETS,
    DEFAULT_MATH_TRAIN_DATASETS,
    build_mopd_runtime_train_sources,
)
from examples.common.dataset_registry import official_eval_dataset_names
from examples.common.dataset_selection import (
    discover_train_sources,
    materialize_training_sources,
    resolve_eval_datasets,
    resolve_train_sources as resolve_train_sources_common,
    resolve_named_datasets as resolve_named_datasets_from_registry,
    train_domains_for_datasets,
    write_training_manifest,
)
from examples.common.eval_prep_utils import materialize_eval_dataset as materialize_eval_dataset_shared
from examples.common.pool_data_utils import iter_json_object_rows, write_source_manifest as write_source_manifest_shared
from examples.common.pool_runtime_semantics import materialize_runtime_pool_row


def _split_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def discover_sources(pool_root: Path) -> list[Path]:
    return discover_train_sources(pool_root)


def align_row_to_v1_normalized_shape(row: dict[str, Any]) -> dict[str, Any] | None:
    return materialize_runtime_pool_row(row)


def resolve_named_datasets(pool_root: Path, dataset_names: Sequence[str]) -> list[Path]:
    return resolve_named_datasets_from_registry(pool_root, list(dataset_names))


def iter_selected_rows(
    sources: Sequence[Path],
    skip_samples: int = 0,
    max_samples: int | None = None,
) -> Iterator[dict]:
    seen = 0
    yielded = 0
    for source in sources:
        for row in iter_json_object_rows(source):
            if seen < skip_samples:
                seen += 1
                continue
            if max_samples is not None and yielded >= max_samples:
                return
            yield row
            yielded += 1


def write_dataset(
    sources: Sequence[Path],
    dest: Path,
    skip_samples: int = 0,
    max_samples: int | None = None,
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("w", encoding="utf-8") as handle:
        for row in iter_selected_rows(sources, skip_samples=skip_samples, max_samples=max_samples):
            row = align_row_to_v1_normalized_shape(row)
            if row is None:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def write_source_manifest(sources: Sequence[Path], dest: Path) -> int:
    return write_source_manifest_shared(sources, dest)


def build_training_manifest(
    *,
    pool_root: Path,
    cache_dir: Path,
    dest_manifest: Path,
    profile: str | None = None,
    datasets: Sequence[str] | None = None,
    dataset_extras: Sequence[str] | None = None,
    paths: Sequence[str | Path] | None = None,
    path_extras: Sequence[str | Path] | None = None,
    manifest: str | Path | None = None,
    include_domains: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
    stem_train_datasets: str = "nemotron_knowledge_mcqa",
    structured_train_datasets: str = "nemotron_structured_outputs",
    math_train_datasets: str = DEFAULT_MATH_TRAIN_DATASETS,
    code_train_datasets: str = DEFAULT_CODE_TRAIN_DATASETS,
) -> int:
    explicit_selection = any((profile, datasets, dataset_extras, paths, path_extras, manifest))
    selected_domains: list[str] = []
    explicit_dataset_names = [*(datasets or []), *(dataset_extras or [])]
    if explicit_dataset_names:
        selected_domains.extend(train_domains_for_datasets(explicit_dataset_names))
    elif explicit_selection:
        resolved_sources = resolve_train_sources_common(
            pool_root=pool_root,
            profile=profile,
            datasets=datasets,
            dataset_extras=dataset_extras,
            paths=paths,
            path_extras=path_extras,
            manifest=manifest,
        )
        for source in resolved_sources:
            try:
                domain = source.relative_to(pool_root).parts[0]
            except ValueError:
                continue
            if domain not in selected_domains:
                selected_domains.append(domain)
    if include_domains:
        for domain in include_domains:
            if domain not in selected_domains:
                selected_domains.append(domain)

    materialized_paths = build_mopd_runtime_train_sources(
        pool_root=pool_root,
        cache_dir=cache_dir,
        include_domains=selected_domains,
        math_train_datasets=math_train_datasets,
        code_train_datasets=code_train_datasets,
    )

    overlap_domains = [domain for domain in selected_domains if domain not in {"math", "code"}]
    if explicit_selection:
        overlap_sources = resolve_train_sources_common(
            pool_root=pool_root,
            profile=profile,
            datasets=datasets,
            dataset_extras=dataset_extras,
            paths=paths,
            path_extras=path_extras,
            manifest=manifest,
        )
        overlap_sources = [
            source for source in overlap_sources if source.relative_to(pool_root).parts[0] not in {"math", "code"}
        ]
    else:
        default_overlap_dataset_names: list[str] = []
        if "stem" in overlap_domains:
            default_overlap_dataset_names.extend(_split_csv(stem_train_datasets))
        if "structured" in overlap_domains:
            default_overlap_dataset_names.extend(_split_csv(structured_train_datasets))
        overlap_sources = resolve_train_sources_common(
            pool_root=pool_root,
            default_dataset_names=default_overlap_dataset_names or None,
            include_domains=overlap_domains,
            exclude_patterns=exclude_patterns,
        )

    materialized_paths.extend(
        materialize_training_sources(
            sources=overlap_sources,
            pool_root=pool_root,
            cache_dir=cache_dir,
        )
    )
    return write_training_manifest(dest_manifest, materialized_paths)


def materialize_eval_data(
    pool_root: Path,
    dest: Path,
    max_samples: int | None = None,
    dataset_name: str | None = None,
    source: Path | None = None,
    profile: str | None = None,
) -> int:
    resolved = resolve_eval_datasets(
        pool_root=pool_root,
        profile=profile,
        datasets=[dataset_name] if dataset_name else None,
        paths=[source] if source is not None else None,
    )
    if len(resolved) != 1:
        raise ValueError("Expected exactly one eval dataset to resolve.")
    return materialize_eval_dataset_shared(
        resolved[0].source,
        dest,
        runtime_mode=resolved[0].runtime_mode,
        max_samples=max_samples,
    )


def _bfcl_single_turn_filter(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    test_category = str(metadata.get("test_category", "")).lower()
    return "multi_turn" not in test_category


def write_eval_config_bundle(
    *,
    pool_root: Path,
    output: Path,
    max_response_len: int,
    profile: str | None = None,
    datasets: Sequence[str] | None = None,
    dataset_extras: Sequence[str] | None = None,
    paths: Sequence[str | Path] | None = None,
    path_extras: Sequence[str | Path] | None = None,
    official_manifest_output: Path | None = None,
) -> None:
    output_dir = output.parent
    eval_data_dir = output_dir / "eval"
    eval_data_dir.mkdir(parents=True, exist_ok=True)

    dataset_entries: list[dict[str, Any]] = []
    resolved_datasets = resolve_eval_datasets(
        pool_root=pool_root,
        profile=profile,
        datasets=datasets,
        dataset_extras=dataset_extras,
        paths=paths,
        path_extras=path_extras,
    )
    for resolved in resolved_datasets:
        if resolved.official:
            continue
        src_path = resolved.source
        if not src_path.exists():
            continue
        dst_path = eval_data_dir / f"{resolved.name}.jsonl"
        row_filter = _bfcl_single_turn_filter if resolved.name == "bfcl_v3" else None
        count = materialize_eval_dataset_shared(
            src_path,
            dst_path,
            runtime_mode=resolved.runtime_mode,
            row_filter=row_filter,
        )
        if count == 0:
            continue
        dataset_entries.append(
            {
                "name": resolved.name,
                "path": str(dst_path),
                "n_samples_per_eval_prompt": resolved.n_samples_per_eval_prompt,
            }
        )

    payload = {
        "eval": {
            "defaults": {
                "input_key": "prompt",
                "label_key": "label",
                "metadata_key": "metadata",
                "tool_key": "tools",
                "max_response_len": max_response_len,
                "top_k": 1,
            },
            "datasets": dataset_entries,
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    if official_manifest_output is None:
        return

    official_datasets: list[dict[str, Any]] = []
    official_specs = resolve_eval_datasets(
        pool_root=pool_root,
        datasets=official_eval_dataset_names(),
    )
    for resolved in official_specs:
        src_path = resolved.source
        if not src_path.exists():
            continue
        dst_path = eval_data_dir / f"{resolved.name}.jsonl"
        row_filter = _bfcl_single_turn_filter if resolved.name == "bfcl_v3" else None
        count = materialize_eval_dataset_shared(
            src_path,
            dst_path,
            runtime_mode=resolved.runtime_mode,
            row_filter=row_filter,
        )
        if count == 0:
            continue
        official_datasets.append({"name": resolved.name, "path": str(dst_path), "n_samples": count})

    official_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with official_manifest_output.open("w", encoding="utf-8") as handle:
        json.dump({"datasets": official_datasets}, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared runtime-data preparation CLI for multidomain examples.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Resolve and materialize runtime training data.")
    train_parser.add_argument("--source", action="append", default=None)
    train_parser.add_argument("--source-extra", action="append", default=None)
    train_parser.add_argument("--dataset", action="append", default=None)
    train_parser.add_argument("--dataset-extra", action="append", default=None)
    train_parser.add_argument("--manifest", default=None)
    train_parser.add_argument("--pool-root", default=None)
    train_parser.add_argument("--profile", default=None)
    train_parser.add_argument("--dest", default=None)
    train_parser.add_argument("--cache-dir", default=None)
    train_parser.add_argument("--manifest-output", default=None)
    train_parser.add_argument("--include-domains", default=None)
    train_parser.add_argument("--exclude-patterns", default=None)
    train_parser.add_argument("--stem-train-datasets", default="nemotron_knowledge_mcqa")
    train_parser.add_argument("--structured-train-datasets", default="nemotron_structured_outputs")
    train_parser.add_argument("--math-train-datasets", default=DEFAULT_MATH_TRAIN_DATASETS)
    train_parser.add_argument("--code-train-datasets", default=DEFAULT_CODE_TRAIN_DATASETS)
    train_parser.add_argument("--print-sources", action="store_true", default=False)
    train_parser.add_argument("--skip-samples", type=int, default=0)
    train_parser.add_argument("--max-samples", type=int, default=None)

    eval_parser = subparsers.add_parser("eval", help="Materialize one eval dataset into runtime JSONL.")
    eval_parser.add_argument("--pool-root", required=True)
    eval_parser.add_argument("--profile", default=None)
    eval_parser.add_argument("--dataset", default=None)
    eval_parser.add_argument("--source", default=None)
    eval_parser.add_argument("--dest", required=True)
    eval_parser.add_argument("--max-samples", type=int, default=None)

    eval_config_parser = subparsers.add_parser("eval-config", help="Materialize runtime eval datasets and emit eval YAML.")
    eval_config_parser.add_argument("--pool-root", required=True)
    eval_config_parser.add_argument("--output", required=True)
    eval_config_parser.add_argument("--official-manifest-output", default=None)
    eval_config_parser.add_argument("--max-response-len", type=int, default=8192)
    eval_config_parser.add_argument("--profile", default=None)
    eval_config_parser.add_argument("--dataset", action="append", default=None)
    eval_config_parser.add_argument("--dataset-extra", action="append", default=None)
    eval_config_parser.add_argument("--source", action="append", default=None)
    eval_config_parser.add_argument("--source-extra", action="append", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        pool_root = Path(args.pool_root) if args.pool_root else None
        if args.manifest_output:
            if pool_root is None:
                raise SystemExit("--pool-root is required with --manifest-output.")
            if not args.cache_dir:
                raise SystemExit("--cache-dir is required with --manifest-output.")
            build_training_manifest(
                pool_root=pool_root,
                cache_dir=Path(args.cache_dir),
                dest_manifest=Path(args.manifest_output),
                profile=args.profile,
                datasets=args.dataset,
                dataset_extras=args.dataset_extra,
                paths=args.source,
                path_extras=args.source_extra,
                manifest=args.manifest,
                include_domains=_split_csv(args.include_domains),
                exclude_patterns=_split_csv(args.exclude_patterns),
                stem_train_datasets=args.stem_train_datasets,
                structured_train_datasets=args.structured_train_datasets,
                math_train_datasets=args.math_train_datasets,
                code_train_datasets=args.code_train_datasets,
            )
            return
        if pool_root is None:
            if args.source:
                sources = [Path(source) for source in args.source]
            else:
                raise SystemExit("Provide --pool-root when using dataset-based train resolution.")
        else:
            sources = resolve_train_sources_common(
                pool_root=pool_root,
                profile=args.profile,
                datasets=args.dataset,
                dataset_extras=args.dataset_extra,
                paths=args.source,
                path_extras=args.source_extra,
                manifest=args.manifest,
            )
        if not sources:
            raise SystemExit("No normalized pool sources were found.")
        if args.print_sources:
            for source in sources:
                print(source)
            return
        if not args.dest:
            raise SystemExit("--dest is required unless --print-sources is set.")
        write_dataset(
            sources,
            Path(args.dest),
            skip_samples=args.skip_samples,
            max_samples=args.max_samples,
        )
        return

    if args.command == "eval-config":
        write_eval_config_bundle(
            pool_root=Path(args.pool_root),
            output=Path(args.output),
            max_response_len=args.max_response_len,
            profile=args.profile,
            datasets=args.dataset,
            dataset_extras=args.dataset_extra,
            paths=args.source,
            path_extras=args.source_extra,
            official_manifest_output=Path(args.official_manifest_output) if args.official_manifest_output else None,
        )
        return

    if not args.dataset and not args.source:
        raise SystemExit("Provide either --dataset or --source.")
    count = materialize_eval_data(
        pool_root=Path(args.pool_root),
        dest=Path(args.dest),
        max_samples=args.max_samples,
        dataset_name=args.dataset,
        source=Path(args.source) if args.source else None,
        profile=args.profile,
    )
    label = args.dataset or args.source
    print(f"Materialized {count} rows for {label} to {args.dest}")


if __name__ == "__main__":
    main()
