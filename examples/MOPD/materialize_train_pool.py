"""Materialize MOPD training sources into runtime JSONL files when needed."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_slime_root = Path(__file__).resolve().parents[2]
if str(_slime_root) not in sys.path:
    sys.path.insert(0, str(_slime_root))

AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
JL_MOPD_DIR = AVALANCHE_ROOT / "jl_workspace" / "experiment" / "mopd"
if str(JL_MOPD_DIR) not in sys.path:
    sys.path.insert(0, str(JL_MOPD_DIR))

from build_mopd_runtime_data import (  # type: ignore  # noqa: E402
    DEFAULT_CODE_TRAIN_DATASETS,
    DEFAULT_MATH_TRAIN_DATASETS,
    MATH_SYSTEM_PROMPT,
    build_train as build_mopd_train,
    normalize_math_names,
    parse_code_names,
)
from examples.common.dataset_selection import (
    materialize_training_sources,
    resolve_train_sources,
    write_training_manifest,
)


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _build_jl_runtime_train(
    *,
    pool_root: Path,
    cache_dir: Path,
    include_domains: list[str],
    math_train_datasets: str,
    code_train_datasets: str,
) -> list[Path]:
    outputs: list[Path] = []

    if "math" in include_domains:
        math_dest = cache_dir / "math" / "mopd_math_train.normalized.jsonl"
        counts = build_mopd_train(
            math_pool_root=pool_root / "math",
            code_pool_root=pool_root / "code",
            math_dataset_names=normalize_math_names(math_train_datasets),
            code_dataset_names=[],
            dest=math_dest,
            math_system_prompt=MATH_SYSTEM_PROMPT,
        )
        print(f"  built jl math runtime train -> {math_dest}")
        for name, count in counts.items():
            print(f"    {name}: {count} rows")
        outputs.append(math_dest)

    if "code" in include_domains:
        code_dest = cache_dir / "code" / "mopd_code_train.normalized.jsonl"
        counts = build_mopd_train(
            math_pool_root=pool_root / "math",
            code_pool_root=pool_root / "code",
            math_dataset_names=[],
            code_dataset_names=parse_code_names(code_train_datasets),
            dest=code_dest,
            math_system_prompt=MATH_SYSTEM_PROMPT,
        )
        print(f"  built jl code runtime train -> {code_dest}")
        for name, count in counts.items():
            print(f"    {name}: {count} rows")
        outputs.append(code_dest)

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pool_root")
    parser.add_argument("cache_dir")
    parser.add_argument("dest_list")
    parser.add_argument("include_domains")
    parser.add_argument("exclude_patterns")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--dataset-extra", action="append", default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--source-extra", action="append", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--stem-train-datasets", default="nemotron_knowledge_mcqa")
    parser.add_argument("--structured-train-datasets", default="nemotron_structured_outputs")
    parser.add_argument("--math-train-datasets", default=DEFAULT_MATH_TRAIN_DATASETS)
    parser.add_argument("--code-train-datasets", default=DEFAULT_CODE_TRAIN_DATASETS)
    args = parser.parse_args()

    pool_root = Path(args.pool_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    dest_list = Path(args.dest_list).resolve()
    include_domains = [d.strip() for d in args.include_domains.split(",") if d.strip()]
    exclude_patterns = [p.strip() for p in args.exclude_patterns.split(",") if p.strip()]

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_list.parent.mkdir(parents=True, exist_ok=True)

    materialized_paths: list[Path] = []
    materialized_paths.extend(
        _build_jl_runtime_train(
            pool_root=pool_root,
            cache_dir=cache_dir,
            include_domains=include_domains,
            math_train_datasets=args.math_train_datasets,
            code_train_datasets=args.code_train_datasets,
        )
    )

    explicit_overlap_requested = any(
        (
            args.profile,
            args.dataset,
            args.dataset_extra,
            args.source,
            args.source_extra,
            args.manifest,
        )
    )
    overlap_domains = [domain for domain in include_domains if domain not in {"math", "code"}]
    if explicit_overlap_requested:
        overlap_sources = resolve_train_sources(
            pool_root=pool_root,
            profile=args.profile,
            datasets=args.dataset,
            dataset_extras=args.dataset_extra,
            paths=args.source,
            path_extras=args.source_extra,
            manifest=args.manifest,
        )
    else:
        default_overlap_dataset_names: list[str] = []
        if "stem" in overlap_domains:
            default_overlap_dataset_names.extend(_split_csv(args.stem_train_datasets))
        if "structured" in overlap_domains:
            default_overlap_dataset_names.extend(_split_csv(args.structured_train_datasets))
        overlap_sources = resolve_train_sources(
            pool_root=pool_root,
            default_dataset_names=default_overlap_dataset_names or None,
            include_domains=overlap_domains,
            exclude_patterns=exclude_patterns,
        )

    overlap_materialized_paths = materialize_training_sources(
        sources=overlap_sources,
        pool_root=pool_root,
        cache_dir=cache_dir,
    )
    for source_path, resolved_path in zip(overlap_sources, overlap_materialized_paths, strict=False):
        if resolved_path != source_path.resolve():
            print(f"  materialized {source_path.relative_to(pool_root)} -> {resolved_path}")
    materialized_paths.extend(overlap_materialized_paths)

    write_training_manifest(dest_list, materialized_paths)

    print(f"wrote {len(materialized_paths)} training sources to {dest_list}")
    for path in materialized_paths:
        print(path)


if __name__ == "__main__":
    main()
