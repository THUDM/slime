"""Pre-materialize pool/train data so that downstream consumers
can read the prompt and metadata in the runtime shape expected by training.

New-format pool rows carry supervision in top-level family-specific fields
plus ``supervision_family`` instead of ``metadata.ground_truth``. 
``materialize_runtime_pool_row`` bridges the gap, but
``slime/slime/utils/data.py`` does not call it during training.

For MOPD this is still useful even though the reward comes from the teacher
model: materialization injects the runtime prompt shape and family-specific
metadata before the generic data loader reads the files.

This script reads pool JSONL files, runs materialization, and writes the result
to a cache directory.  The output ``.list`` file points to the materialized
files so the training pipeline can load them transparently.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing from examples/
_examples_dir = Path(__file__).resolve().parent.parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
JL_MOPD_DIR = AVALANCHE_ROOT / "jl_workspace" / "experiment" / "mopd"
if str(JL_MOPD_DIR) not in sys.path:
    sys.path.insert(0, str(JL_MOPD_DIR))

from pool_data_utils import (
    file_has_supervision_family,
    transform_jsonl,
)
from pool_runtime_semantics import materialize_runtime_pool_row  # noqa: E402
from build_mopd_runtime_data import (  # type: ignore  # noqa: E402
    DEFAULT_CODE_TRAIN_DATASETS,
    DEFAULT_MATH_TRAIN_DATASETS,
    MATH_SYSTEM_PROMPT,
    build_train as build_mopd_train,
    normalize_math_names,
    parse_code_names,
)
from dataset_selection import (
    legacy_train_sources,
    materialize_training_sources,
    resolve_train_sources,
    write_training_manifest,
)


def materialize_file(src: Path, dst: Path) -> int:
    """Materialize a single pool JSONL file.  Returns number of rows written."""
    return transform_jsonl(
        src,
        dst,
        row_builder=materialize_runtime_pool_row,
        skip_invalid_json=True,
        open_errors="replace",
    )


def _file_needs_materialize(src: Path) -> bool:
    return file_has_supervision_family(src)


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
        overlap_sources = legacy_train_sources(pool_root, overlap_domains, exclude_patterns)

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
