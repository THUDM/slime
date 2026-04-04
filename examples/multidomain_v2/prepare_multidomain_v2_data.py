#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from pool_runtime_semantics import materialize_runtime_pool_row


POOL_SUBDIRS = ("structured", "stem", "tool")
DATASET_SOURCE_MAP: dict[str, tuple[str, ...]] = {
    "toolbench_v1": (
        "tool/toolbench_v1_train-00000-of-00004.jsonl",
        "tool/toolbench_v1_train-00001-of-00004.jsonl",
        "tool/toolbench_v1_train-00002-of-00004.jsonl",
        "tool/toolbench_v1_train-00003-of-00004.jsonl",
    ),
    "apibench": (
        "tool/apibench_huggingface_train.jsonl",
        "tool/apibench_tensorflow_train.jsonl",
        "tool/apibench_torchhub_train.jsonl",
    ),
    "apigen": ("tool/apigen_mt_5k_apigen-mt_5k.jsonl",),
    "xlam_function_calling_60k": ("tool/xlam_function_calling_60k_xlam-function-calling-60k.jsonl",),
    "agent": (
        "tool/agent_function_calling_open_dataset_deepnlp_agent_function_call_202510.jsonl",
        "tool/agent_function_calling_open_dataset_deepnlp_agent_function_call_202601.jsonl",
    ),
    "jsonschemabench": ("structured/jsonschemabench_train-00000-of-00001.jsonl",),
    "ifeval": ("structured/eval/ifeval_ifeval_input_data.jsonl",),
    "ifbench_test": ("structured/eval/ifbench_test_train-00000-of-00001.jsonl",),
    "nemotron_structured_outputs": ("structured/nemotron_structured_outputs_structured_outputs_251027_nano_v3_sdg_json_train.jsonl",),
    "nemotron_knowledge_mcqa": (
        "stem/nemotron_knowledge_mcqa_data_train-00000-of-00004.jsonl",
        "stem/nemotron_knowledge_mcqa_data_train-00001-of-00004.jsonl",
        "stem/nemotron_knowledge_mcqa_data_train-00002-of-00004.jsonl",
        "stem/nemotron_knowledge_mcqa_data_train-00003-of-00004.jsonl",
    ),
    "medmcqa": ("stem/medmcqa_data_train-00000-of-00001.jsonl",),
}


def discover_sources(pool_root: Path) -> list[Path]:
    if not pool_root.exists():
        raise FileNotFoundError(f"Pool root does not exist: {pool_root}")
    sources: list[Path] = []
    for subdir in POOL_SUBDIRS:
        candidate = pool_root / subdir
        if not candidate.is_dir():
            continue
        train_dir = candidate / "train"
        if not train_dir.is_dir():
            continue
        for path in train_dir.rglob("*.jsonl"):
            if not path.is_file():
                continue
            if "_data_" in path.name:
                continue
            sources.append(path)
    return sorted(sources)


def iter_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield payload


def align_row_to_v1_normalized_shape(row: dict[str, Any]) -> dict[str, Any] | None:
    return materialize_runtime_pool_row(row)


def _expand_layout_candidates(relative_path: str) -> list[str]:
    # Prefer the current pool layout (<domain>/train/*.jsonl), but keep backward
    # compatibility with the old <domain>/*.jsonl structure.
    if "/train/" in relative_path:
        return [relative_path, relative_path.replace("/train/", "/", 1)]
    head, sep, tail = relative_path.partition("/")
    if not sep:
        return [relative_path]
    return [f"{head}/train/{tail}", relative_path]


def resolve_named_datasets(pool_root: Path, dataset_names: Sequence[str]) -> list[Path]:
    if not pool_root.exists():
        raise FileNotFoundError(f"Pool root does not exist: {pool_root}")

    sources: list[Path] = []
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_SOURCE_MAP:
            supported = ", ".join(sorted(DATASET_SOURCE_MAP))
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")
        for relative_path in DATASET_SOURCE_MAP[dataset_name]:
            resolved = None
            tried: list[Path] = []
            for candidate_relpath in _expand_layout_candidates(relative_path):
                candidate = pool_root / candidate_relpath
                tried.append(candidate)
                if candidate.is_file():
                    resolved = candidate
                    break
            if resolved is None:
                tried_str = ", ".join(str(path) for path in tried)
                raise FileNotFoundError(
                    f"Dataset source does not exist for '{dataset_name}': expected one of [{tried_str}]"
                )
            sources.append(resolved)
    return sources


def iter_selected_rows(
    sources: Sequence[Path],
    skip_samples: int = 0,
    max_samples: int | None = None,
) -> Iterator[dict]:
    seen = 0
    yielded = 0
    for source in sources:
        for row in iter_rows(source):
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for source in sources:
            handle.write(f"{source}\n")
    return len(sources)


def _resolve_sources(args: argparse.Namespace) -> list[Path]:
    explicit_sources = [Path(source) for source in args.source or []]
    if explicit_sources:
        return explicit_sources
    named_datasets = [dataset for dataset in args.dataset or [] if dataset]
    if named_datasets:
        if not args.pool_root:
            raise ValueError("--dataset requires --pool-root.")
        return resolve_named_datasets(Path(args.pool_root), named_datasets)
    if args.pool_root:
        return discover_sources(Path(args.pool_root))
    raise ValueError("Provide at least one --source or set --pool-root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve or concatenate pre-normalized multidomain v2 pool data.")
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--pool-root", default=None)
    parser.add_argument("--dest", default=None)
    parser.add_argument("--print-sources", action="store_true", default=False)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = _resolve_sources(args)
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


if __name__ == "__main__":
    main()
