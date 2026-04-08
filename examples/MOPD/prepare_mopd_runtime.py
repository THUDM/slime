from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

DEFAULT_MATH_TRAIN_DATASETS = "deepmath,dapo,bigmath"
DEFAULT_CODE_TRAIN_DATASETS = "apps,code_contests,taco,codeforces"

AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
JL_MOPD_DIR = AVALANCHE_ROOT / "jl_workspace" / "experiment" / "mopd"


def _load_mopd_builder():
    jl_dir = str(JL_MOPD_DIR)
    if jl_dir not in sys.path:
        sys.path.insert(0, jl_dir)

    from build_mopd_runtime_data import (  # type: ignore
        MATH_SYSTEM_PROMPT,
        build_train,
        normalize_math_names,
        parse_code_names,
    )

    return {
        "build_train": build_train,
        "math_system_prompt": MATH_SYSTEM_PROMPT,
        "normalize_math_names": normalize_math_names,
        "parse_code_names": parse_code_names,
    }


def build_mopd_runtime_train_sources(
    *,
    pool_root: Path,
    cache_dir: Path,
    include_domains: Sequence[str],
    math_train_datasets: str = DEFAULT_MATH_TRAIN_DATASETS,
    code_train_datasets: str = DEFAULT_CODE_TRAIN_DATASETS,
) -> list[Path]:
    builder = _load_mopd_builder()
    outputs: list[Path] = []

    if "math" in include_domains:
        math_dest = cache_dir / "math" / "mopd_math_train.normalized.jsonl"
        builder["build_train"](
            math_pool_root=pool_root / "math",
            code_pool_root=pool_root / "code",
            math_dataset_names=builder["normalize_math_names"](math_train_datasets),
            code_dataset_names=[],
            dest=math_dest,
            math_system_prompt=builder["math_system_prompt"],
        )
        outputs.append(math_dest)

    if "code" in include_domains:
        code_dest = cache_dir / "code" / "mopd_code_train.normalized.jsonl"
        builder["build_train"](
            math_pool_root=pool_root / "math",
            code_pool_root=pool_root / "code",
            math_dataset_names=[],
            code_dataset_names=builder["parse_code_names"](code_train_datasets),
            dest=code_dest,
            math_system_prompt=builder["math_system_prompt"],
        )
        outputs.append(code_dest)

    return outputs
