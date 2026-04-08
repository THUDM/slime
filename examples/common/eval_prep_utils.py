from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from examples.common.pool_data_utils import transform_jsonl
from examples.common.pool_runtime_semantics import materialize_runtime_pool_row


AVALANCHE_ROOT = Path(__file__).resolve().parents[4]
JL_EXPERIMENT_ROOT = AVALANCHE_ROOT / "jl_workspace" / "experiment"
JL_MATH_DIR = JL_EXPERIMENT_ROOT / "math"
JL_CODE_DIR = JL_EXPERIMENT_ROOT / "code"

for path in (JL_MATH_DIR, JL_CODE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from build_math_runtime_data import (  # type: ignore  # noqa: E402
    DEFAULT_SYSTEM_PROMPT as MATH_SYSTEM_PROMPT,
    ensure_math_metadata,
    make_prompt as make_math_prompt,
)
from build_code_runtime_data import (  # type: ignore  # noqa: E402
    build_prompt as build_code_prompt,
    ensure_code_metadata,
)


def build_math_eval_row(row: dict) -> dict | None:
    question = str(row.get("question") or "").strip()
    if not question:
        return None

    label = row.get("label", "")
    if isinstance(label, list):
        if len(label) != 1:
            return None
        label = label[0]
    label = str(label).strip()
    if not label:
        return None

    metadata = ensure_math_metadata(row.get("metadata"))
    metadata.setdefault("dataset_key", str(metadata.get("dataset_name") or "math").strip().lower())

    runtime_row = {
        "prompt": make_math_prompt(question, MATH_SYSTEM_PROMPT),
        "label": label,
        "metadata": metadata,
    }
    if row.get("id") not in (None, ""):
        runtime_row["id"] = row["id"]
    return runtime_row


def build_code_eval_row(row: dict) -> dict | None:
    try:
        prompt = build_code_prompt(row)
    except Exception:
        return None

    metadata = ensure_code_metadata(row.get("metadata"))
    metadata.setdefault("dataset_key", str(metadata.get("dataset_name") or "code").strip().lower())

    runtime_row = {
        "prompt": prompt,
        "label": row.get("label", ""),
        "metadata": metadata,
    }
    if row.get("id") not in (None, ""):
        runtime_row["id"] = row["id"]
    return runtime_row


def runtime_row_builder_for_mode(runtime_mode: str) -> Callable[[dict], dict | None]:
    if runtime_mode == "math":
        return build_math_eval_row
    if runtime_mode == "code":
        return build_code_eval_row
    return materialize_runtime_pool_row


def materialize_eval_dataset(
    src: Path,
    dst: Path,
    *,
    runtime_mode: str,
    max_samples: int | None = None,
    row_filter=None,
) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    row_builder = runtime_row_builder_for_mode(runtime_mode)
    written = 0

    def _bounded_row_builder(row: dict) -> dict | None:
        nonlocal written
        if max_samples is not None and written >= max_samples:
            return None
        payload = row_builder(row)
        if payload is None:
            return None
        if row_filter is not None and not row_filter(payload):
            return None
        written += 1
        return payload

    transform_jsonl(
        src,
        dst,
        row_builder=_bounded_row_builder,
        skip_invalid_json=True,
    )
    return written
