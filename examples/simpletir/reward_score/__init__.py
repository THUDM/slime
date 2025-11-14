from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    from . import hf_math_verify

    _HF_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    hf_math_verify = None
    _HF_IMPORT_ERROR = exc


async def compute_reward_async(
    data_source: str,
    solution: str,
    ground_truth: str,
    extra_info: Any = None,
) -> Dict[str, Any]:
    """Route reward computation based on dataset/source name."""
    if data_source is None:
        raise ValueError("data_source is required for reward computation.")

    if "simplelr_math_35" in data_source or "deepscaler" in data_source:
        if hf_math_verify is None:
            raise RuntimeError(
                "math-verify is required for SimpleTIR math rewards. "
                "Please install the optional dependency or remove the math datasets."
            ) from _HF_IMPORT_ERROR
        return hf_math_verify.compute_score(solution, ground_truth)
    raise ValueError(f"Unknown data source: {data_source}")
