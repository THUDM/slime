from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "slime" / "rollout" / "rm_hub" / "code_verifier.py"
    spec = importlib.util.spec_from_file_location("mopd_code_verifier_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


code_verifier = _load_module()

MBPPPLUS_PATH = Path(
    "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/"
    "mopd-3node-h200-liteeval-noeval0-dist-h200-noroutingreplay-retry-0407-0633/data_cache/eval/mbppplus.jsonl"
)
HUMANEVALPLUS_PATH = Path(
    "/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/"
    "mopd-3node-h200-0406-1955/data_cache/eval/humanevalplus.jsonl"
)


def _first_mbppplus_row() -> dict:
    with MBPPPLUS_PATH.open("r", encoding="utf-8") as handle:
        return json.loads(next(handle))


def _first_humanevalplus_row() -> dict:
    with HUMANEVALPLUS_PATH.open("r", encoding="utf-8") as handle:
        return json.loads(next(handle))


def test_parse_test_code_harness_supports_mbppplus_layout():
    row = _first_mbppplus_row()
    harness = code_verifier.parse_test_code_harness(row["metadata"]["test_code"])

    assert harness is not None
    assert harness["loop_target"] == "(inp, exp)"
    assert harness["loop_iter"] == "zip(inputs, results)"
    assert "assertion(similar_elements(*inp), exp, 0)" in harness["loop_body"]


def test_parse_test_code_harness_supports_humanevalplus_layout():
    row = _first_humanevalplus_row()
    harness = code_verifier.parse_test_code_harness(row["metadata"]["test_code"])

    assert harness is not None
    assert harness["kind"] == "check_function"
    assert harness["candidate_param"] == "candidate"
    assert harness["loop_iter"] == "zip(inputs, results)"
    assert "assertion(candidate(*inp), exp, 0)" in harness["loop_body"]


def test_compute_score_supports_mbppplus_test_code_harness():
    row = _first_mbppplus_row()
    completion = """```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1).intersection(test_tup2))
```"""

    score, details = code_verifier.compute_score(completion, row["metadata"], continuous=True, max_partial_cases=10)

    assert score == 1.0
    assert details


def test_compute_score_supports_humanevalplus_check_candidate_harness():
    row = _first_humanevalplus_row()
    completion = """```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    nums = sorted(numbers)
    for left, right in zip(nums, nums[1:]):
        if right - left < threshold:
            return True
    return False
```"""

    score, details = code_verifier.compute_score(completion, row["metadata"], continuous=True, max_partial_cases=10)

    assert score == 1.0
    assert details


def test_compute_score_supports_ref_func_test_code_harness():
    metadata = {
        "test_code": "\n".join(
            [
                "def assertion(out, exp, atol):",
                "    assert out == exp",
                "",
                "def ref_func(x):",
                "    return x + 1",
                "",
                "inputs = [[1], [3], [9]]",
                "for i, inp in enumerate(inputs):",
                "    assertion(add_one(*inp), ref_func(*inp), 0)",
            ]
        )
    }
    completion = """```python
def add_one(x):
    return x + 1
```"""

    score, details = code_verifier.compute_score(completion, metadata, continuous=True, max_partial_cases=10)

    assert score == 1.0
    assert details
