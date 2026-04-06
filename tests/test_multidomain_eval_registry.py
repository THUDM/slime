from __future__ import annotations

import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from multidomain_shared import GENERIC_EVAL_DATASETS, OFFICIAL_EVAL_DATASETS


def test_generic_eval_registry_includes_ifbench_test() -> None:
    assert (
        "ifbench_test",
        "structured/eval/ifbench_test_data_train-00000-of-00001.jsonl",
        1,
    ) in GENERIC_EVAL_DATASETS


def test_ifbench_test_is_not_marked_official_only() -> None:
    assert ("ifbench_test", "structured/eval/ifbench_test_data_train-00000-of-00001.jsonl") not in OFFICIAL_EVAL_DATASETS
