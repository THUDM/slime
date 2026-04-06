from __future__ import annotations

from pathlib import Path


def test_submit_script_forwards_explicit_train_and_eval_overrides():
    script_path = Path(__file__).resolve().parents[1] / "examples" / "MOPD" / "submit_mopd_qwen3_30b_3node_h200.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "MOPD_STEM_TRAIN_DATASETS MOPD_STRUCTURED_TRAIN_DATASETS MOPD_MATH_TRAIN_DATASETS MOPD_CODE_TRAIN_DATASETS" in script_text
    assert "TRAIN_DATASETS TRAIN_DATASETS_EXTRA TRAIN_PATHS TRAIN_PATHS_EXTRA TRAIN_MANIFEST" in script_text
    assert "TEACHER_STEP0_INCLUDE_BFCL EVAL_DATASETS EVAL_DATASETS_EXTRA EVAL_PATHS EVAL_PATHS_EXTRA" in script_text
