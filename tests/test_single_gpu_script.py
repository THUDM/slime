from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run-qwen3-0.6B-single-gpu.sh"


@pytest.mark.unit
def test_single_gpu_script_covers_standard_grpo_train_eval_and_resume_artifacts():
    source = SCRIPT.read_text()

    required_arguments = (
        "--actor-num-gpus-per-node 1",
        "--colocate",
        '--num-rollout "$NUM_ROLLOUT"',
        "--advantage-estimator grpo",
        '--eval-prompt-data gsm8k "$EVAL_DATA"',
        '--load "$LOAD_CHECKPOINT"',
        '--save "$SAVE_DIR"',
        '--save-interval "$SAVE_INTERVAL"',
        "--tensor-model-parallel-size 1",
        "--pipeline-model-parallel-size 1",
        "--rollout-num-gpus-per-engine 1",
    )
    for argument in required_arguments:
        assert argument in source

    assert 'source "${SCRIPT_DIR}/models/qwen3-0.6B.sh"' in source
    assert 'if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]' in source
    assert "--debug-train-only" not in source
    assert "--debug-rollout-only" not in source
