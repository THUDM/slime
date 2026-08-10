"""Exact per-layer GLM-5 train/rollout alignment gate."""

import pytest

from test_glm52_6layer_deterministic_e2e import run_gate

NUM_GPUS = 8


def test_glm52_first_six_layers_match_exactly():
    # Keep this diagnostic gate short: the main 4096-token test owns the
    # realistic full-parameter training/logprob bound, while this run records
    # every visible decoder-layer boundary and requires bitwise equality.
    run_gate(layerwise_zero=True, rollout_max_response_len=32)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
