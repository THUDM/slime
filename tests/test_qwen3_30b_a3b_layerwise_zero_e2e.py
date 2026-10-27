"""Layerwise-zero gate for Qwen3-30B-A3B train/rollout alignment.

Delegates to test_qwen3_30b_a3b_deterministic_e2e.run_gate(layerwise_zero=True),
which dumps per-layer hidden states from both Megatron and SGLang and asserts
bit-exact equality (--max-hidden-diff 0) across all 48 MoE layers.

Run:
    pytest tests/test_qwen3_30b_a3b_layerwise_zero_e2e.py -s

Environment overrides (all optional):
    HF_MODEL        -- path to Qwen3-30B-A3B HF checkpoint
    PROMPT_DATA     -- path to JSONL prompt dataset
    MAX_TRAIN_ROLLOUT_DIFF -- logprob threshold (default 9.999e-7)
    SGLANG_KV_CACHE_DTYPE  -- fp8_e4m3 (default) or bfloat16
"""

from test_qwen3_30b_a3b_deterministic_e2e import run_gate

NUM_GPUS = 8


def test_qwen3_30b_a3b_layerwise_zero():
    run_gate(layerwise_zero=True, rollout_max_response_len=32)
