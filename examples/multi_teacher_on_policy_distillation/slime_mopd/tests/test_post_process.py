from types import SimpleNamespace

import pytest
import torch

from slime_mopd.post_process import extract_teacher_log_probs, mopd_post_process


def _reward(values):
    return {"meta_info": {"input_token_logprobs": [[0.0, 0, ""]] + [[v, i, "t"] for i, v in enumerate(values)]}}


def _args(**overrides):
    args = SimpleNamespace(mopd={"teachers": [{"id": "t"}]})
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_extract_returns_last_response_tokens():
    values = [-0.1, -0.2, -0.3, -0.4]
    log_probs = extract_teacher_log_probs(_reward(values), response_length=2, dtype=torch.float32)
    assert torch.allclose(log_probs, torch.tensor([-0.3, -0.4], dtype=torch.float32))


def test_extract_zero_response_length_is_empty():
    log_probs = extract_teacher_log_probs(_reward([-0.1, -0.2]), response_length=0, dtype=torch.float32)
    assert log_probs.numel() == 0


def test_extract_rejects_too_few_tokens():
    with pytest.raises(ValueError, match="token log-probs"):
        extract_teacher_log_probs(_reward([-0.1]), response_length=5, dtype=torch.float32)


def test_extract_rejects_non_dict_reward():
    with pytest.raises(TypeError, match="raw SGLang response"):
        extract_teacher_log_probs(1.0, response_length=1, dtype=torch.float32)


def test_extract_respects_dtype():
    log_probs = extract_teacher_log_probs(_reward([-0.1, -0.2]), response_length=2, dtype=torch.bfloat16)
    assert log_probs.dtype is torch.bfloat16


def test_post_process_sets_teacher_log_probs_and_zero_rewards():
    samples = [
        SimpleNamespace(reward=_reward([-0.1, -0.2]), response_length=2, metadata={"teacher": "math"}),
        SimpleNamespace(reward=_reward([-0.3]), response_length=1, metadata={"teacher": "code"}),
    ]
    raw_rewards, rewards = mopd_post_process(_args(), samples)

    assert rewards == [0.0, 0.0]
    assert raw_rewards == [0.0, 0.0]
    assert torch.allclose(samples[0].teacher_log_probs, torch.tensor([-0.1, -0.2]))
    assert torch.allclose(samples[1].teacher_log_probs, torch.tensor([-0.3]))


def test_post_process_uses_model_params_dtype():
    samples = [SimpleNamespace(reward=_reward([-0.1]), response_length=1, metadata={})]
    mopd_post_process(_args(params_dtype=torch.bfloat16), samples)
    assert samples[0].teacher_log_probs.dtype is torch.bfloat16


def test_post_process_config_dtype_overrides_model_dtype():
    samples = [SimpleNamespace(reward=_reward([-0.1]), response_length=1, metadata={})]
    args = _args(params_dtype=torch.bfloat16)
    args.mopd["logprob_dtype"] = "float16"
    mopd_post_process(args, samples)
    assert samples[0].teacher_log_probs.dtype is torch.float16


def test_humaneval_reward_passes_and_fails():
    import asyncio
    from slime_mopd.rm import mopd_humaneval_reward

    sample_ok = SimpleNamespace(
        response="<think>let me solve this</think>\n```python\ndef add_two(a: int, b: int) -> int:\n    return a + b\n```",
        metadata={
            "entry_point": "add_two",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
            "canonical_prompt": "def add_two(a: int, b: int) -> int:\n",
        },
        label="",
    )
    sample_bad = SimpleNamespace(
        response="```python\ndef add_two(a: int, b: int) -> int:\n    return a - b\n```",
        metadata=sample_ok.metadata,
        label="",
    )
    args = SimpleNamespace(humaneval_timeout_s=3.0)
    assert asyncio.run(mopd_humaneval_reward(args, sample_ok)) == 1.0
    assert asyncio.run(mopd_humaneval_reward(args, sample_bad)) == 0.0
