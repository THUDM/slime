from types import SimpleNamespace

import pytest

from slime.rollout.on_policy_distillation import _build_teacher_payload, _validate_teacher_temperature


def test_teacher_payload_requests_temperature_scaled_input_logprobs():
    args = SimpleNamespace(rollout_temperature=0.8)
    sample = SimpleNamespace(tokens=[1, 2, 3], multimodal_inputs=None)

    payload = _build_teacher_payload(args, sample)

    assert payload["sampling_params"] == {
        "temperature": 1.0,
        "max_new_tokens": 0,
        "skip_special_tokens": False,
    }
    assert payload["input_logprob_temperature"] == 0.8


def test_teacher_temperature_requires_patched_sglang_for_non_unit_temperature():
    args = SimpleNamespace(rollout_temperature=0.8)

    with pytest.raises(RuntimeError, match="temperature-scaled input log-probs"):
        _validate_teacher_temperature(args, {"meta_info": {}})


def test_unit_temperature_does_not_require_sglang_extension():
    args = SimpleNamespace(rollout_temperature=1.0)
    sample = SimpleNamespace(tokens=[1, 2, 3], multimodal_inputs=None)

    payload = _build_teacher_payload(args, sample)

    assert "input_logprob_temperature" not in payload


def test_teacher_temperature_accepts_matching_acknowledgement():
    args = SimpleNamespace(rollout_temperature=0.8)

    _validate_teacher_temperature(
        args,
        {"meta_info": {"input_logprob_temperature": 0.8}},
    )


def test_unit_teacher_temperature_remains_compatible_with_vanilla_sglang():
    args = SimpleNamespace(rollout_temperature=1.0)

    _validate_teacher_temperature(args, {"meta_info": {}})
