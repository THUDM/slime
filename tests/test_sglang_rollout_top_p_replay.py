from argparse import Namespace

import numpy as np
import pybase64
import pytest

from slime.ray.rollout import compute_metrics_from_samples
from slime.rollout import sglang_rollout
from slime.rollout.sglang_rollout import GenerateState, _append_rollout_top_p_token_data
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.fixture
def generate_state_deps(monkeypatch):
    GenerateState.clear_instances()
    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(sglang_rollout, "get_rollout_num_engines", lambda args: 1)
    yield
    GenerateState.clear_instances()


def _generate_state_args(top_p: float) -> Namespace:
    return Namespace(
        hf_checkpoint="dummy",
        sglang_server_concurrency=1,
        rollout_temperature=1.0,
        rollout_top_p=top_p,
        rollout_top_k=-1,
        rollout_max_response_len=16,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        sglang_enable_deterministic_inference=False,
        sglang_dp_size=None,
    )


def _b64_int32(values: list[int]) -> str:
    return pybase64.b64encode(np.asarray(values, dtype=np.int32).tobytes()).decode("ascii")


def _metric_args() -> Namespace:
    return Namespace(
        advantage_estimator="ppo",
        log_reward_category=None,
        sglang_speculative_algorithm=None,
    )


@pytest.mark.unit
def test_generate_state_requests_top_p_token_ids_when_rollout_top_p_enabled(generate_state_deps):
    state = GenerateState(_generate_state_args(top_p=0.95))

    assert state.sampling_params["custom_params"] == {"return_top_p_token_ids": True}


@pytest.mark.unit
def test_generate_state_skips_top_p_token_ids_when_rollout_top_p_is_one(generate_state_deps):
    state = GenerateState(_generate_state_args(top_p=1.0))

    assert "custom_params" not in state.sampling_params


@pytest.mark.unit
def test_append_rollout_top_p_token_data_decodes_base64_and_shifts_offsets():
    sample = Sample(
        response_length=1,
        rollout_top_p_token_ids=[3, 4],
        rollout_top_p_token_offsets=[0, 2],
    )
    meta_info = {
        "top_p_token_ids": _b64_int32([10, 11, 12, 20]),
        "top_p_token_offsets": _b64_int32([0, 3, 4]),
    }

    _append_rollout_top_p_token_data(sample, meta_info, expected_num_tokens=2)

    assert sample.rollout_top_p_token_ids == [3, 4, 10, 11, 12, 20]
    assert sample.rollout_top_p_token_offsets == [0, 2, 5, 6]


@pytest.mark.unit
def test_append_rollout_top_p_token_data_accepts_debug_lists():
    sample = Sample()
    meta_info = {
        "top_p_kept_token_ids": [101, 102, 201],
        "top_p_kept_token_offsets": [0, 2, 3],
    }

    _append_rollout_top_p_token_data(sample, meta_info, expected_num_tokens=2)

    assert sample.rollout_top_p_token_ids == [101, 102, 201]
    assert sample.rollout_top_p_token_offsets == [0, 2, 3]


@pytest.mark.unit
def test_append_rollout_top_p_token_data_rejects_mismatched_offsets():
    sample = Sample()
    meta_info = {
        "top_p_token_ids": _b64_int32([10, 11]),
        "top_p_token_offsets": _b64_int32([0, 3]),
    }

    with pytest.raises(ValueError, match="ids/offsets mismatch"):
        _append_rollout_top_p_token_data(sample, meta_info, expected_num_tokens=1)


@pytest.mark.unit
def test_append_rollout_top_p_token_data_ignores_absent_payload():
    sample = Sample()

    _append_rollout_top_p_token_data(sample, {"finish_reason": {"type": "stop"}}, expected_num_tokens=0)

    assert sample.rollout_top_p_token_ids is None
    assert sample.rollout_top_p_token_offsets is None


@pytest.mark.unit
def test_top_p_kept_vocab_metric_reports_only_when_payload_exists():
    metrics = compute_metrics_from_samples(_metric_args(), [Sample(response_length=2)])
    assert "top_p_kept_vocab_per_token" not in metrics

    metrics = compute_metrics_from_samples(
        _metric_args(),
        [
            Sample(response_length=2, rollout_top_p_token_offsets=[0, 3, 4]),
            Sample(response_length=1, rollout_top_p_token_offsets=[0, 5]),
        ],
    )

    assert metrics["top_p_kept_vocab_per_token"] == 3.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
