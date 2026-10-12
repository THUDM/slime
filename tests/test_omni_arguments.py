"""Deployment validation runs before Ray or GPU processes are created."""

import argparse
import sys

import pytest

from slime.utils.arguments import get_slime_extra_args_provider, validate_omni_args

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def _args(tmp_path, **overrides):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("config_cls: MossTTSPrismPipelineConfig\nenable_rl: true\n")
    values = dict(
        rollout_backend="sglang-omni",
        omni_config=str(pipeline),
        omni_python=None,
        colocate=False,
        offload_rollout=False,
        rollout_external=False,
        update_weight_mode="full",
        update_weight_transport="nccl",
        check_weight_update_equal=False,
        debug_train_only=False,
        omni_request_timeout_s=600.0,
        omni_cancel_drain_timeout_s=600.0,
    )
    return argparse.Namespace(**(values | overrides))


def test_cli_defaults_keep_original_backend():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1"])
    assert args.rollout_backend == "sglang"
    assert args.custom_megatron_log_prob_function_path is None
    assert not args.use_rollout_logprobs
    assert args.policy_stop_loss_weight == 1.0


def test_disaggregated_omni_defaults(tmp_path):
    args = _args(tmp_path)
    validate_omni_args(args)
    assert args.omni_python == sys.executable
    assert args.omni_config == str(tmp_path / "pipeline.yaml")


@pytest.mark.parametrize("flag", ["colocate", "offload_rollout", "rollout_external", "check_weight_update_equal"])
def test_unsupported_capabilities_fail_before_launch(tmp_path, flag):
    with pytest.raises(ValueError):
        validate_omni_args(_args(tmp_path, **{flag: True}))


@pytest.mark.parametrize("overrides", [{"update_weight_mode": "delta"}, {"update_weight_transport": "ipc"}])
def test_requires_full_nccl_publication(tmp_path, overrides):
    with pytest.raises(ValueError, match="full NCCL"):
        validate_omni_args(_args(tmp_path, **overrides))


@pytest.mark.parametrize("name", ["omni_request_timeout_s", "omni_cancel_drain_timeout_s"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_timeouts_fail_before_launch(tmp_path, name, value):
    with pytest.raises(ValueError, match="positive and finite"):
        validate_omni_args(_args(tmp_path, **{name: value}))


def test_missing_pipeline_and_train_only(tmp_path):
    with pytest.raises(ValueError, match="existing --omni-config"):
        validate_omni_args(_args(tmp_path, omni_config=None))
    validate_omni_args(_args(tmp_path, omni_config=None, debug_train_only=True))


def test_original_backend_bypasses_omni_validation(tmp_path):
    args = _args(tmp_path, rollout_backend="sglang", colocate=True, omni_config=None)
    validate_omni_args(args)
    assert args.omni_python is None
