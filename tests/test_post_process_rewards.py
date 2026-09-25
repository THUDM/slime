"""CPU unit tests for ``RolloutManager._post_process_rewards``.

Aborted samples keep ``reward=None``. Converting that list to a tensor in
GRPO used to crash; defaulting the missing reward to 0.0 was rejected
because aborted samples must not reach training. These tests bind the
real method onto a dummy object so we can exercise the fail-closed check
without starting Ray.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample


NUM_GPUS = 0


def _install_import_stubs() -> None:
    """Stub Ray and rollout-only imports so this file stays CPU-only.

    ``slime.ray.rollout`` pulls SGLang deployment, health monitors, and Ray
    actors at import time. CI images have those; a laptop running
    ``pytest tests/test_post_process_rewards.py`` usually does not.
    """

    def ensure(name: str, **attrs):
        if name in sys.modules:
            mod = sys.modules[name]
        else:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
            if "." in name:
                parent_name, attr = name.rsplit(".", 1)
                parent = sys.modules.get(parent_name) or ensure(parent_name)
                setattr(parent, attr, mod)
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    if "ray" not in sys.modules:
        ray = ensure("ray")

        def remote(*args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]

            def deco(obj):
                return obj

            return deco

        ray.remote = remote

    import slime.backends  # noqa: F401
    import slime.observability  # noqa: F401
    import slime.ray  # noqa: F401
    import slime.rollout  # noqa: F401

    # Real ``slime.backends.sglang_utils`` init loads accelerator / SGLang.
    ensure("slime.backends.sglang_utils")
    ensure("slime.backends.sglang_utils.deployment", start_rollout_servers=lambda *a, **k: ({}, []))
    ensure(
        "slime.observability.logging_utils",
        configure_logger=lambda *a, **k: None,
        init_tracking=lambda *a, **k: None,
        finish_tracking=lambda *a, **k: None,
    )
    ensure(
        "slime.observability.rollout_data_utils",
        load_debug_rollout_data=None,
        save_debug_rollout_data=None,
        tensorize_rollout_data_for_training=None,
        validate_rollout_id_annotated=None,
        validate_rollout_routed_experts_for_replay=None,
    )
    ensure("slime.observability.rollout_metrics", log_eval_rollout_data=None, log_rollout_data=None)
    ensure("slime.rollout.base_types", call_rollout_fn=None)
    ensure("slime.rollout.sample_hooks", set_current_rollout_id=None)
    ensure("slime.utils.data", get_source=None)
    ensure("slime.utils.dp_schedule", build_dp_schedule=None)
    ensure("slime.utils.health_monitor", RolloutHealthMonitor=object)
    ensure("slime.ray.utils", Lock=object, add_default_ray_env_vars=lambda *a, **k: {})


def _import_rollout_manager():
    try:
        from slime.ray.rollout import RolloutManager
    except Exception:
        sys.modules.pop("slime.ray.rollout", None)
        _install_import_stubs()
        from slime.ray.rollout import RolloutManager
    return RolloutManager


RolloutManager = _import_rollout_manager()


class Dummy:
    custom_reward_post_process_func = None
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        n_samples_per_prompt=2,
        rollout_batch_size=1,
        grpo_std_normalization=True,
        reward_key=None,
    )


Dummy._post_process_rewards = RolloutManager._post_process_rewards

_ABORT_MSG = "must not reach training"


def _completed(reward: float) -> Sample:
    return Sample(status=Sample.Status.COMPLETED, reward=reward)


@pytest.mark.unit
def test_post_process_rewards_completed_numeric():
    dummy = Dummy()
    samples = [_completed(1.0), _completed(0.0)]

    raw_rewards, rewards = dummy._post_process_rewards(samples)

    assert raw_rewards == [1.0, 0.0]
    assert isinstance(rewards, list)
    tensor = torch.tensor(rewards, dtype=torch.float)
    assert tensor.shape == (2,)
    assert all(isinstance(value, float) for value in rewards)


@pytest.mark.unit
def test_post_process_rewards_rejects_aborted_sample():
    dummy = Dummy()
    samples = [_completed(1.0), Sample(status=Sample.Status.ABORTED, reward=None)]

    with pytest.raises(ValueError, match=_ABORT_MSG):
        dummy._post_process_rewards(samples)


@pytest.mark.unit
def test_post_process_rewards_rejects_none_reward():
    dummy = Dummy()
    samples = [_completed(1.0), Sample(status=Sample.Status.COMPLETED, reward=None)]

    with pytest.raises(ValueError, match=_ABORT_MSG):
        dummy._post_process_rewards(samples)


@pytest.mark.unit
def test_custom_reward_post_process_owns_aborted_samples():
    seen = {}

    def custom(args, samples):
        seen["n"] = len(samples)
        seen["status"] = samples[0].status
        return [0.0], [0.0]

    dummy = Dummy()
    dummy.custom_reward_post_process_func = custom
    samples = [Sample(status=Sample.Status.ABORTED, reward=None)]

    raw_rewards, rewards = dummy._post_process_rewards(samples)

    assert seen == {"n": 1, "status": Sample.Status.ABORTED}
    assert raw_rewards == [0.0]
    assert rewards == [0.0]
