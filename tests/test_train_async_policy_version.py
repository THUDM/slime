"""CPU tests for async-training policy-version publication."""

from __future__ import annotations

import sys
import types

import pytest

# Keep this lifecycle test independent of GPU/server packages imported by the
# real training bootstrap. Restore the real module entries after importing the
# helper so these stubs cannot leak into other test modules.
_stubs = {
    "slime.ray.placement_group": {
        "create_placement_groups": None,
        "create_rollout_manager": None,
        "create_training_models": None,
    },
    "slime.utils.arguments": {"parse_args": None},
    "slime.utils.logging_utils": {
        "configure_logger": None,
        "finish_tracking": None,
        "init_tracking": None,
    },
}
_original_modules = {name: sys.modules.get(name) for name in _stubs}
try:
    for name, attributes in _stubs.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        sys.modules[name] = module

    import train_async
finally:
    for name, original_module in _original_modules.items():
        if original_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original_module

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeRolloutManager:
    def __init__(self):
        self.policy_version = 0
        self.events = []
        self.before_weight_update = _RemoteMethod(self._before_weight_update)
        self.after_weight_update = _RemoteMethod(self._after_weight_update)

    def _before_weight_update(self):
        self.events.append(("before", self.policy_version))
        return self.policy_version

    def _after_weight_update(self, *, succeeded):
        if succeeded:
            self.policy_version += 1
        self.events.append(("after", self.policy_version, succeeded))
        return self.policy_version


class _FakeActorModel:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.update_calls = 0

    def update_weights(self):
        self.update_calls += 1
        if self.fail:
            raise RuntimeError("weight update failed")


@pytest.mark.unit
def test_policy_version_advances_after_successful_weight_update(monkeypatch):
    monkeypatch.setattr(train_async.ray, "get", lambda value: value)
    actor_model = _FakeActorModel()
    rollout_manager = _FakeRolloutManager()

    train_async._update_actor_weights(actor_model, rollout_manager)

    assert actor_model.update_calls == 1
    assert rollout_manager.policy_version == 1
    assert rollout_manager.events == [("before", 0), ("after", 1, True)]


@pytest.mark.unit
def test_failed_weight_update_keeps_policy_version(monkeypatch):
    monkeypatch.setattr(train_async.ray, "get", lambda value: value)
    actor_model = _FakeActorModel(fail=True)
    rollout_manager = _FakeRolloutManager()

    with pytest.raises(RuntimeError, match="weight update failed"):
        train_async._update_actor_weights(actor_model, rollout_manager)

    assert actor_model.update_calls == 1
    assert rollout_manager.policy_version == 0
    assert rollout_manager.events == [("before", 0), ("after", 0, False)]
