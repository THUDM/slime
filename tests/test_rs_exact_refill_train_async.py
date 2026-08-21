import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_TRAIN_ASYNC_PATH = Path(__file__).resolve().parents[1] / "train_async.py"
_TRAIN_ASYNC_SPEC = importlib.util.spec_from_file_location("test_train_async_module", _TRAIN_ASYNC_PATH)
assert _TRAIN_ASYNC_SPEC is not None and _TRAIN_ASYNC_SPEC.loader is not None
train_async = importlib.util.module_from_spec(_TRAIN_ASYNC_SPEC)
_ARGUMENTS_STUB = types.ModuleType("slime.utils.arguments")
_ARGUMENTS_STUB.parse_args = lambda: None
# This CPU scheduling contract does not need the SGLang-backed CLI parser.
with mock.patch.dict(sys.modules, {"slime.utils.arguments": _ARGUMENTS_STUB}):
    _TRAIN_ASYNC_SPEC.loader.exec_module(train_async)

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _RolloutManager:
    def __init__(self, events):
        self.generate = _RemoteMethod(lambda rollout_id: self._record(events, "generate", rollout_id))
        self.save = _RemoteMethod(lambda rollout_id: self._record(events, "manager_save", rollout_id))
        self.eval = _RemoteMethod(lambda rollout_id: self._record(events, "eval", rollout_id))
        self.dispose = _RemoteMethod(lambda: self._record(events, "dispose", None))

    @staticmethod
    def _record(events, name, rollout_id):
        events.append((name, rollout_id))
        return (name, rollout_id)


class _ActorModel:
    def __init__(self, events):
        self.events = events

    def update_weights(self):
        self.events.append(("update_weights", None))

    def async_train(self, rollout_id, _data, external_data=None):
        assert external_data is None
        self.events.append(("train", rollout_id))
        return ("train_ref", rollout_id)

    def save_model(self, rollout_id, *, force_sync):
        self.events.append(("actor_save", rollout_id, force_sync))


def test_refill_checkpoint_finishes_before_the_next_rollout(monkeypatch):
    events = []
    manager = _RolloutManager(events)
    actor = _ActorModel(events)

    def resolve(value):
        if isinstance(value, tuple) and value[0] == "generate":
            return value[1]
        return None

    monkeypatch.setattr(train_async.ray, "get", resolve)
    monkeypatch.setattr(train_async, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(train_async, "create_rollout_manager", lambda _args, _pg: (manager, None))
    monkeypatch.setattr(train_async, "create_training_models", lambda _args, _pgs, _manager: (actor, None))
    monkeypatch.setattr(train_async, "configure_logger", lambda: None)
    monkeypatch.setattr(train_async, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_async, "finish_tracking", lambda _args: None)
    monkeypatch.setattr(
        train_async,
        "run_rs_batch_refill",
        lambda _actor, _manager, rollout_id, **_kwargs: ("train_data", rollout_id),
    )

    args = SimpleNamespace(
        colocate=False,
        release_train=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=2,
        rs_batch_refill=True,
        rs_refill_rpc_timeout_seconds=123.0,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=1,
        rollout_global_dataset=True,
        update_weights_interval=1,
        eval_interval=None,
    )

    train_async.train(args)

    manager_save = events.index(("manager_save", 0))
    next_generate = events.index(("generate", 1))
    updates_before_next = [i for i, event in enumerate(events[:next_generate]) if event[0] == "update_weights"]
    assert manager_save < updates_before_next[-1] < next_generate


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
