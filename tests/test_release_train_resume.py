import importlib
import sys
import types
from argparse import Namespace

import pytest

from slime.ray.placement_group import _resolve_start_rollout_id


class _RemoteMethod:
    def __init__(self, function):
        self.remote = function


class _RolloutManager:
    def __init__(self):
        self.generate = _RemoteMethod(lambda rollout_id: f"rollout-{rollout_id}")
        self.save = _RemoteMethod(lambda rollout_id: None)
        self.cleanup_rollout_data = _RemoteMethod(lambda rollout_id: None)
        self.dispose = _RemoteMethod(lambda: None)


class _ActorModel:
    def __init__(self, recreated_start_rollout_ids):
        self.recreated_start_rollout_ids = recreated_start_rollout_ids
        self.train_calls = []

    def create(self):
        return self.recreated_start_rollout_ids

    def update_weights(self):
        pass

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.train_calls.append(rollout_id)

    def save_model(self, rollout_id, force_sync=False):
        pass

    def clear_memory(self):
        pass


@pytest.fixture
def train_module(monkeypatch):
    ray = types.ModuleType("ray")
    ray.get = lambda value: value

    logging_utils = types.ModuleType("slime.observability.logging_utils")
    logging_utils.configure_logger = lambda: None
    logging_utils.finish_tracking = lambda args: None
    logging_utils.init_tracking = lambda args: None

    placement_group = types.ModuleType("slime.ray.placement_group")
    placement_group._resolve_start_rollout_id = _resolve_start_rollout_id
    placement_group.create_placement_groups = lambda args: {"rollout": None}
    placement_group.create_rollout_manager = None
    placement_group.create_training_models = None

    arguments = types.ModuleType("slime.utils.arguments")
    arguments.parse_args = lambda: None

    misc = types.ModuleType("slime.utils.misc")
    misc.should_run_periodic_action = lambda *args: False

    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "slime.observability.logging_utils", logging_utils)
    monkeypatch.setitem(sys.modules, "slime.ray.placement_group", placement_group)
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", arguments)
    monkeypatch.setitem(sys.modules, "slime.utils.misc", misc)
    monkeypatch.delitem(sys.modules, "train", raising=False)
    module = importlib.import_module("train")
    yield module
    sys.modules.pop("train", None)


def _args(**overrides):
    values = {
        "release_train": True,
        "offload_rollout": False,
        "check_weight_update_equal": False,
        "num_rollout": 2,
        "eval_interval": None,
        "start_rollout_id": 1,
        "skip_eval_before_train": False,
        "offload_train": False,
        "use_critic": False,
        "num_critic_only_steps": 0,
        "save_interval": None,
        "finetune": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_release_train_rejects_recreated_actor_mismatch_before_training(train_module):
    actor_model = _ActorModel([2, 2])
    rollout_manager = _RolloutManager()
    train_module.create_rollout_manager = lambda args, pg: (rollout_manager, None)
    train_module.create_training_models = lambda args, pgs, manager: (actor_model, None)

    with pytest.raises(ValueError, match=r"requested=1.*actor=2"):
        train_module.train(_args())

    assert actor_model.train_calls == []


@pytest.mark.parametrize(
    ("args", "recreated_start_rollout_ids", "expected_train_calls"),
    [
        (_args(start_rollout_id=0, num_rollout=1, finetune=True), [1, 1], [0]),
        (_args(), [1, 1], [1]),
    ],
)
def test_release_train_accepts_matching_recreated_actor(
    train_module,
    args,
    recreated_start_rollout_ids,
    expected_train_calls,
):
    actor_model = _ActorModel(recreated_start_rollout_ids)
    rollout_manager = _RolloutManager()
    train_module.create_rollout_manager = lambda args, pg: (rollout_manager, None)
    train_module.create_training_models = lambda args, pgs, manager: (actor_model, None)

    train_module.train(args)

    assert actor_model.train_calls == expected_train_calls
