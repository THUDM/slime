import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest


class _Ref:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error


class _RemoteMethod:
    def __init__(self, function):
        self._function = function

    def remote(self, *args, **kwargs):
        return self._function(*args, **kwargs)


class _Actor:
    def __init__(self, fake_ray, rank):
        self.fake_ray = fake_ray
        self.rank = rank
        self.alive = True
        self.get_master_addr_and_port = _RemoteMethod(self._get_master_addr_and_port)
        self.init = _RemoteMethod(self._init)
        self.set_rollout_manager = _RemoteMethod(self._set_rollout_manager)

    def _get_master_addr_and_port(self):
        return _Ref(value=("127.0.0.1", 12345), error=self.fake_ray.master_error)

    def _init(self, *_args, **_kwargs):
        return _Ref(value=self.rank, error=self.fake_ray.init_errors.get(self.rank))

    def _set_rollout_manager(self, _rollout_manager):
        return _Ref(value=None, error=self.fake_ray.manager_error)


class _RemoteActorClass:
    def __init__(self, fake_ray):
        self.fake_ray = fake_ray

    def options(self, **_options):
        return self

    def remote(self, _world_size, rank, _master_addr, _master_port):
        actor = _Actor(self.fake_ray, rank)
        self.fake_ray.actors.append(actor)
        return actor


class _FakeRay(ModuleType):
    def __init__(self):
        super().__init__("ray")
        self.reset()

    def reset(self):
        self.actors = []
        self.killed = []
        self.master_error = None
        self.init_errors = {}
        self.manager_error = None
        self.kill_errors = {}

    def remote(self, **_options):
        return lambda _actor_impl: _RemoteActorClass(self)

    def get(self, refs):
        if isinstance(refs, list):
            return [self.get(ref) for ref in refs]
        if refs.error is not None:
            raise refs.error
        return refs.value

    def kill(self, actor, no_restart):
        assert no_restart is True
        error = self.kill_errors.get(actor.rank)
        if error is not None:
            raise error
        actor.alive = False
        self.killed.append(actor)


class _PlacementGroupSchedulingStrategy:
    def __init__(self, **_options):
        pass


def _load_actor_group_module(fake_ray):
    placement_group_module = ModuleType("ray.util.placement_group")
    placement_group_module.PlacementGroup = object
    scheduling_module = ModuleType("ray.util.scheduling_strategies")
    scheduling_module.PlacementGroupSchedulingStrategy = _PlacementGroupSchedulingStrategy
    ray_util_module = ModuleType("ray.util")
    ray_utils_module = ModuleType("slime.ray.utils")
    ray_utils_module.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []
    ray_utils_module.add_default_ray_env_vars = lambda env_vars: env_vars

    stubs = {
        "ray": fake_ray,
        "ray.util": ray_util_module,
        "ray.util.placement_group": placement_group_module,
        "ray.util.scheduling_strategies": scheduling_module,
        "slime.ray.utils": ray_utils_module,
    }
    previous_modules = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module_path = Path(__file__).resolve().parents[1] / "slime" / "ray" / "actor_group.py"
        spec = importlib.util.spec_from_file_location("actor_group_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


@pytest.fixture
def fake_ray():
    return _FakeRay()


@pytest.fixture
def actor_group_module(fake_ray):
    return _load_actor_group_module(fake_ray)


def _group(actor_group_module, world_size=2):
    args = Namespace(
        offload_train=False,
        train_env_vars={},
        update_weight_start_version=0,
        use_routing_replay=False,
    )
    placement_group = (object(), list(range(world_size)), list(range(world_size)))
    return actor_group_module.RayTrainGroup(
        args,
        num_nodes=1,
        num_gpus_per_node=world_size,
        pg=placement_group,
        actor_cls=object,
    )


def test_create_rolls_back_rank_zero_when_master_lookup_fails(actor_group_module, fake_ray):
    expected_error = RuntimeError("master lookup failed")
    fake_ray.master_error = expected_error
    group = _group(actor_group_module)

    with pytest.raises(RuntimeError) as raised:
        group.create()

    assert raised.value is expected_error
    assert group._actor_handlers == []
    assert fake_ray.killed == fake_ray.actors
    assert [actor.alive for actor in fake_ray.actors] == [False]


def test_create_rolls_back_all_actors_when_init_fails(actor_group_module, fake_ray):
    expected_error = RuntimeError("init failed")
    fake_ray.init_errors[1] = expected_error
    group = _group(actor_group_module)

    with pytest.raises(RuntimeError) as raised:
        group.create()

    assert raised.value is expected_error
    assert group._actor_handlers == []
    assert fake_ray.killed == fake_ray.actors
    assert [actor.alive for actor in fake_ray.actors] == [False, False]


def test_create_rolls_back_all_actors_when_setting_manager_fails(actor_group_module, fake_ray):
    expected_error = RuntimeError("set rollout manager failed")
    fake_ray.manager_error = expected_error
    group = _group(actor_group_module)

    with pytest.raises(RuntimeError) as raised:
        group.create(rollout_manager=object())

    assert raised.value is expected_error
    assert group._actor_handlers == []
    assert fake_ray.killed == fake_ray.actors
    assert [actor.alive for actor in fake_ray.actors] == [False, False]


def test_create_preserves_original_error_and_continues_when_one_kill_fails(actor_group_module, fake_ray):
    expected_error = RuntimeError("init failed")
    cleanup_error = RuntimeError("kill failed")
    fake_ray.init_errors[0] = expected_error
    fake_ray.kill_errors[0] = cleanup_error
    group = _group(actor_group_module)

    with pytest.raises(RuntimeError) as raised:
        group.create()

    assert raised.value is expected_error
    assert group._actor_handlers == []
    assert fake_ray.killed == [fake_ray.actors[1]]
    assert [actor.alive for actor in fake_ray.actors] == [True, False]


def test_create_can_retry_after_rollback(actor_group_module, fake_ray):
    fake_ray.init_errors[1] = RuntimeError("init failed")
    group = _group(actor_group_module)

    with pytest.raises(RuntimeError):
        group.create()

    failed_actors = list(fake_ray.actors)
    fake_ray.init_errors.clear()

    assert group.create() == [0, 1]
    retry_actors = fake_ray.actors[2:]
    assert group._actor_handlers == retry_actors
    assert [actor.alive for actor in failed_actors] == [False, False]
    assert [actor.alive for actor in retry_actors] == [True, True]

    assert group.create() is None
    assert fake_ray.actors == failed_actors + retry_actors
