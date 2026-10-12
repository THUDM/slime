"""Resume seeds shared by trainer updaters and the Ray disk coordinator."""

import copy
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest

from slime.backends.megatron_utils.update_weight import create_weight_updater
from slime.utils.weight_sync import initial_weight_version

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("start", "existing", "loaded_start", "expected"),
    [
        pytest.param(0, 0, 1, 0, id="native-fresh"),
        pytest.param(None, 0, 1, 1, id="resume-iteration-zero"),
        pytest.param(None, 0, 12, 12, id="resume-later-iteration"),
        pytest.param(5, 0, 12, 5, id="explicit-start"),
        pytest.param(None, 19, 12, 19, id="higher-live-version"),
        pytest.param(0, 19, 1, 19, id="explicit-fresh-preserves-live-version"),
        pytest.param(None, 7, None, 7, id="debug-no-loaded-iteration"),
    ],
)
def test_initial_version_preserves_explicit_and_restored_boundaries(start, existing, loaded_start, expected):
    args = Namespace(start_rollout_id=start, update_weight_start_version=existing)
    assert initial_weight_version(args, loaded_start) == expected


class _Updater:
    def __init__(self, args, model, weights_getter, *, model_name, quantization_config):
        self.weight_version = 0

    def update_weights(self):
        self.weight_version += 1


def _stub_updaters(monkeypatch):
    for suffix, class_name in [
        ("update_weight_from_distributed", "UpdateWeightFromDistributed"),
        ("update_weight_from_disk", "UpdateWeightFromDisk"),
    ]:
        name = f"slime.backends.megatron_utils.update_weight.{suffix}"
        module = ModuleType(name)
        setattr(module, class_name, _Updater)
        monkeypatch.setitem(sys.modules, name, module)


def _updater(args):
    return create_weight_updater(args, [], lambda: {}, model_name="moss_tts_prism", quantization_config=None)


@pytest.mark.parametrize("transport", ["nccl", "disk"])
def test_real_factory_seeds_the_first_resumed_publication(monkeypatch, transport):
    _stub_updaters(monkeypatch)
    args = Namespace(
        start_rollout_id=None,
        update_weight_start_version=0,
        update_weight_mode="full",
        update_weight_transport=transport,
        colocate=False,
    )
    args.update_weight_start_version = initial_weight_version(args, 1)
    updater = _updater(args)
    assert updater.weight_version == 1
    updater.update_weights()
    assert updater.weight_version == 2


def _group(monkeypatch, *, starts, start=None, existing=0, manager_seed=0, transport="nccl"):
    from slime.ray import actor_group

    _stub_updaters(monkeypatch)
    actors, events = [], []

    class Actor:
        def __init__(self, world_size, rank, master_addr, master_port):
            self.rank = rank
            self.get_master_addr_and_port = SimpleNamespace(remote=lambda: ("127.0.0.1", 15000))
            self.init = SimpleNamespace(remote=self.initialize)
            self.set_rollout_manager = SimpleNamespace(remote=lambda manager: events.append(("attach", rank)))
            self.update_weights = SimpleNamespace(remote=self.publish)

        def initialize(self, args, role, **kwargs):
            events.append(("init", self.rank, args.update_weight_start_version))
            # Match Ray's argument serialization: trainer mutations must not
            # accidentally update the driver's copied configuration.
            self.args = copy.deepcopy(args)
            loaded_start = starts[self.rank]
            self.updater = None
            if loaded_start is not None:
                self.args.update_weight_start_version = initial_weight_version(self.args, loaded_start)
                self.updater = _updater(self.args)
            return loaded_start

        def publish(self):
            self.updater.update_weights()

    class ActorFactory:
        def options(self, **kwargs):
            return self

        def remote(self, *args):
            actor = Actor(*args)
            actors.append(actor)
            return actor

    def register(role, handlers, configuration):
        events.append(("register", manager_seed))
        assert handlers == actors
        return {"update_weight_start_version": manager_seed}

    manager = SimpleNamespace(register_training_actors=SimpleNamespace(remote=register))
    monkeypatch.setattr(
        actor_group,
        "ray",
        SimpleNamespace(remote=lambda **kwargs: lambda cls: ActorFactory(), get=lambda value: value),
    )
    monkeypatch.setattr(actor_group, "PlacementGroupSchedulingStrategy", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(actor_group, "training_recovery_enabled", lambda args: False)
    args = Namespace(
        start_rollout_id=start,
        update_weight_start_version=existing,
        update_weight_mode="full",
        update_weight_transport=transport,
        colocate=False,
        train_env_vars={},
        offload_train=False,
        use_routing_replay=False,
        release_train=False,
    )
    group = actor_group.RayTrainGroup(
        args,
        num_nodes=1,
        num_gpus_per_node=len(starts),
        pg=(object(), list(range(len(starts))), list(range(len(starts)))),
        actor_cls=Actor,
    )
    return group, manager, actors, events


@pytest.mark.parametrize(
    ("start", "existing", "manager_seed", "loaded_start", "expected"),
    [
        pytest.param(0, 0, 0, 1, 0, id="fresh-zero-is-not-loaded-one"),
        pytest.param(None, 0, 0, 1, 1, id="manager-placeholder-zero"),
        pytest.param(None, 9, 0, 1, 9, id="higher-driver-seed"),
        pytest.param(None, 0, 15, 4, 15, id="higher-manager-seed"),
        pytest.param(None, 7, 0, None, 7, id="debug-no-loaded-iteration"),
    ],
)
def test_real_group_create_synchronizes_seed_after_manager_registration(
    monkeypatch, start, existing, manager_seed, loaded_start, expected
):
    group, manager, actors, events = _group(
        monkeypatch, starts=[loaded_start, loaded_start], start=start, existing=existing, manager_seed=manager_seed
    )
    assert group.create(rollout_manager=manager) == [loaded_start, loaded_start]
    assert group.args.update_weight_start_version == expected
    assert group._disk_weight_version == expected
    assert events[0] == ("register", manager_seed)
    assert [event for event in events if event[0] == "attach"] == [("attach", 0), ("attach", 1)]
    if loaded_start is not None:
        assert [actor.updater.weight_version for actor in actors] == [expected, expected]
    else:
        assert all(actor.updater is None for actor in actors)


def test_group_rejects_different_loaded_iterations_before_manager_attachment(monkeypatch):
    group, manager, actors, events = _group(monkeypatch, starts=[1, 2])
    with pytest.raises(AssertionError, match="different checkpoint iterations"):
        group.create(rollout_manager=manager)
    assert group._disk_weight_version == group.args.update_weight_start_version == 0
    assert not any(event[0] == "attach" for event in events)


def test_resumed_disk_group_and_updaters_publish_the_same_version_directory(monkeypatch, tmp_path):
    group, manager, actors, events = _group(monkeypatch, starts=[1, 1], transport="disk")
    group.args.update_weight_disk_dir = str(tmp_path)
    reloads = []
    monkeypatch.setattr(
        group, "_reload_rollout_weights_from_disk", lambda path, version: reloads.append((path, version))
    )
    group.create(rollout_manager=manager)
    group.update_weights()
    assert [actor.updater.weight_version for actor in actors] == [2, 2]
    assert group._disk_weight_version == 2
    assert reloads == [(tmp_path / "weight_v000002", "2")]
