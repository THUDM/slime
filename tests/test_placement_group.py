import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.ray import placement_group
from slime.ray.placement_group import _create_placement_group, _get_placement_group_layout

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "actor_num_nodes": 2,
        "actor_num_gpus_per_node": 8,
        "rollout_num_gpus": 32,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "colocate": False,
        "rollout_external": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, (48, 16), id="normal_non_colocate"),
        pytest.param({"debug_train_only": True}, (16, 0), id="debug_train_only"),
        pytest.param({"debug_rollout_only": True}, (32, 0), id="debug_rollout_only"),
        pytest.param({"colocate": True, "rollout_num_gpus": 8}, (16, 0), id="colocate_rollout_less_than_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 16}, (16, 0), id="colocate_rollout_equals_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 32}, (32, 0), id="colocate_rollout_more_than_actor"),
        pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
        pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
        pytest.param({"rollout_external": True}, (16, 16), id="external"),
        pytest.param({"rollout_external": True, "debug_rollout_only": True}, (16, 0), id="external_debug_rollout"),
    ],
)
def test_placement_group_layout(overrides, expected):
    assert _get_placement_group_layout(_args(**overrides)) == expected


def test_create_zero_gpu_placement_group_is_empty():
    assert _create_placement_group(0) == (None, [], [])


@pytest.mark.parametrize("retain_serving", [True, False])
def test_internal_serving_lifetime_can_be_disabled(monkeypatch, retain_serving):
    serving_options = []
    manager_options = []
    deployment = SimpleNamespace(
        placements={"rollout": ("rollout_pg", [], [])},
        restore_plan=None,
        routers={},
        reused=False,
    )
    serving = SimpleNamespace(attach_training=SimpleNamespace(remote=Mock(return_value="serving-attach")))
    resume = SimpleNamespace(restore_plan=None, reused=False, apply=lambda args: args)
    manager = SimpleNamespace(
        attach_training=SimpleNamespace(remote=Mock(return_value="manager-attach")),
    )

    class ActorFactory:
        def __init__(self, actor, calls):
            self.actor = actor
            self.calls = calls

        def options(self, **options):
            self.calls.append(options)
            return SimpleNamespace(remote=Mock(return_value=self.actor))

    from slime.ray import rollout
    from slime.ray import serving as serving_module
    from slime.ray.training_recovery import RECOVERY_NAMESPACE, training_session_name

    monkeypatch.setattr(serving_module, "ServingCluster", ActorFactory(serving, serving_options))
    monkeypatch.setattr(rollout, "RolloutManager", ActorFactory(manager, manager_options))
    monkeypatch.setattr(
        placement_group.ray,
        "get",
        lambda ref, timeout=None: deployment if ref == "serving-attach" else resume,
    )
    monkeypatch.setattr(
        placement_group.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: "job"),
    )

    args = SimpleNamespace(
        retain_serving=retain_serving,
        rollout_external=False,
        rollout_data_transport="object-store",
        rollout_session_id="demo",
        num_rollout=1,
        check_weight_update_equal=False,
        offload_rollout=False,
    )
    placement_group._attach_rollout_manager(placement_group.RolloutStartup(args))

    lifecycle_keys = {"name", "namespace", "lifetime", "get_if_exists"}
    for options, suffix in ((serving_options[0], ":serving"), (manager_options[0], "")):
        if retain_serving:
            assert options["name"] == training_session_name(args) + suffix
            assert options["namespace"] == RECOVERY_NAMESPACE
            assert options["lifetime"] == "detached"
            assert options["get_if_exists"] is True
        else:
            assert lifecycle_keys.isdisjoint(options)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
