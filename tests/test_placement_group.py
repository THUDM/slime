import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.ray.placement_group import _create_placement_group, _get_placement_group_layout, _resolve_start_rollout_id

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


def test_start_rollout_id_rejects_inconsistent_actor_ranks():
    with pytest.raises(ValueError, match=r"actor.*\[7, 8\]"):
        _resolve_start_rollout_id([7, 8])


def test_start_rollout_id_rejects_inconsistent_critic_ranks():
    with pytest.raises(ValueError, match=r"critic.*\[7, 8\]"):
        _resolve_start_rollout_id([7, 7], critic_start_rollout_ids=[7, 8])


def test_start_rollout_id_rejects_actor_critic_mismatch():
    with pytest.raises(ValueError, match=r"actor=7.*critic=8"):
        _resolve_start_rollout_id([7, 7], critic_start_rollout_ids=[8, 8])


def test_start_rollout_id_rejects_requested_id_mismatch():
    with pytest.raises(ValueError, match=r"requested=9.*actor=7.*critic=7"):
        _resolve_start_rollout_id(
            [7, 7],
            critic_start_rollout_ids=[7, 7],
            requested_start_rollout_id=9,
        )


@pytest.mark.parametrize("critic_start_rollout_ids", [None, [1, 1]])
def test_start_rollout_id_maps_fresh_model_sentinel_to_rollout_zero(critic_start_rollout_ids):
    assert (
        _resolve_start_rollout_id(
            [1, 1],
            critic_start_rollout_ids=critic_start_rollout_ids,
            requested_start_rollout_id=0,
            allow_fresh_start=True,
        )
        == 0
    )


def test_start_rollout_id_does_not_treat_real_resume_as_fresh():
    with pytest.raises(ValueError, match=r"requested=0.*actor=2"):
        _resolve_start_rollout_id([2, 2], requested_start_rollout_id=0)


def test_start_rollout_id_requires_fresh_provenance_for_zero_one_sentinel():
    with pytest.raises(ValueError, match=r"requested=0.*actor=1"):
        _resolve_start_rollout_id([1, 1], requested_start_rollout_id=0)


@pytest.mark.parametrize(
    ("critic_start_rollout_ids", "requested_start_rollout_id"),
    [(None, None), ([7, 7], 7)],
)
def test_start_rollout_id_accepts_consistent_loaded_ids(critic_start_rollout_ids, requested_start_rollout_id):
    assert (
        _resolve_start_rollout_id(
            [7, 7],
            critic_start_rollout_ids=critic_start_rollout_ids,
            requested_start_rollout_id=requested_start_rollout_id,
        )
        == 7
    )


def test_create_training_models_validates_requested_start_rollout_id(monkeypatch):
    from slime.ray import placement_group as module

    args = Namespace(use_critic=False, num_rollout=1, start_rollout_id=9)
    monkeypatch.setattr(module, "create_actor_model", lambda *args, **kwargs: (object(), [7, 7]))
    monkeypatch.setattr(module.ray, "get", lambda value: value, raising=False)
    rollout_manager = Namespace(load=Namespace(remote=lambda rollout_id: rollout_id))

    with pytest.raises(ValueError, match=r"requested=9.*actor=7"):
        module.create_training_models(args, {}, rollout_manager)


def test_create_training_models_preserves_fresh_rollout_zero(monkeypatch):
    from slime.ray import placement_group as module

    args = Namespace(
        use_critic=False,
        num_rollout=1,
        start_rollout_id=0,
        finetune=True,
        debug_rollout_only=False,
    )
    monkeypatch.setattr(module, "create_actor_model", lambda *args, **kwargs: (object(), [1, 1]))
    monkeypatch.setattr(module.ray, "get", lambda value: value, raising=False)
    loaded_rollout_ids = []
    rollout_manager = Namespace(load=Namespace(remote=loaded_rollout_ids.append))

    module.create_training_models(args, {}, rollout_manager)

    assert args.start_rollout_id == 0
    assert loaded_rollout_ids == [-1]


@pytest.mark.parametrize(
    ("requested_start_rollout_id", "expected_start_rollout_id", "expected_loaded_rollout_ids"),
    [(7, 7, [6]), (None, 0, [-1])],
)
def test_debug_rollout_only_uses_generation_cursor(
    monkeypatch,
    requested_start_rollout_id,
    expected_start_rollout_id,
    expected_loaded_rollout_ids,
):
    from slime.ray import placement_group as module

    args = Namespace(
        use_critic=False,
        num_rollout=10,
        start_rollout_id=requested_start_rollout_id,
        finetune=False,
        debug_rollout_only=True,
    )
    monkeypatch.setattr(module, "create_actor_model", lambda *args, **kwargs: (object(), [0, 0]))
    monkeypatch.setattr(module.ray, "get", lambda value: value, raising=False)
    loaded_rollout_ids = []
    rollout_manager = Namespace(load=Namespace(remote=loaded_rollout_ids.append))

    module.create_training_models(args, {}, rollout_manager)

    assert args.start_rollout_id == expected_start_rollout_id
    assert loaded_rollout_ids == expected_loaded_rollout_ids


def test_debug_rollout_only_still_rejects_inconsistent_actor_ranks(monkeypatch):
    from slime.ray import placement_group as module

    args = Namespace(use_critic=False, num_rollout=10, start_rollout_id=7, debug_rollout_only=True)
    monkeypatch.setattr(module, "create_actor_model", lambda *args, **kwargs: (object(), [0, 1]))

    with pytest.raises(ValueError, match=r"actor.*\[0, 1\]"):
        module.create_training_models(args, {}, object())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
