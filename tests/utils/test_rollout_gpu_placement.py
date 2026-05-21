from argparse import Namespace

import pytest

from slime.ray.rollout import ServerGroup, start_rollout_servers


def _args(**overrides):
    defaults = {
        "debug_train_only": False,
        "num_gpus_per_node": 4,
        "rollout_num_gpus": 1,
        "rollout_num_gpus_per_engine": 1,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_server_group_rejects_incomplete_placement_metadata():
    group = ServerGroup(
        args=_args(),
        pg=(None, [0, 1], [0]),
        all_engines=[None],
        num_gpus_per_engine=1,
        num_new_engines=0,
    )

    with pytest.raises(ValueError, match="Invalid rollout placement group metadata"):
        group.validate_gpu_placement(num_gpu_per_engine=1, reordered_bundle_indices=[0, 1], reordered_gpu_ids=[0])


def test_server_group_rejects_non_positive_local_gpu_count():
    group = ServerGroup(
        args=_args(),
        pg=(None, [0], [0]),
        all_engines=[None],
        num_gpus_per_engine=1,
        num_new_engines=0,
    )

    with pytest.raises(ValueError, match="num_gpu_per_engine must be positive"):
        group.validate_gpu_placement(num_gpu_per_engine=0, reordered_bundle_indices=[0], reordered_gpu_ids=[0])


def test_server_group_rejects_gpu_bundle_index_out_of_range():
    group = ServerGroup(
        args=_args(rollout_num_gpus=3, rollout_num_gpus_per_engine=2),
        pg=(None, [0, 1, 2], [0, 1, 2]),
        all_engines=[None, None],
        num_gpus_per_engine=2,
        num_new_engines=0,
    )

    with pytest.raises(ValueError) as exc_info:
        group.validate_gpu_placement(
            num_gpu_per_engine=2,
            reordered_bundle_indices=[0, 1, 2],
            reordered_gpu_ids=[0, 1, 2],
        )

    message = str(exc_info.value)
    assert "placement GPU bundle index 3" in message
    assert "only 3 rollout GPU slots are available" in message
    assert "rollout_num_gpus=3" in message
    assert "rollout_num_gpus_per_engine=2" in message
    assert "num_gpus_per_node=4" in message


def test_start_rollout_servers_rejects_non_divisible_server_group():
    args = _args(
        rollout_num_gpus=3,
        rollout_num_gpus_per_engine=2,
        sglang_config=None,
        prefill_num_servers=None,
        debug_rollout_only=True,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        offload_rollout=False,
        hf_checkpoint="/tmp/model",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )

    with pytest.raises(ValueError) as exc_info:
        start_rollout_servers(args, pg=(None, [0, 1, 2], [0, 1, 2]))

    message = str(exc_info.value)
    assert "Invalid sglang server group GPU configuration" in message
    assert "num_gpus=3" in message
    assert "local_num_gpus_per_engine=2" in message
    assert "rollout_num_gpus=3" in message
