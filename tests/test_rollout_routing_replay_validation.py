from types import SimpleNamespace

import pytest
import torch

from slime.ray.rollout import _validate_rollout_routed_experts_for_replay

NUM_GPUS = 0


def _args():
    return SimpleNamespace(
        num_layers=6,
        moe_router_topk=2,
        moe_layer_freq=[0, 0, 0, 1, 1, 1],
    )


def test_r3_validation_accepts_dense_zeros_and_complete_moe_routes():
    routes = torch.zeros((4, 6, 2), dtype=torch.uint8)
    routes[:, 3:, 1] = 7
    _validate_rollout_routed_experts_for_replay([routes], _args())


def test_r3_validation_rejects_missing_pipeline_layers():
    routes = torch.zeros((4, 6, 2), dtype=torch.uint8)
    routes[:, 3, 1] = 7

    with pytest.raises(ValueError, match=r"all zero.*\[4, 5\]"):
        _validate_rollout_routed_experts_for_replay([routes], _args())


def test_r3_validation_rejects_wrong_shape():
    routes = torch.zeros((4, 5, 2), dtype=torch.uint8)

    with pytest.raises(ValueError, match="Invalid rollout routed-experts shape"):
        _validate_rollout_routed_experts_for_replay([routes], _args())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
