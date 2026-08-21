from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slime.observability.rollout_data_utils import (
    load_debug_rollout_data,
    save_debug_rollout_data,
    tensorize_rollout_data_for_training,
    validate_rollout_routed_experts_for_replay,
)
from slime.utils.types import Sample

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
    validate_rollout_routed_experts_for_replay([routes], _args())


def test_r3_validation_rejects_missing_pipeline_layers():
    routes = torch.zeros((4, 6, 2), dtype=torch.uint8)
    routes[:, 3, 1] = 7

    with pytest.raises(ValueError, match=r"all zero.*\[4, 5\]"):
        validate_rollout_routed_experts_for_replay([routes], _args())


def test_r3_validation_rejects_wrong_shape():
    routes = torch.zeros((4, 5, 2), dtype=torch.uint8)

    with pytest.raises(ValueError, match="Invalid rollout routed-experts shape"):
        validate_rollout_routed_experts_for_replay([routes], _args())


def test_tensorize_rollout_data_for_training_normalizes_cpu_tensors():
    readonly_tokens = np.array([1, 2, 3])
    readonly_tokens.flags.writeable = False
    rollout_data = {
        "tokens": [readonly_tokens],
        "loss_masks": [[1, 0]],
        "rs_preflight_log_probs": [torch.tensor([-0.2, -0.3], dtype=torch.float64, requires_grad=True)],
        "multimodal_train_inputs": [
            {
                "pixel_values": torch.tensor([1.0], requires_grad=True),
                "metadata": "unchanged",
            }
        ],
        "rollout_mask_sums": [2],
    }

    tensorize_rollout_data_for_training(rollout_data)

    assert rollout_data["tokens"][0].dtype == torch.long
    assert rollout_data["loss_masks"][0].dtype == torch.int
    assert rollout_data["rs_preflight_log_probs"][0].dtype == torch.float32
    assert rollout_data["rs_preflight_log_probs"][0].device.type == "cpu"
    assert rollout_data["rs_preflight_log_probs"][0].is_contiguous()
    assert not rollout_data["rs_preflight_log_probs"][0].requires_grad
    assert rollout_data["multimodal_train_inputs"][0]["metadata"] == "unchanged"
    assert not rollout_data["multimodal_train_inputs"][0]["pixel_values"].requires_grad
    assert rollout_data["rollout_mask_sums"].dtype == torch.float32


def test_save_and_load_debug_rollout_data_round_trip(tmp_path):
    path_template = str(tmp_path / "rollout_{rollout_id}.pt")
    samples = [
        Sample(index=1, rollout_id=3, prompt="question", response="answer", response_length=1),
    ]

    save_debug_rollout_data(path_template, samples, rollout_id=3, evaluation=False)
    loaded = load_debug_rollout_data(path_template, rollout_id=3)

    assert len(loaded) == 1
    assert loaded[0].index == 1
    assert loaded[0].rollout_id == 3
    assert loaded[0].prompt == "question"
    assert loaded[0].response == "answer"


def test_save_debug_eval_rollout_data_flattens_datasets(tmp_path):
    path_template = str(tmp_path / "rollout_{rollout_id}.pt")
    data = {
        "math": {"samples": [Sample(index=1)]},
        "code": {"samples": [Sample(index=2)]},
    }

    save_debug_rollout_data(path_template, data, rollout_id=4, evaluation=True)

    saved = torch.load(tmp_path / "rollout_eval_4.pt", weights_only=False)
    assert saved["rollout_id"] == 4
    assert [sample["index"] for sample in saved["samples"]] == [1, 2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
