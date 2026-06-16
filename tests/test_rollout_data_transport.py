import sys
from pathlib import Path

import numpy as np
import pytest
import torch

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from slime.ray.rollout import _tensorize_rollout_data_for_training


@pytest.mark.unit
def test_tensorize_rollout_data_for_training_converts_large_fields():
    read_only_experts = np.frombuffer(
        np.array([[1, 2], [3, 4]], dtype=np.int32).tobytes(),
        dtype=np.int32,
    ).reshape(2, 2)
    assert not read_only_experts.flags.writeable

    rollout_data = {
        "tokens": [[1, 2, 3], torch.tensor([4, 5], dtype=torch.int32)],
        "loss_masks": [[1, 0], np.array([1, 1, 0], dtype=np.int64)],
        "rollout_log_probs": [[-0.1, -0.2]],
        "teacher_log_probs": [torch.tensor([-0.3, -0.4], dtype=torch.float64)],
        "rollout_routed_experts": [read_only_experts],
        "multimodal_train_inputs": [
            {
                "pixel_values": np.array([[1.0, 2.0]], dtype=np.float32),
                "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.int64).t(),
                "metadata": "keep-me-too",
            },
            None,
        ],
        "rollout_mask_sums": [2, 3],
        "total_lengths": [3, 2],
        "prompt": ["keep-me"],
    }

    _tensorize_rollout_data_for_training(rollout_data)

    assert [t.dtype for t in rollout_data["tokens"]] == [torch.long, torch.long]
    assert [t.device.type for t in rollout_data["tokens"]] == ["cpu", "cpu"]
    assert all(t.is_contiguous() for t in rollout_data["tokens"])

    assert [t.dtype for t in rollout_data["loss_masks"]] == [torch.int, torch.int]
    assert rollout_data["rollout_log_probs"][0].dtype == torch.float32
    assert rollout_data["teacher_log_probs"][0].dtype == torch.float32
    assert rollout_data["rollout_routed_experts"][0].dtype == torch.int32
    assert rollout_data["rollout_routed_experts"][0].is_contiguous()
    assert rollout_data["rollout_mask_sums"].dtype == torch.float32
    assert isinstance(rollout_data["multimodal_train_inputs"][0]["pixel_values"], torch.Tensor)
    assert rollout_data["multimodal_train_inputs"][0]["pixel_values"].device.type == "cpu"
    assert rollout_data["multimodal_train_inputs"][0]["image_grid_thw"].is_contiguous()
    assert rollout_data["multimodal_train_inputs"][0]["metadata"] == "keep-me-too"
    assert rollout_data["multimodal_train_inputs"][1] is None

    assert rollout_data["total_lengths"] == [3, 2]
    assert rollout_data["prompt"] == ["keep-me"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
