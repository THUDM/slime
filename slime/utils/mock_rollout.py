"""Mock rollout data utilities for tests and benchmarks."""

import pickle

import numpy as np


def make_mock_rollout_data(
    batch_size: int = 16,
    seq_len: int = 2048,
    n_samples_per_prompt: int = 4,
    use_routing_replay: bool = False,
    num_layers: int = 64,
    moe_router_topk: int = 2,
) -> dict:
    """Create mock rollout data resembling real training data structure."""
    num_samples = batch_size * n_samples_per_prompt
    total_lengths = [seq_len + np.random.randint(100, 500) for _ in range(num_samples)]
    # response_length must be < total_length for rollout_log_probs shape
    response_lengths = [
        min(np.random.randint(100, 500), tot - 1) for tot in total_lengths
    ]

    tokens = [np.random.randint(0, 32000, size=l, dtype=np.int32) for l in total_lengths]
    loss_masks = [[1] * (resp_len - 50) + [0] * 50 for resp_len in response_lengths]
    rollout_log_probs = [
        np.random.randn(tot_len - resp_len).astype(np.float32).tolist()
        for tot_len, resp_len in zip(total_lengths, response_lengths)
    ]

    data = {
        "partition": list(range(num_samples)),
        "tokens": tokens,
        "response_lengths": response_lengths,
        "rewards": [1.0] * num_samples,
        "loss_masks": loss_masks,
        "rollout_log_probs": rollout_log_probs,
        "total_lengths": total_lengths,
    }
    if use_routing_replay:
        data["rollout_routed_experts"] = [
            np.random.randint(0, 8, size=(tot_len - 1, num_layers, moe_router_topk), dtype=np.int32)
            for tot_len in total_lengths
        ]
    return data


def get_serialized_size(data: dict) -> int:
    """Get size in bytes of serialized data."""
    return len(pickle.dumps(data, protocol=5))
