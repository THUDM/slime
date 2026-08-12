import pytest
import torch

from slime.utils.compare_glm52_layerwise import (
    TrainSequence,
    _sglang_layer_token_rows,
    compare_layer_outputs,
    load_train_sequences,
    map_requests_to_train_sequences,
)

NUM_GPUS = 0


def test_train_sequences_use_adjacent_cumulative_offsets(tmp_path):
    dump_dir = tmp_path / "megatron"
    dump_file = dump_dir / "rank00000" / "actor_Pass00000.pt"
    dump_file.parent.mkdir(parents=True)
    torch.save(
        {
            "input_ids": torch.tensor([7, 8, 9]),
            "cu_seqlens": torch.tensor([0, 2, 3]),
            "layers": {0: torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.bfloat16)},
        },
        dump_file,
    )

    sequences = load_train_sequences(dump_dir, {0})

    assert [sequence.tokens.tolist() for sequence in sequences] == [[7, 8], [9]]
    assert [sequence.layers[0].flatten().tolist() for sequence in sequences] == [[1.0, 2.0], [3.0]]


def test_health_check_requests_are_excluded_from_sequence_mapping(tmp_path):
    dump_file = tmp_path / "Chunk00000.pt"
    torch.save(
        {
            "model.forward_batch_info.input_ids": torch.tensor([99, 7, 8]),
            "model.forward_batch_info.positions": torch.tensor([0, 0, 1]),
            "model.forward_batch_info.rids": ["HEALTH_CHECK_probe", "rollout-0"],
            "model.forward_batch_info.extend_seq_lens": torch.tensor([1, 2]),
        },
        dump_file,
    )
    train_sequences = [TrainSequence(tokens=torch.tensor([7, 8]), layers={}, source="test")]

    assert map_requests_to_train_sequences([dump_file], train_sequences) == {"rollout-0": 0}


def test_sglang_layer_output_reconstructs_the_visible_residual_sum():
    delta = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    residual = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)

    output = _sglang_layer_token_rows([delta, residual], 1, "layer0")

    torch.testing.assert_close(output, torch.tensor([[4.0, 6.0]], dtype=torch.bfloat16))


def test_layerwise_comparison_excludes_terminal_causal_state(tmp_path):
    dump_file = tmp_path / "Chunk00000.pt"
    torch.save(
        {
            "model.forward_batch_info.input_ids": torch.tensor([7, 8, 9]),
            "model.forward_batch_info.positions": torch.tensor([0, 1, 2]),
            "model.forward_batch_info.rids": ["rollout-0"],
            "model.forward_batch_info.extend_seq_lens": torch.tensor([3]),
            "model.layers.0": [
                torch.tensor([[1.0], [2.0], [100.0]], dtype=torch.bfloat16),
                torch.zeros(3, 1, dtype=torch.bfloat16),
            ],
        },
        dump_file,
    )
    train_sequences = [
        TrainSequence(
            tokens=torch.tensor([7, 8, 9]),
            layers={0: torch.tensor([[1.0], [2.0], [-100.0]], dtype=torch.bfloat16)},
            source="test",
        )
    ]

    stats = compare_layer_outputs([dump_file], train_sequences, {"rollout-0": 0}, {0})

    assert stats[0]["tokens"] == 2
    assert stats[0]["max_abs"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
