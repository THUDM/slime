from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slime.backends.megatron_utils.alignment.layerwise_alignment import enable_megatron_layerwise_dump

NUM_GPUS = 0


class _Layer(nn.Module):
    def __init__(self, layer_number: int):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, value):
        return value + self.layer_number


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(1), _Layer(2)])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()

    def forward(self, *, input_ids, packed_seq_params):
        del packed_seq_params
        return self.decoder(input_ids.float())


def test_megatron_layerwise_dump(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR", str(tmp_path))
    args = SimpleNamespace(megatron_deepgemm_forward_layers=[0, 1])
    model = _Model()
    enable_megatron_layerwise_dump(args, [model], store_prefix="actor_")

    model(
        input_ids=torch.tensor([[7, 8]]),
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2])),
    )

    (dump_file,) = list(tmp_path.glob("rank*/actor_Pass*.pt"))
    values = torch.load(dump_file, weights_only=False)
    torch.testing.assert_close(values["input_ids"], torch.tensor([[7, 8]]))
    torch.testing.assert_close(values["layers"][0], torch.tensor([[8.0, 9.0]]))
    torch.testing.assert_close(values["layers"][1], torch.tensor([[10.0, 11.0]]))


def test_megatron_layerwise_dump_is_enabled_on_nonzero_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("SLIME_LAYERWISE_ALIGNMENT_MODULE_SUFFIXES", "decoder.layers.0")
    monkeypatch.setattr("slime.backends.megatron_utils.alignment.layerwise_alignment._global_rank", lambda: 3)
    args = SimpleNamespace(megatron_deepgemm_forward_layers=[0, 1])
    model = _Model()

    enable_megatron_layerwise_dump(args, [model], store_prefix="actor_")
    model(
        input_ids=torch.tensor([[7, 8]]),
        packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2])),
    )

    (dump_file,) = list(tmp_path.glob("rank00003/actor_Pass*.pt"))
    values = torch.load(dump_file, weights_only=False)
    assert "decoder.layers.0" in values["modules"]


def _comparison_fixture(tmp_path, *, positions=(0, 1)):
    from glm52_layerwise_comparator import TrainSequence

    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sequence = TrainSequence(tokens=torch.tensor([7, 8, 9]), layers={0: rows}, source="train")
    record = {
        "model.forward_batch_info.input_ids": sequence.tokens[list(positions)],
        "model.forward_batch_info.positions": torch.tensor(positions),
        "model.forward_batch_info.rids": ["request"],
        "model.forward_batch_info.extend_seq_lens": torch.tensor([len(positions)]),
        "model.layers.0": (rows[list(positions)].clone(), torch.zeros(len(positions), 2)),
    }
    path = tmp_path / "Pass00000.pt"
    torch.save(record, path)
    return sequence, record, path


def test_layerwise_comparison_covers_all_score_producing_positions(tmp_path):
    from glm52_layerwise_comparator import compare_layer_outputs

    sequence, _, path = _comparison_fixture(tmp_path)
    stats = compare_layer_outputs([path], [sequence], {"request": 0}, {0})
    assert stats[0]["tokens"] == 2
    assert stats[0]["max_abs"] == 0


@pytest.mark.parametrize("source", ["train", "rollout"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_layerwise_comparison_rejects_nonfinite_values(tmp_path, source, invalid):
    from glm52_layerwise_comparator import compare_layer_outputs

    sequence, record, path = _comparison_fixture(tmp_path)
    if source == "train":
        sequence.layers[0][0, 0] = invalid
    else:
        record["model.layers.0"][0][0, 0] = invalid
        torch.save(record, path)
    with pytest.raises(ValueError, match="Non-finite"):
        compare_layer_outputs([path], [sequence], {"request": 0}, {0})


def test_layerwise_comparison_rejects_missing_response_positions(tmp_path):
    from glm52_layerwise_comparator import compare_layer_outputs

    sequence, _, path = _comparison_fixture(tmp_path, positions=(0,))
    with pytest.raises(ValueError, match="Missing hidden states"):
        compare_layer_outputs([path], [sequence], {"request": 0}, {0})


def test_layerwise_request_mapping_rejects_missing_sequences(tmp_path):
    from glm52_layerwise_comparator import TrainSequence, map_requests_to_train_sequences

    sequence, _, path = _comparison_fixture(tmp_path)
    other = TrainSequence(tokens=torch.tensor([10, 11, 12]), layers=sequence.layers, source="other")
    with pytest.raises(RuntimeError, match="missing 1 Megatron token sequences"):
        map_requests_to_train_sequences([path], [sequence, other])


def _alignment_rollout(tmp_path):
    sample = {"response_length": 1, "rollout_log_probs": [-0.5], "loss_mask": [1], "weight_versions": ["1"]}
    path = tmp_path / "rollout.pt"
    torch.save({"samples": [dict(sample) for _ in range(8)]}, path)
    return path


def test_alignment_gate_accepts_finite_metric_and_synchronized_rollout(tmp_path):
    from test_glm52_6layer_deterministic_e2e import _assert_alignment_result

    path = _alignment_rollout(tmp_path)
    _assert_alignment_result("{'train/train_rollout_logprob_abs_diff': 2e-7}", str(path), 1e-6)


@pytest.mark.parametrize("metric", [None, "nan", "inf", "1e-5"])
def test_alignment_gate_rejects_missing_or_invalid_metric(tmp_path, metric):
    from test_glm52_6layer_deterministic_e2e import _assert_alignment_result

    path = _alignment_rollout(tmp_path)
    output = "{}" if metric is None else "{'train/train_rollout_logprob_abs_diff': " + metric + "}"
    with pytest.raises(AssertionError):
        _assert_alignment_result(output, str(path), 1e-6)


@pytest.mark.parametrize(
    "key,value",
    [("response_length", 0), ("rollout_log_probs", [float("nan")]), ("loss_mask", [0]), ("weight_versions", ["0"])],
)
def test_alignment_gate_rejects_empty_masked_or_stale_rollout(tmp_path, key, value):
    from test_glm52_6layer_deterministic_e2e import _assert_alignment_result

    path = _alignment_rollout(tmp_path)
    data = torch.load(path, weights_only=False)
    data["samples"][0][key] = value
    torch.save(data, path)
    with pytest.raises(AssertionError):
        _assert_alignment_result("{'train/train_rollout_logprob_abs_diff': 0}", str(path), 1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
