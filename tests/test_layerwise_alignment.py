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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
