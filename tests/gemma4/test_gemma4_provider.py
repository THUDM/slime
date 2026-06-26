import json
from types import SimpleNamespace

import pytest
import torch

from tests.gemma4._standalone_imports import load_gemma4_provider_module

_provider = load_gemma4_provider_module()


def test_install_hooks_softcap_wraps_tensor_output():
    inner = torch.nn.Module()
    inner.output_layer = torch.nn.Linear(4, 8, bias=False)

    hf_text = SimpleNamespace(final_logit_softcapping=30.0)
    orig = _provider._load_hf_text_config
    _provider._load_hf_text_config = lambda _path: hf_text
    try:
        args = SimpleNamespace(hf_checkpoint="/nonexistent")
        config = SimpleNamespace(hidden_size=4)
        _provider._install_hooks(
            model=inner,
            args=args,
            config=config,
            pre_process=False,
            post_process=True,
        )
    finally:
        _provider._load_hf_text_config = orig

    x = torch.randn(2, 4)
    raw = x @ inner.output_layer.weight.T
    hooked = inner.output_layer(x)
    expected = torch.tanh(raw / 30.0) * 30.0
    assert torch.allclose(hooked, expected, atol=1e-6)
    assert hooked.abs().max().item() <= 30.0


def test_install_hooks_softcap_reuses_storage_with_correct_gradient():
    class _CaptureOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw = None
            self.raw_before = None

        def forward(self, x):
            self.raw = x * 1.0
            self.raw_before = self.raw.detach().clone()
            return self.raw

    inner = torch.nn.Module()
    inner.output_layer = _CaptureOutput()

    hf_text = SimpleNamespace(final_logit_softcapping=30.0)
    orig = _provider._load_hf_text_config
    _provider._load_hf_text_config = lambda _path: hf_text
    try:
        args = SimpleNamespace(hf_checkpoint="/nonexistent")
        config = SimpleNamespace(hidden_size=4)
        _provider._install_hooks(
            model=inner,
            args=args,
            config=config,
            pre_process=False,
            post_process=True,
        )
    finally:
        _provider._load_hf_text_config = orig

    base = torch.linspace(-3.0, 3.0, steps=12, dtype=torch.float64).view(3, 4)
    base.requires_grad_(True)
    weights = torch.linspace(0.1, 1.2, steps=12, dtype=torch.float64).view(3, 4)

    hooked = inner.output_layer(base)
    (hooked * weights).sum().backward()

    expected = 30.0 * torch.tanh(inner.output_layer.raw_before / 30.0)
    expected_grad = weights * (1.0 - torch.tanh(inner.output_layer.raw_before / 30.0).pow(2))
    assert hooked.data_ptr() == inner.output_layer.raw.data_ptr()
    assert torch.allclose(hooked, expected)
    assert torch.allclose(base.grad, expected_grad)


def test_install_hooks_softcap_wraps_tuple_output():
    inner = torch.nn.Module()

    class _TupleOutLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8, 4))

        def forward(self, x):
            return x @ self.w.T, None  # (output, bias)

    inner.output_layer = _TupleOutLayer()
    hf_text = SimpleNamespace(final_logit_softcapping=30.0)
    orig = _provider._load_hf_text_config
    _provider._load_hf_text_config = lambda _path: hf_text
    try:
        args = SimpleNamespace(hf_checkpoint="/nonexistent")
        config = SimpleNamespace(hidden_size=4)
        _provider._install_hooks(
            model=inner,
            args=args,
            config=config,
            pre_process=False,
            post_process=True,
        )
    finally:
        _provider._load_hf_text_config = orig

    x = torch.randn(3, 4)
    hooked, bias = inner.output_layer(x)
    raw = x @ inner.output_layer.w.T
    expected = torch.tanh(raw / 30.0) * 30.0
    assert torch.allclose(hooked, expected, atol=1e-6)
    assert bias is None  # tuple tail preserved


def test_install_hooks_no_softcap_when_disabled():
    inner = torch.nn.Module()
    inner.output_layer = torch.nn.Linear(4, 8, bias=False)

    for cap_value in (None, 0, 0.0):
        for h in list(inner.output_layer._forward_hooks.keys()):
            inner.output_layer._forward_hooks.pop(h)

        hf_text = SimpleNamespace(final_logit_softcapping=cap_value)
        orig = _provider._load_hf_text_config
        _provider._load_hf_text_config = lambda _p, _t=hf_text: _t
        try:
            args = SimpleNamespace(hf_checkpoint="/nonexistent")
            config = SimpleNamespace(hidden_size=4)
            _provider._install_hooks(
                model=inner,
                args=args,
                config=config,
                pre_process=False,
                post_process=True,
            )
        finally:
            _provider._load_hf_text_config = orig
        assert len(inner.output_layer._forward_hooks) == 0, f"softcap hook should not register when cap={cap_value!r}"


def _install_embed_hook(inner, hidden):
    hf_text = SimpleNamespace(final_logit_softcapping=None)
    orig = _provider._load_hf_text_config
    _provider._load_hf_text_config = lambda _path: hf_text
    try:
        args = SimpleNamespace(hf_checkpoint="/nonexistent")
        config = SimpleNamespace(hidden_size=hidden)
        _provider._install_hooks(
            model=inner,
            args=args,
            config=config,
            pre_process=True,
            post_process=False,
        )
    finally:
        _provider._load_hf_text_config = orig


def test_install_hooks_embedding_scale_fp32_weight():
    hidden = 1024
    inner = torch.nn.Module()
    inner.embedding = torch.nn.Embedding(100, hidden)  # fp32 by default
    _install_embed_hook(inner, hidden)

    ids = torch.tensor([[1, 2, 3]])
    hooked = inner.embedding(ids)
    raw = inner.embedding.weight[ids]
    expected_scale = torch.tensor(hidden**0.5)
    assert torch.allclose(hooked, raw * expected_scale, atol=1e-6)


def test_install_hooks_embedding_scale_bf16_weight():
    hidden = 1024
    inner = torch.nn.Module()
    inner.embedding = torch.nn.Embedding(100, hidden).to(torch.bfloat16)
    _install_embed_hook(inner, hidden)

    ids = torch.tensor([[1, 2, 3]])
    hooked = inner.embedding(ids)
    raw = inner.embedding.weight[ids]
    expected_scale = torch.tensor(hidden**0.5).to(torch.bfloat16)
    assert torch.allclose(hooked, raw * expected_scale, atol=1e-2)


def _write_fake_safetensors_layer_scalars(ckpt_dir, scalars):
    from safetensors.torch import save_file

    weight_map = {}
    for layer_idx, value in scalars.items():
        tensor_name = f"model.language_model.layers.{layer_idx}.layer_scalar"
        fname = f"layer_{layer_idx}.safetensors"
        save_file({tensor_name: torch.tensor(value)}, str(ckpt_dir / fname))
        weight_map[tensor_name] = fname
    index = {"metadata": {}, "weight_map": weight_map}
    (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps(index))


def test_load_layer_scalars_applies_values_to_layers(tmp_path):
    scalars = {0: 0.5, 1: 1.5, 2: 2.5}
    _write_fake_safetensors_layer_scalars(tmp_path, scalars)

    inner = torch.nn.Module()
    inner.decoder = torch.nn.Module()
    layers = []
    for _ in range(3):
        layer = torch.nn.Module()
        layer.register_buffer("layer_scalar", torch.ones(1))
        layers.append(layer)
    inner.decoder.layers = torch.nn.ModuleList(layers)

    import megatron.core.transformer.transformer_layer as tl

    orig_offset = tl.get_transformer_layer_offset
    tl.get_transformer_layer_offset = lambda _cfg: 0
    try:
        _provider._load_layer_scalars(inner, str(tmp_path), config=SimpleNamespace())
    finally:
        tl.get_transformer_layer_offset = orig_offset

    for i, expected in scalars.items():
        assert inner.decoder.layers[i].layer_scalar.item() == pytest.approx(expected)


def test_load_layer_scalars_respects_pp_offset(tmp_path):
    scalars = {10: 0.7, 11: 0.8, 12: 0.9}
    _write_fake_safetensors_layer_scalars(tmp_path, scalars)

    inner = torch.nn.Module()
    inner.decoder = torch.nn.Module()
    layers = []
    for _ in range(3):
        layer = torch.nn.Module()
        layer.register_buffer("layer_scalar", torch.ones(1))
        layers.append(layer)
    inner.decoder.layers = torch.nn.ModuleList(layers)

    import megatron.core.transformer.transformer_layer as tl

    orig_offset = tl.get_transformer_layer_offset
    tl.get_transformer_layer_offset = lambda _cfg: 10  # PP offset
    try:
        _provider._load_layer_scalars(inner, str(tmp_path), config=SimpleNamespace())
    finally:
        tl.get_transformer_layer_offset = orig_offset

    assert inner.decoder.layers[0].layer_scalar.item() == pytest.approx(0.7)
    assert inner.decoder.layers[1].layer_scalar.item() == pytest.approx(0.8)
    assert inner.decoder.layers[2].layer_scalar.item() == pytest.approx(0.9)


def test_load_layer_scalars_raises_by_default_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMMA4_ALLOW_MISSING_LAYER_SCALARS", raising=False)
    scalars = {0: 0.5}
    _write_fake_safetensors_layer_scalars(tmp_path, scalars)

    inner = torch.nn.Module()
    inner.decoder = torch.nn.Module()
    layers = []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.register_buffer("layer_scalar", torch.ones(1))
        layers.append(layer)
    inner.decoder.layers = torch.nn.ModuleList(layers)

    import megatron.core.transformer.transformer_layer as tl

    orig_offset = tl.get_transformer_layer_offset
    tl.get_transformer_layer_offset = lambda _cfg: 0
    try:
        with pytest.raises(KeyError, match="missing in checkpoint"):
            _provider._load_layer_scalars(inner, str(tmp_path), config=SimpleNamespace())
    finally:
        tl.get_transformer_layer_offset = orig_offset


def test_load_layer_scalars_defaults_to_one_when_missing_with_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMMA4_ALLOW_MISSING_LAYER_SCALARS", "1")
    scalars = {0: 0.5}
    _write_fake_safetensors_layer_scalars(tmp_path, scalars)

    inner = torch.nn.Module()
    inner.decoder = torch.nn.Module()
    layers = []
    for _ in range(2):
        layer = torch.nn.Module()
        layer.register_buffer("layer_scalar", torch.ones(1))
        layers.append(layer)
    inner.decoder.layers = torch.nn.ModuleList(layers)

    import megatron.core.transformer.transformer_layer as tl

    orig_offset = tl.get_transformer_layer_offset
    tl.get_transformer_layer_offset = lambda _cfg: 0
    try:
        _provider._load_layer_scalars(inner, str(tmp_path), config=SimpleNamespace())
    finally:
        tl.get_transformer_layer_offset = orig_offset

    assert inner.decoder.layers[0].layer_scalar.item() == pytest.approx(0.5)
    assert inner.decoder.layers[1].layer_scalar.item() == pytest.approx(1.0)


def test_load_layer_scalars_raises_when_no_index_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMMA4_ALLOW_MISSING_LAYER_SCALARS", raising=False)
    inner = torch.nn.Module()
    inner.decoder = torch.nn.Module()
    inner.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    inner.decoder.layers[0].register_buffer("layer_scalar", torch.ones(1))

    with pytest.raises(RuntimeError, match="No layer_scalar weights found"):
        _provider._load_layer_scalars(inner, str(tmp_path), config=SimpleNamespace())


def test_load_layer_scalars_skips_when_no_index_file_with_opt_in(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setenv("GEMMA4_ALLOW_MISSING_LAYER_SCALARS", "1")
    inner = torch.nn.Module()
    inner.decoder = torch.nn.Module()
    inner.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    inner.decoder.layers[0].register_buffer("layer_scalar", torch.ones(1))

    with caplog.at_level(logging.WARNING, logger=_provider.__name__):
        _provider._load_layer_scalars(inner, str(tmp_path), config=SimpleNamespace())
    assert inner.decoder.layers[0].layer_scalar.item() == 1.0
    assert any("No safetensors index" in r.message for r in caplog.records)
