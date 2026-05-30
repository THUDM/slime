"""Unit tests for ``gemma4_provider.py`` hooks and helpers.

Covers:
- ``_install_hooks``: embedding-scale, softcap, dual-RoPE wiring (integration
  bits are in ``test_gemma4_dual_rope.py``).
- ``_load_layer_scalars``: reading per-layer scalar buffers from an HF
  safetensors checkpoint, with the PP offset translation.

These are pure-Python/CPU tests. We import the provider module by hand to
avoid Megatron-wide imports that require CUDA / a process group."""

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch


def _load_provider_module():
    """Import ``gemma4_provider`` without triggering the module-level
    ``from megatron.training import get_args``."""
    import types

    if "megatron.training" not in sys.modules:
        stub = types.ModuleType("megatron.training")
        stub.get_args = lambda: None
        sys.modules["megatron.training"] = stub
    if "megatron.training.arguments" not in sys.modules:
        stub2 = types.ModuleType("megatron.training.arguments")
        stub2.core_transformer_config_from_args = lambda *a, **k: None
        sys.modules["megatron.training.arguments"] = stub2

    repo_path = pathlib.Path(__file__).resolve().parents[2] / (
        "slime_plugins/models/gemma4_provider.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_gemma4_provider_under_test", repo_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_provider = _load_provider_module()


# ============================================================================
# Softcap hook (part of _install_hooks)
# ============================================================================

def test_install_hooks_softcap_wraps_tensor_output():
    """With final_logit_softcapping=30, the output_layer forward hook must
    transform tensor output `x` → tanh(x / 30) * 30."""
    # Build a minimal `inner` and `args` that _install_hooks recognises.
    inner = torch.nn.Module()
    inner.output_layer = torch.nn.Linear(4, 8, bias=False)

    hf_text = SimpleNamespace(final_logit_softcapping=30.0)
    # Monkey-patch the helper in the provider module so we don't need a real
    # HF checkpoint on disk.
    orig = _provider._load_hf_text_config
    _provider._load_hf_text_config = lambda _path: hf_text
    try:
        args = SimpleNamespace(hf_checkpoint="/nonexistent")
        config = SimpleNamespace(hidden_size=4)
        _provider._install_hooks(
            model=inner, args=args, config=config,
            pre_process=False, post_process=True,
        )
    finally:
        _provider._load_hf_text_config = orig

    # Run output_layer forward — the hook should softcap the result. We
    # compute `raw` manually (the forward hook wraps __call__, so calling
    # output_layer(x) returns the softcapped version, not the raw).
    x = torch.randn(2, 4)
    raw = x @ inner.output_layer.weight.T  # same math as Linear but no hook
    hooked = inner.output_layer(x)  # goes through the hook
    expected = torch.tanh(raw / 30.0) * 30.0
    assert torch.allclose(hooked, expected, atol=1e-6), (
        "softcap hook did not apply tanh(x/cap)*cap to the tensor output"
    )
    # And the hooked output is in the softcap range.
    assert hooked.abs().max().item() <= 30.0


def test_install_hooks_softcap_wraps_tuple_output():
    """Some Megatron layers return (output, bias) tuples. The softcap hook
    must only transform the first element and leave the rest untouched."""
    inner = torch.nn.Module()
    # Minimal stand-in for Megatron ColumnParallelLinear-style output.
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
            model=inner, args=args, config=config,
            pre_process=False, post_process=True,
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
    """When final_logit_softcapping is None / 0, no hook is registered."""
    inner = torch.nn.Module()
    inner.output_layer = torch.nn.Linear(4, 8, bias=False)

    for cap_value in (None, 0, 0.0):
        # Clear any previous hook
        for h in list(inner.output_layer._forward_hooks.keys()):
            inner.output_layer._forward_hooks.pop(h)

        hf_text = SimpleNamespace(final_logit_softcapping=cap_value)
        orig = _provider._load_hf_text_config
        _provider._load_hf_text_config = lambda _p, _t=hf_text: _t
        try:
            args = SimpleNamespace(hf_checkpoint="/nonexistent")
            config = SimpleNamespace(hidden_size=4)
            _provider._install_hooks(
                model=inner, args=args, config=config,
                pre_process=False, post_process=True,
            )
        finally:
            _provider._load_hf_text_config = orig
        assert len(inner.output_layer._forward_hooks) == 0, (
            f"softcap hook should not register when cap={cap_value!r}"
        )


# ============================================================================
# Embedding scale hook (part of _install_hooks)
# ============================================================================

def _install_embed_hook(inner, hidden):
    hf_text = SimpleNamespace(final_logit_softcapping=None)
    orig = _provider._load_hf_text_config
    _provider._load_hf_text_config = lambda _path: hf_text
    try:
        args = SimpleNamespace(hf_checkpoint="/nonexistent")
        config = SimpleNamespace(hidden_size=hidden)
        _provider._install_hooks(
            model=inner, args=args, config=config,
            pre_process=True, post_process=False,
        )
    finally:
        _provider._load_hf_text_config = orig


def test_install_hooks_embedding_scale_fp32_weight():
    """With fp32 embedding weights, the scale is applied in fp32 — matches
    ``Gemma4TextScaledWordEmbedding.forward = emb * embed_scale.to(weight.dtype)``."""
    hidden = 1024
    inner = torch.nn.Module()
    inner.embedding = torch.nn.Embedding(100, hidden)  # fp32 by default
    _install_embed_hook(inner, hidden)

    ids = torch.tensor([[1, 2, 3]])
    hooked = inner.embedding(ids)
    raw = inner.embedding.weight[ids]
    expected_scale = torch.tensor(hidden ** 0.5)  # fp32 full precision
    assert torch.allclose(hooked, raw * expected_scale, atol=1e-6), (
        "embed scale must be applied in fp32 when weight is fp32"
    )


def test_install_hooks_embedding_scale_bf16_weight():
    """With bf16 embedding weights, the scale is cast to bf16 before
    multiplying — matching HF's ``embed_scale.to(weight.dtype)`` semantics.
    This guards against a previous impl that pre-cast the scale to bf16
    regardless of weight dtype."""
    hidden = 1024
    inner = torch.nn.Module()
    inner.embedding = torch.nn.Embedding(100, hidden).to(torch.bfloat16)
    _install_embed_hook(inner, hidden)

    ids = torch.tensor([[1, 2, 3]])
    hooked = inner.embedding(ids)
    raw = inner.embedding.weight[ids]
    # Expected: scale cast to bf16 at forward time.
    expected_scale = torch.tensor(hidden ** 0.5).to(torch.bfloat16)
    assert torch.allclose(hooked, raw * expected_scale, atol=1e-2), (
        "embed scale must be cast to bf16 when weight is bf16"
    )


# ============================================================================
# _load_layer_scalars
# ============================================================================

def _write_fake_safetensors_layer_scalars(ckpt_dir, scalars):
    """Write a minimal safetensors checkpoint containing only layer_scalar
    tensors, plus an index.json so _load_layer_scalars can find them."""
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
    """Confirm scalars from safetensors are copied into layer.layer_scalar."""
    scalars = {0: 0.5, 1: 1.5, 2: 2.5}
    _write_fake_safetensors_layer_scalars(tmp_path, scalars)

    # Build a minimal "inner" with layer_scalar buffers on 3 layers.
    inner = torch.nn.Module()
    inner.decoder = torch.nn.Module()
    layers = []
    for _ in range(3):
        layer = torch.nn.Module()
        layer.register_buffer("layer_scalar", torch.ones(1))
        layers.append(layer)
    inner.decoder.layers = torch.nn.ModuleList(layers)

    # Stub get_transformer_layer_offset -> 0 (no PP).
    import megatron.core.transformer.transformer_layer as tl
    orig_offset = tl.get_transformer_layer_offset
    tl.get_transformer_layer_offset = lambda _cfg: 0
    try:
        _provider._load_layer_scalars(inner, str(tmp_path), config=SimpleNamespace())
    finally:
        tl.get_transformer_layer_offset = orig_offset

    for i, expected in scalars.items():
        assert inner.decoder.layers[i].layer_scalar.item() == pytest.approx(expected), (
            f"layer {i}: expected {expected}, got {inner.decoder.layers[i].layer_scalar.item()}"
        )


def test_load_layer_scalars_respects_pp_offset(tmp_path):
    """Under PP, inner.decoder.layers holds only this rank's local subset;
    local index i must translate to global HF index i + pp_offset."""
    # HF checkpoint has scalars for layers 10, 11, 12 (e.g., PP rank 1 of 2
    # on a 20-layer model — local layers 0..9 map to global 10..19).
    scalars = {10: 0.7, 11: 0.8, 12: 0.9}
    _write_fake_safetensors_layer_scalars(tmp_path, scalars)

    # Local `inner` has 3 layers representing global 10, 11, 12.
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
    """By default, a missing layer_scalar for any local layer fails loudly
    (wrong scalars materially change activations vs HF). This mirrors the
    provider's fail-loud posture for checkpoint drift."""
    monkeypatch.delenv("GEMMA4_ALLOW_MISSING_LAYER_SCALARS", raising=False)
    scalars = {0: 0.5}  # only layer 0 has a scalar
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
    """With GEMMA4_ALLOW_MISSING_LAYER_SCALARS=1, a missing scalar logs a
    warning and falls back to the default 1.0."""
    monkeypatch.setenv("GEMMA4_ALLOW_MISSING_LAYER_SCALARS", "1")
    scalars = {0: 0.5}  # only layer 0 has a scalar
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
    """Missing index.json is fail-loud by default (checkpoint lacks the
    layer_scalar tensors we need). The legacy skip-and-warn behavior is
    available via GEMMA4_ALLOW_MISSING_LAYER_SCALARS=1."""
    monkeypatch.delenv("GEMMA4_ALLOW_MISSING_LAYER_SCALARS", raising=False)
    inner = torch.nn.Module()
    inner.decoder = torch.nn.Module()
    inner.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    inner.decoder.layers[0].register_buffer("layer_scalar", torch.ones(1))

    # No index.json in tmp_path — read returns None, provider should raise.
    with pytest.raises(RuntimeError, match="No layer_scalar weights found"):
        _provider._load_layer_scalars(inner, str(tmp_path), config=SimpleNamespace())


def test_load_layer_scalars_skips_when_no_index_file_with_opt_in(tmp_path, monkeypatch, caplog):
    """With opt-in flag, missing index.json degrades to a warning + default."""
    import logging
    monkeypatch.setenv("GEMMA4_ALLOW_MISSING_LAYER_SCALARS", "1")
    inner = torch.nn.Module()
    inner.decoder = torch.nn.Module()
    inner.decoder.layers = torch.nn.ModuleList([torch.nn.Module()])
    inner.decoder.layers[0].register_buffer("layer_scalar", torch.ones(1))

    # No index.json in tmp_path.
    with caplog.at_level(logging.WARNING, logger=_provider.__name__):
        _provider._load_layer_scalars(inner, str(tmp_path), config=SimpleNamespace())
    # Scalar unchanged.
    assert inner.decoder.layers[0].layer_scalar.item() == 1.0
    assert any("No safetensors index" in r.message for r in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
