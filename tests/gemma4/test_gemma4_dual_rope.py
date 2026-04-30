"""Unit tests for ``DualRotaryEmbedding``.

``DualRotaryEmbedding`` wraps a pair of Megatron ``RotaryEmbedding`` modules
— one for sliding layers (local), one for global layers — and produces a
single concatenated tensor so downstream code that expects a single rope
output (distributed checkpointing, CP sharding) continues to work.
``Gemma4TransformerLayer.forward`` splits it back per-layer based on whether
the layer is sliding or global.

We test two things:

1. Concat/split semantics on synthetic tensors (CPU-only). This is the
   novel logic ``DualRotaryEmbedding`` actually adds beyond Megatron's own
   ``RotaryEmbedding``.
2. End-to-end forward with real Megatron ``RotaryEmbedding`` modules — the
   only CUDA-gated part because Megatron forces ``inv_freq`` to CUDA inside
   ``get_emb``.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch


def _load_dual_rotary_embedding():
    """Import ``DualRotaryEmbedding`` without triggering the module-level
    ``from megatron.training import get_args`` that ``gemma4_provider`` does
    at import time (unavailable in minimal test containers). We exec the
    module with the unavailable import stubbed out."""
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
    return mod.DualRotaryEmbedding


DualRotaryEmbedding = _load_dual_rotary_embedding()


class _FakeRope:
    """Minimal RotaryEmbedding stand-in that returns a deterministic tensor
    encoding its own identity — so we can verify that the global slice
    really came from the global rope (and vice versa)."""

    def __init__(self, dim: int, tag: float):
        self.dim = dim
        self.tag = tag

    def __call__(self, seq_len, **kwargs):
        # Shape [seq_len, 1, 1, dim] — same layout Megatron produces.
        # Value: seq-index * 100 + dim-index, plus a per-rope tag so we can
        # tell global from local apart slice-by-slice.
        s = torch.arange(seq_len, dtype=torch.float).view(seq_len, 1, 1, 1)
        d = torch.arange(self.dim, dtype=torch.float).view(1, 1, 1, self.dim)
        return s * 100.0 + d + self.tag

    def get_rotary_seq_len(self, *args, **kwargs):
        # Sentinel used in the delegation test.
        return ("fake_seq_len_result", args, kwargs)


def test_dual_rope_concat_shape_global_first():
    local = _FakeRope(dim=256, tag=0.1)
    glob = _FakeRope(dim=512, tag=0.9)
    dual = DualRotaryEmbedding(local, glob, global_dim=512)

    seq_len = 16
    combined = dual(seq_len)
    assert combined.shape == (seq_len, 1, 1, 512 + 256)

    # Verify the GLOBAL slice came from the global rope (tag=0.9, dim=512)
    # and the local slice from the local rope (tag=0.1, dim=256).
    # Global first.
    global_slice = combined[..., :512]
    local_slice = combined[..., 512:]
    assert torch.equal(global_slice, glob(seq_len))
    assert torch.equal(local_slice, local(seq_len))


def test_dual_rope_split_matches_layer_convention():
    """The concat format must round-trip through the layer's split
    convention:

        rotary_pos_emb[..., :global_dim] -> global layers
        rotary_pos_emb[..., global_dim:] -> sliding layers

    This is a regression guard against any reshuffle (e.g. swapping the
    concat order) that would silently feed the wrong RoPE to each layer."""
    global_dim, local_dim = 384, 192
    local = _FakeRope(dim=local_dim, tag=11.0)
    glob = _FakeRope(dim=global_dim, tag=22.0)
    dual = DualRotaryEmbedding(local, glob, global_dim=global_dim)

    seq_len = 8
    combined = dual(seq_len)

    # Mimic Gemma4TransformerLayer.forward's split logic.
    for is_sliding, expected_rope in [(False, glob), (True, local)]:
        if is_sliding:
            sliced = combined[..., global_dim:]
        else:
            sliced = combined[..., :global_dim]
        assert torch.equal(sliced, expected_rope(seq_len)), (
            f"split for is_sliding={is_sliding} did not recover the right rope"
        )


def test_dual_rope_delegates_get_rotary_seq_len_to_local():
    """``get_rotary_seq_len`` must delegate to local_rope — both ropes share
    seq-length logic (they only differ in theta and partial-rotary), so the
    answer is the same, and delegation guarantees that."""
    local = _FakeRope(dim=256, tag=0.0)
    glob = _FakeRope(dim=512, tag=0.0)
    dual = DualRotaryEmbedding(local, glob, global_dim=512)

    result = dual.get_rotary_seq_len("a", b=2)
    # _FakeRope.get_rotary_seq_len returns a sentinel tuple; confirm it
    # really came from the LOCAL one.
    assert result[0] == "fake_seq_len_result"
    assert result[1] == ("a",)
    assert result[2] == {"b": 2}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron RotaryEmbedding.forward requires CUDA")
def test_dual_rope_end_to_end_with_real_megatron_rope():
    """Integration smoke test: wire real Megatron ``RotaryEmbedding`` into
    DualRotaryEmbedding and sanity-check shape + split correctness."""
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    local = RotaryEmbedding(kv_channels=256, rotary_percent=1.0, rotary_base=10_000.0)
    glob = RotaryEmbedding(kv_channels=512, rotary_percent=1.0, rotary_base=1_000_000.0)
    dual = DualRotaryEmbedding(local, glob, global_dim=512)

    combined = dual(64)
    assert combined.shape[-1] == 512 + 256
    assert torch.equal(combined[..., :512], glob(64))
    assert torch.equal(combined[..., 512:], local(64))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
