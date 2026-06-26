import pytest
import torch

from tests.gemma4._standalone_imports import load_gemma4_provider_module

DualRotaryEmbedding = load_gemma4_provider_module().DualRotaryEmbedding


class _FakeRope:
    def __init__(self, dim: int, tag: float):
        self.dim = dim
        self.tag = tag
        self.calls = []

    def __call__(self, seq_len, **kwargs):
        self.calls.append((seq_len, kwargs))
        s = torch.arange(seq_len, dtype=torch.float).view(seq_len, 1, 1, 1)
        d = torch.arange(self.dim, dtype=torch.float).view(1, 1, 1, self.dim)
        return s * 100.0 + d + self.tag

    def get_rotary_seq_len(self, *args, **kwargs):
        return ("fake_seq_len_result", args, kwargs)


def test_dual_rope_concat_shape_global_first():
    local = _FakeRope(dim=256, tag=0.1)
    glob = _FakeRope(dim=512, tag=0.9)
    dual = DualRotaryEmbedding(local, glob, global_dim=512)

    seq_len = 16
    combined = dual(seq_len)
    assert combined.shape == (seq_len, 1, 1, 512 + 256)

    global_slice = combined[..., :512]
    local_slice = combined[..., 512:]
    assert torch.equal(global_slice, glob(seq_len))
    assert torch.equal(local_slice, local(seq_len))


def test_dual_rope_split_matches_layer_convention():
    global_dim, local_dim = 384, 192
    local = _FakeRope(dim=local_dim, tag=11.0)
    glob = _FakeRope(dim=global_dim, tag=22.0)
    dual = DualRotaryEmbedding(local, glob, global_dim=global_dim)

    seq_len = 8
    combined = dual(seq_len)

    for is_sliding, expected_rope in [(False, glob), (True, local)]:
        if is_sliding:
            sliced = combined[..., global_dim:]
        else:
            sliced = combined[..., :global_dim]
        assert torch.equal(
            sliced, expected_rope(seq_len)
        ), f"split for is_sliding={is_sliding} did not recover the right rope"


def test_dual_rope_delegates_get_rotary_seq_len_to_local():
    local = _FakeRope(dim=256, tag=0.0)
    glob = _FakeRope(dim=512, tag=0.0)
    dual = DualRotaryEmbedding(local, glob, global_dim=512)

    result = dual.get_rotary_seq_len("a", b=2)
    assert result[0] == "fake_seq_len_result"
    assert result[1] == ("a",)
    assert result[2] == {"b": 2}


def test_dual_rope_forwards_packed_seq_params_to_both_ropes():
    local = _FakeRope(dim=4, tag=0.0)
    glob = _FakeRope(dim=8, tag=0.0)
    dual = DualRotaryEmbedding(local, glob, global_dim=8)
    packed_seq_params = object()

    combined = dual(12, offset=3, packed_seq_params=packed_seq_params)

    assert combined.shape == (12, 1, 1, 12)
    assert glob.calls == [(12, {"offset": 3, "packed_seq_params": packed_seq_params})]
    assert local.calls == [(12, {"offset": 3, "packed_seq_params": packed_seq_params})]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron RotaryEmbedding.forward requires CUDA")
def test_dual_rope_end_to_end_with_real_megatron_rope():
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    local = RotaryEmbedding(kv_channels=256, rotary_percent=1.0, rotary_base=10_000.0)
    glob = RotaryEmbedding(kv_channels=512, rotary_percent=1.0, rotary_base=1_000_000.0)
    dual = DualRotaryEmbedding(local, glob, global_dim=512)

    combined = dual(64)
    assert combined.shape[-1] == 512 + 256
    assert torch.equal(combined[..., :512], glob(64))
    assert torch.equal(combined[..., 512:], local(64))
