import importlib.util
import pickle
import sys
import types
from pathlib import Path

import pytest
import torch

# The CPU CI image does not install Megatron; mount the source package the
# same way tests/test_hf_to_megatron.py does, without its runtime patches.
try:
    _has_megatron = importlib.util.find_spec("megatron.core") is not None
except ModuleNotFoundError:
    _has_megatron = False
if not _has_megatron:
    _megatron_utils = types.ModuleType("slime.backends.megatron_utils")
    _megatron_utils.__path__ = [str(Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils")]
    sys.modules["slime.backends.megatron_utils"] = _megatron_utils

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool(module_name):
    path = _REPO_ROOT / "tools" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    unpickler = pickle.Unpickler
    try:
        spec.loader.exec_module(module)
    finally:
        # Both command-line tools replace the global Unpickler on import.
        pickle.Unpickler = unpickler
    return module


convert_torch_dist_to_hf = _load_tool("convert_torch_dist_to_hf")
convert_torch_dist_to_hf_parallel = _load_tool("convert_torch_dist_to_hf_parallel")

NUM_GPUS = 0

IN_PROJ_PREFIX = "decoder.layers.0.self_attention.in_proj.weight"
CONV1D_PREFIX = "decoder.layers.3.self_attention.conv1d.weight"


@pytest.fixture(params=[convert_torch_dist_to_hf, convert_torch_dist_to_hf_parallel], ids=["serial", "parallel"])
def converter(request):
    return request.param


@pytest.fixture
def in_proj_sections():
    return {
        "query": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "key": torch.arange(6, 12, dtype=torch.float32).reshape(2, 3),
        "value": torch.arange(12, 24, dtype=torch.float32).reshape(4, 3),
        "z": torch.arange(24, 36, dtype=torch.float32).reshape(4, 3),
        "beta": torch.arange(36, 39, dtype=torch.float32).reshape(1, 3),
        "alpha": torch.arange(39, 42, dtype=torch.float32).reshape(1, 3),
    }


@pytest.fixture
def conv1d_sections():
    return {
        "query": torch.arange(4, dtype=torch.float32).reshape(2, 1, 2),
        "key": torch.arange(4, 8, dtype=torch.float32).reshape(2, 1, 2),
        "value": torch.arange(8, 16, dtype=torch.float32).reshape(4, 1, 2),
    }


def test_merge_gdn_in_proj_sections(converter, in_proj_sections):
    # Checkpoint key iteration order must not determine projection order.
    order = ("alpha", "query", "beta", "value", "z", "key")
    state_dict = {f"{IN_PROJ_PREFIX}.{name}": in_proj_sections[name] for name in order}
    router = torch.ones(2, 3)
    state_dict["decoder.layers.0.mlp.router.weight"] = router

    converter._merge_gdn_sections(state_dict)

    expected = torch.cat([in_proj_sections[name] for name in ("query", "key", "value", "z", "beta", "alpha")])
    assert state_dict[IN_PROJ_PREFIX].shape == (14, 3)
    assert state_dict[IN_PROJ_PREFIX].partition_sections == (2, 2, 4, 4, 1, 1)
    torch.testing.assert_close(state_dict[IN_PROJ_PREFIX], expected, rtol=0, atol=0)
    assert all(f"{IN_PROJ_PREFIX}.{name}" not in state_dict for name in in_proj_sections)
    assert state_dict["decoder.layers.0.mlp.router.weight"] is router


def test_merge_gdn_conv1d_sections(converter, conv1d_sections):
    state_dict = {f"{CONV1D_PREFIX}.{name}": conv1d_sections[name] for name in ("value", "query", "key")}

    converter._merge_gdn_sections(state_dict)

    expected = torch.cat([conv1d_sections[name] for name in ("query", "key", "value")])
    assert state_dict[CONV1D_PREFIX].shape == (8, 1, 2)
    assert state_dict[CONV1D_PREFIX].partition_sections == (2, 2, 4)
    torch.testing.assert_close(state_dict[CONV1D_PREFIX], expected, rtol=0, atol=0)
    assert set(state_dict) == {CONV1D_PREFIX}


def test_merge_gdn_rejects_incomplete_sections(converter):
    state_dict = {
        f"{IN_PROJ_PREFIX}.query": torch.ones(1, 3),
        f"{IN_PROJ_PREFIX}.key": torch.ones(1, 3),
    }

    with pytest.raises(RuntimeError, match="unexpected GDN sections"):
        converter._merge_gdn_sections(state_dict)


def test_merge_gdn_is_noop_for_packed_parameters(converter):
    packed = torch.ones(14, 3)
    packed.partition_sections = (2, 2, 4, 4, 1, 1)
    state_dict = {IN_PROJ_PREFIX: packed}

    converter._merge_gdn_sections(state_dict)

    assert state_dict[IN_PROJ_PREFIX] is packed
    assert packed.partition_sections == (2, 2, 4, 4, 1, 1)


def test_merge_gdn_rejects_packed_and_split_parameters(converter, in_proj_sections):
    state_dict = {f"{IN_PROJ_PREFIX}.{name}": tensor for name, tensor in in_proj_sections.items()}
    state_dict[IN_PROJ_PREFIX] = torch.ones(14, 3)

    with pytest.raises(RuntimeError):
        converter._merge_gdn_sections(state_dict)


def test_merged_gdn_parameters_convert_to_hf(converter, in_proj_sections, conv1d_sections):
    state_dict = {f"{IN_PROJ_PREFIX}.{name}": in_proj_sections[name] for name in reversed(in_proj_sections)}
    state_dict.update({f"{CONV1D_PREFIX}.{name}": conv1d_sections[name] for name in reversed(conv1d_sections)})
    args = types.SimpleNamespace(
        num_layers=4,
        hidden_size=3,
        num_attention_heads=1,
        num_query_groups=1,
        kv_channels=3,
        vocab_size=16,
        q_lora_rank=None,
    )

    converter._merge_gdn_sections(state_dict)
    converted = {}
    for name, param in converter.get_named_params(args, state_dict):
        converted.update(converter.convert_to_hf(args, "qwen3_5", name, param))

    prefix = "model.language_model.layers.0.linear_attn"
    expected = {
        f"{prefix}.in_proj_qkv.weight": torch.cat([in_proj_sections[name] for name in ("query", "key", "value")]),
        f"{prefix}.in_proj_z.weight": in_proj_sections["z"],
        f"{prefix}.in_proj_b.weight": in_proj_sections["beta"],
        f"{prefix}.in_proj_a.weight": in_proj_sections["alpha"],
        "model.language_model.layers.3.linear_attn.conv1d.weight": torch.cat(
            [conv1d_sections[name] for name in ("query", "key", "value")]
        ),
    }
    assert set(converted) == set(expected)
    for name, tensor in expected.items():
        torch.testing.assert_close(converted[name], tensor, rtol=0, atol=0)


def test_group_gdn_sections_keeps_each_projection_together():
    in_proj_keys = [f"{IN_PROJ_PREFIX}.{name}" for name in ("alpha", "query", "beta", "value", "z", "key")]
    conv1d_keys = [f"{CONV1D_PREFIX}.{name}" for name in ("value", "query", "key")]
    # Interleave the two projections and an unrelated parameter.
    keys = in_proj_keys[:2] + conv1d_keys[:1] + ["decoder.layers.0.mlp.router.weight"]
    keys += in_proj_keys[2:] + conv1d_keys[1:]

    groups = convert_torch_dist_to_hf_parallel._group_gdn_sections(keys)

    assert set(groups) == {IN_PROJ_PREFIX, CONV1D_PREFIX}
    assert groups[IN_PROJ_PREFIX] == in_proj_keys
    assert groups[CONV1D_PREFIX] == conv1d_keys


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
