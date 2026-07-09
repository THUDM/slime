import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


NUM_GPUS = 0


def load_raw_export_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "megatron_to_hf" / "gpt_oss.py"
    )
    module_name = "test_gpt_oss_raw_export_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def convert_args():
    return types.SimpleNamespace(
        hidden_size=6,
        kv_channels=None,
        num_attention_heads=2,
        num_query_groups=1,
        num_experts=2,
    )


def convert_expert(module, args, name, param, expert_idx):
    return module.convert_gpt_oss_to_hf(
        args,
        f"module.module.decoder.layers.3.mlp.experts.{name}{expert_idx}",
        param,
    )


def interleave_gate_up(param):
    gate, up = param.chunk(2, dim=0)
    return torch.stack([gate, up], dim=1).reshape(-1, *param.shape[1:]).contiguous()


@pytest.mark.unit
def test_gpt_oss_raw_fc1_weight_fuses_experts_for_sglang_gate_up_proj():
    module = load_raw_export_module()
    args = convert_args()
    expert0 = torch.arange(24, dtype=torch.float32).view(4, 6)
    expert1 = expert0 + 100

    assert convert_expert(module, args, "linear_fc1.weight", expert1, 1) == []
    converted = convert_expert(module, args, "linear_fc1.weight", expert0, 0)

    expected = torch.stack(
        [
            interleave_gate_up(expert0).transpose(0, 1).contiguous(),
            interleave_gate_up(expert1).transpose(0, 1).contiguous(),
        ],
        dim=0,
    )
    assert len(converted) == 1
    hf_name, hf_weight = converted[0]
    assert hf_name == "model.layers.3.mlp.experts.gate_up_proj"
    assert torch.equal(hf_weight, expected)


@pytest.mark.unit
def test_gpt_oss_raw_fc2_weight_transposes_and_fuses_experts_for_sglang_down_proj():
    module = load_raw_export_module()
    args = convert_args()
    expert0 = torch.arange(18, dtype=torch.float32).view(6, 3)
    expert1 = expert0 + 100

    assert convert_expert(module, args, "linear_fc2.weight", expert0, 0) == []
    converted = convert_expert(module, args, "linear_fc2.weight", expert1, 1)

    expected = torch.stack([expert0.transpose(0, 1).contiguous(), expert1.transpose(0, 1).contiguous()], dim=0)
    assert len(converted) == 1
    hf_name, hf_weight = converted[0]
    assert hf_name == "model.layers.3.mlp.experts.down_proj"
    assert torch.equal(hf_weight, expected)


@pytest.mark.unit
def test_gpt_oss_raw_fc1_bias_fuses_interleaved_gate_up_biases():
    module = load_raw_export_module()
    args = convert_args()
    expert0 = torch.arange(4, dtype=torch.float32)
    expert1 = expert0 + 100

    assert convert_expert(module, args, "linear_fc1.bias", expert0, 0) == []
    converted = convert_expert(module, args, "linear_fc1.bias", expert1, 1)

    expected = torch.stack([interleave_gate_up(expert0), interleave_gate_up(expert1)], dim=0)
    assert len(converted) == 1
    hf_name, hf_weight = converted[0]
    assert hf_name == "model.layers.3.mlp.experts.gate_up_proj_bias"
    assert torch.equal(hf_weight, expected)


@pytest.mark.unit
def test_gpt_oss_raw_fc2_bias_fuses_down_proj_biases():
    module = load_raw_export_module()
    args = convert_args()
    expert0 = torch.arange(3, dtype=torch.float32)
    expert1 = expert0 + 100

    assert convert_expert(module, args, "linear_fc2.bias", expert1, 1) == []
    converted = convert_expert(module, args, "linear_fc2.bias", expert0, 0)

    expected = torch.stack([expert0, expert1], dim=0)
    assert len(converted) == 1
    hf_name, hf_weight = converted[0]
    assert hf_name == "model.layers.3.mlp.experts.down_proj_bias"
    assert torch.equal(hf_weight, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
