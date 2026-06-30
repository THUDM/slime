import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools import convert_hf_to_int4_direct as converter


def _load_module(module_name, relative_path):
    module_path = REPO_ROOT / relative_path
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


qwen3_5 = _load_module(
    "test_qwen35_megatron_to_hf_qwen3_5",
    "slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py",
)
quantizer_compressed_tensors = _load_module(
    "test_qwen35_quantizer_compressed_tensors",
    "slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py",
)


NUM_EXPERTS = 256
EP_SIZE = 4
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE
FFN = 8
HIDDEN = 4


def _args():
    return SimpleNamespace(
        num_experts=NUM_EXPERTS,
        kv_channels=None,
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_query_groups=1,
    )


def _ep_offset(ep_rank):
    return ep_rank * NUM_EXPERTS // EP_SIZE


def _expert_ids(named_params, suffix):
    ids = []
    for name, _ in named_params:
        marker = ".mlp.experts."
        if suffix in name and marker in name:
            ids.append(int(name.split(marker, 1)[1].split(".", 1)[0]))
    return sorted(ids)


@pytest.mark.unit
def test_runtime_fused_experts_keep_ep_global_ids():
    args = _args()
    all_gate_ids = []
    all_down_ids = []

    for ep_rank in range(EP_SIZE):
        offset = _ep_offset(ep_rank)
        fc1_name = f"module.module.decoder.layers.0.mlp.experts.experts.linear_fc1.weight.__ep_offset{offset}"
        fc2_name = f"module.module.decoder.layers.0.mlp.experts.experts.linear_fc2.weight.__ep_offset{offset}"

        fc1_out = qwen3_5.convert_qwen3_5_to_hf(args, fc1_name, torch.zeros(LOCAL_EXPERTS, 2 * FFN, HIDDEN))
        fc2_out = qwen3_5.convert_qwen3_5_to_hf(args, fc2_name, torch.zeros(LOCAL_EXPERTS, HIDDEN, FFN))

        expected = list(range(offset, offset + LOCAL_EXPERTS))
        assert _expert_ids(fc1_out, "gate_proj.weight") == expected
        assert _expert_ids(fc2_out, "down_proj.weight") == expected
        all_gate_ids.extend(expected)
        all_down_ids.extend(expected)

    assert sorted(all_gate_ids) == list(range(NUM_EXPERTS))
    assert sorted(all_down_ids) == list(range(NUM_EXPERTS))


@pytest.mark.unit
def test_runtime_fused_expert_split_matches_offline_split():
    args = _args()
    fc1 = torch.arange(LOCAL_EXPERTS * 2 * FFN * HIDDEN, dtype=torch.float32).view(LOCAL_EXPERTS, 2 * FFN, HIDDEN)
    fc2 = torch.arange(LOCAL_EXPERTS * HIDDEN * FFN, dtype=torch.float32).view(LOCAL_EXPERTS, HIDDEN, FFN)

    out = dict(
        qwen3_5.convert_qwen3_5_to_hf(
            args,
            "module.module.decoder.layers.0.mlp.experts.experts.linear_fc1.weight.__ep_offset0",
            fc1,
        )
        + qwen3_5.convert_qwen3_5_to_hf(
            args,
            "module.module.decoder.layers.0.mlp.experts.experts.linear_fc2.weight.__ep_offset0",
            fc2,
        )
    )

    for expert_id in range(LOCAL_EXPERTS):
        gate, up = fc1[expert_id].chunk(2, dim=0)
        assert torch.equal(out[f"model.language_model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"], gate)
        assert torch.equal(out[f"model.language_model.layers.0.mlp.experts.{expert_id}.up_proj.weight"], up)
        assert torch.equal(
            out[f"model.language_model.layers.0.mlp.experts.{expert_id}.down_proj.weight"], fc2[expert_id]
        )


@pytest.mark.unit
def test_offline_converter_default_ignore_rules_stay_backward_compatible():
    assert converter.DEFAULT_IGNORE_RULES == [
        "re:.*lm_head.*",
        "re:.*norm.*",
        "re:.*embed.*",
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
        "re:.*mlp\\.gate\\.*",
    ]
    for qwen35_only_rule in ["re:.*linear_attn.*", "re:.*conv1d.*", "re:.*visual.*", "re:.*mtp.*"]:
        assert qwen35_only_rule not in converter.DEFAULT_IGNORE_RULES


@pytest.mark.unit
def test_offline_converter_fused_mode_quantizes_only_split_routed_experts(monkeypatch):
    def fake_pack_layer(weight, group_size, sym=True):
        packed = torch.full((weight.shape[0], 1), fill_value=weight.shape[0], dtype=torch.int32)
        scale = torch.ones(weight.shape[0], 1, dtype=weight.dtype)
        return packed, scale, None

    monkeypatch.setattr(converter, "pack_layer", fake_pack_layer)

    weights = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 2 * FFN, HIDDEN),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(2, HIDDEN, FFN),
        "model.language_model.layers.0.linear_attn.out_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(2, HIDDEN),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(HIDDEN, FFN),
        "model.visual.patch_embed.weight": torch.randn(HIDDEN, HIDDEN),
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(HIDDEN, FFN),
    }

    q_weights = converter.convert_weights(
        weights,
        group_size=128,
        is_symmetric=True,
        ignore_rules=converter.DEFAULT_IGNORE_RULES,
        qwen35_fused_expert_only=True,
    )

    for expert_id in range(2):
        assert f"model.language_model.layers.0.mlp.experts.{expert_id}.gate_proj.weight_packed" in q_weights
        assert f"model.language_model.layers.0.mlp.experts.{expert_id}.up_proj.weight_packed" in q_weights
        assert f"model.language_model.layers.0.mlp.experts.{expert_id}.down_proj.weight_packed" in q_weights

    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in q_weights
    assert "model.language_model.layers.0.mlp.experts.down_proj" not in q_weights
    assert "model.language_model.layers.0.linear_attn.out_proj.weight" in q_weights
    assert "model.language_model.layers.0.linear_attn.out_proj.weight_packed" not in q_weights
    assert "model.language_model.layers.0.mlp.gate.weight" in q_weights
    assert "model.language_model.layers.0.mlp.gate.weight_packed" not in q_weights
    assert "model.language_model.layers.0.mlp.shared_expert.down_proj.weight" in q_weights
    assert "model.language_model.layers.0.mlp.shared_expert.down_proj.weight_packed" not in q_weights
    assert "model.visual.patch_embed.weight" in q_weights
    assert "model.visual.patch_embed.weight_packed" not in q_weights
    assert "mtp.layers.0.mlp.down_proj.weight" in q_weights
    assert "mtp.layers.0.mlp.down_proj.weight_packed" not in q_weights


@pytest.mark.unit
def test_offline_converter_non_fused_mode_keeps_existing_quantization_behavior(monkeypatch):
    def fake_pack_layer(weight, group_size, sym=True):
        return torch.ones(weight.shape[0], 1, dtype=torch.int32), torch.ones(weight.shape[0], 1), None

    monkeypatch.setattr(converter, "pack_layer", fake_pack_layer)

    q_weights = converter.convert_weights(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
            "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(HIDDEN, FFN),
        },
        group_size=128,
        is_symmetric=True,
        ignore_rules=converter.DEFAULT_IGNORE_RULES,
        qwen35_fused_expert_only=False,
    )

    assert "model.layers.0.self_attn.q_proj.weight" in q_weights
    assert "model.layers.0.self_attn.q_proj.weight_packed" not in q_weights
    assert "model.layers.0.mlp.experts.0.down_proj.weight" not in q_weights
    assert "model.layers.0.mlp.experts.0.down_proj.weight_packed" in q_weights


@pytest.mark.unit
def test_runtime_quantizer_skips_non_2d_tensors(monkeypatch):
    def fail_pack_layer(weight, group_size, sym=True):
        raise AssertionError("3D tensors must not be packed")

    monkeypatch.setattr(quantizer_compressed_tensors, "pack_layer", fail_pack_layer)

    result = quantizer_compressed_tensors.quantize_params_compressed_tensors(
        [("model.language_model.layers.0.linear_attn.conv1d.weight", torch.randn(2, 3, 4))],
        {
            "config_groups": {
                "group_0": {
                    "weights": {
                        "group_size": 128,
                        "symmetric": True,
                    }
                }
            },
            "ignore": [],
        },
    )

    assert result[0][0] == "model.language_model.layers.0.linear_attn.conv1d.weight"
    assert result[0][1].shape == (2, 3, 4)


@pytest.mark.unit
def test_qwen35_effective_quantization_config_keeps_non_experts_bf16():
    effective_ignore = converter.get_effective_ignore_rules(qwen35_fused_expert_only=True)

    assert "re:.*linear_attn.*" in effective_ignore
    assert "re:.*conv1d.*" in effective_ignore
    assert "re:.*visual.*" in effective_ignore
    assert "re:.*shared_expert.*" in effective_ignore
    assert "re:.*mtp.*" in effective_ignore


if __name__ == "__main__":
    pytest.main([__file__])
