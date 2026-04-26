import sys
import types
from argparse import Namespace

import pytest
import torch

from slime.backends.megatron_utils import peft


def _args(**overrides):
    values = {
        "enable_lora": True,
        "train_backend": "megatron",
        "megatron_to_hf_mode": "bridge",
        "advantage_estimator": "grpo",
        "colocate": True,
        "debug_train_only": False,
        "custom_model_provider_path": None,
        "only_train_params_name_list": None,
        "freeze_params_name_list": None,
        "enable_weights_backuper": True,
        "use_opd": False,
        "num_experts": None,
        "ref_update_interval": None,
        "lora_target_modules": None,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture
def fake_lora_runtime(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    bridge_mod = types.ModuleType("megatron.bridge")
    peft_mod = types.ModuleType("megatron.bridge.peft")
    lora_mod = types.ModuleType("megatron.bridge.peft.lora")

    class FakeLoRA:
        instances = []

        def __init__(self, target_modules=None, dim=32, alpha=32, dropout=0.0):
            self.target_modules = target_modules
            self.dim = dim
            self.alpha = alpha
            self.dropout = dropout
            FakeLoRA.instances.append(self)

        def __call__(self, model, training=True):
            model.lora_applied = True
            model.lora_training = training
            model.lora_config = self
            return model

    class FakeLoRAMerge:
        def transform(self, module):
            module.merge_count = getattr(module, "merge_count", 0) + 1
            return module

    lora_mod.LoRA = FakeLoRA
    lora_mod.LoRAMerge = FakeLoRAMerge

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge_mod)
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft", peft_mod)
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.lora", lora_mod)

    return FakeLoRA, FakeLoRAMerge


def test_validate_lora_args_disabled_does_not_require_runtime():
    peft.validate_lora_args(_args(enable_lora=False))


def test_validate_lora_args_accepts_supported_first_slice(fake_lora_runtime):
    peft.validate_lora_args(_args())


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"megatron_to_hf_mode": "raw"}, "--megatron-to-hf-mode bridge"),
        ({"advantage_estimator": "ppo"}, "--advantage-estimator grpo"),
        ({"colocate": False}, "--colocate"),
        ({"custom_model_provider_path": "custom.provider"}, "--custom-model-provider-path"),
        ({"only_train_params_name_list": ["adapter"]}, "--only-train-params-name-list"),
        ({"freeze_params_name_list": ["linear"]}, "--freeze-params-name-list"),
        ({"enable_weights_backuper": False}, "--disable-weights-backuper"),
        ({"use_opd": True}, "on-policy distillation"),
        ({"num_experts": 8}, "MoE models"),
        ({"ref_update_interval": 10}, "--ref-update-interval"),
        ({"lora_rank": 0}, "--lora-rank"),
        ({"lora_alpha": 0}, "--lora-alpha"),
        ({"lora_dropout": 1.0}, "--lora-dropout"),
        ({"lora_dropout": -0.1}, "--lora-dropout"),
    ],
)
def test_validate_lora_args_rejects_unsupported_combinations(fake_lora_runtime, override, expected):
    with pytest.raises(ValueError, match=expected):
        peft.validate_lora_args(_args(**override))


def test_validate_lora_args_allows_non_colocated_debug_train_only(fake_lora_runtime):
    peft.validate_lora_args(_args(colocate=False, debug_train_only=True))


def test_build_lora_config_maps_cli_args(fake_lora_runtime):
    FakeLoRA, _ = fake_lora_runtime

    config = peft.build_lora_config(
        _args(
            lora_target_modules=["linear_qkv", "linear_proj"],
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
    )

    assert isinstance(config, FakeLoRA)
    assert config.target_modules == ["linear_qkv", "linear_proj"]
    assert config.dim == 8
    assert config.alpha == 16
    assert config.dropout == 0.1


def test_maybe_apply_lora_only_applies_to_actor(fake_lora_runtime):
    actor_model = types.SimpleNamespace()
    critic_model = types.SimpleNamespace()

    applied = peft.maybe_apply_lora(actor_model, _args(), role="actor")
    untouched = peft.maybe_apply_lora(critic_model, _args(), role="critic")

    assert applied is actor_model
    assert actor_model.lora_applied is True
    assert actor_model._slime_lora_enabled is True
    assert untouched is critic_model
    assert not hasattr(critic_model, "lora_applied")


def test_merge_lora_weights_for_export_visits_modules(fake_lora_runtime):
    class FakeModule:
        def __init__(self, children=None):
            self.children = children or []

        def modules(self):
            yield self
            for child in self.children:
                yield child

    child = FakeModule()
    root = FakeModule(children=[child])

    peft.merge_lora_weights_for_export(root)

    assert root.merge_count == 1
    assert child.merge_count == 1


class _FakeVpStage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.Module()
        self.module.module = torch.nn.Module()
        self.module.module.linear = torch.nn.Linear(2, 2, bias=False)


def test_restore_model_from_named_tensors_restores_vanilla_backup_names():
    stage = _FakeVpStage()
    stage.module.module.linear.weight.data.zero_()
    backup = {"vp_stages.0.linear.weight": torch.ones_like(stage.module.module.linear.weight)}

    peft.restore_model_from_named_tensors([stage], backup)

    assert torch.equal(stage.module.module.linear.weight, backup["vp_stages.0.linear.weight"])


def test_restore_model_from_named_tensors_rejects_missing_backup_tensor():
    stage = _FakeVpStage()

    with pytest.raises(KeyError, match="missing model tensors"):
        peft.restore_model_from_named_tensors([stage], {})
