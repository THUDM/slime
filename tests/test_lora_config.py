"""CPU unit tests for the LoRA configuration layer.

Covers argument defaults, every fail-fast validation rule, target-preset
resolution and adapter-metadata compatibility checks. Nothing here imports torch
or Megatron: ``slime.utils.lora_config`` is deliberately dependency-free so that
``slime/utils/arguments.py`` can validate the CLI before a backend is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from slime.utils.lora_config import (
    ADAPTER_CONFIG_FILE,
    DEFAULT_LORA_EXCLUDE_MODULES,
    DEFAULT_LORA_TARGET_PRESET,
    LORA_TARGET_PRESETS,
    LoRAConfig,
    LoRAConfigError,
    add_lora_arguments,
    compute_base_model_config_hash,
    is_lora_param_name,
    validate_lora_args,
    validate_lora_checkpoint_args,
)

NUM_GPUS = 0


def _args(argv: list[str] | None = None, **overrides):
    parser = argparse.ArgumentParser()
    add_lora_arguments(parser)
    args = parser.parse_args(argv or [])
    args.train_backend = "megatron"
    args.lr = 1e-6
    args.hf_checkpoint = None
    args.finetune = True
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# --------------------------------------------------------------------------- #
# defaults / off-by-default guarantee
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_lora_is_disabled_by_default_and_validation_is_a_noop():
    """Existing scripts must not need a single new flag."""
    args = _args()
    assert args.use_lora is False
    # Deliberately leave out everything LoRA validation could look at: with the
    # feature off, validate_lora_args must not touch args at all.
    validate_lora_args(argparse.Namespace())
    validate_lora_args(args)


@pytest.mark.unit
def test_defaults_match_the_documented_values():
    args = _args()
    assert (args.lora_rank, args.lora_alpha, args.lora_dropout) == (64, 128.0, 0.0)
    assert args.lora_bias == "none"
    assert args.lora_rollout_sync_mode == "merged"
    assert args.lora_weight_decay == 0.0
    assert args.lora_learning_rate is None
    assert args.lora_target_preset is None and args.lora_target_modules is None


@pytest.mark.unit
def test_config_from_args_resolves_default_preset_and_scale():
    config = LoRAConfig.from_args(_args(["--use-lora"]))
    assert config.preset == DEFAULT_LORA_TARGET_PRESET
    assert config.target_modules == LORA_TARGET_PRESETS[DEFAULT_LORA_TARGET_PRESET]
    assert config.exclude_modules == DEFAULT_LORA_EXCLUDE_MODULES
    assert config.scale == pytest.approx(128.0 / 64.0)


# --------------------------------------------------------------------------- #
# numeric validation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (["--use-lora", "--lora-rank", "0"], "--lora-rank must be > 0"),
        (["--use-lora", "--lora-rank", "-8"], "--lora-rank must be > 0"),
        (["--use-lora", "--lora-alpha", "0"], "--lora-alpha must be > 0"),
        (["--use-lora", "--lora-alpha", "-1"], "--lora-alpha must be > 0"),
        (["--use-lora", "--lora-dropout", "1.0"], "must be in [0.0, 1.0)"),
        (["--use-lora", "--lora-dropout", "-0.1"], "must be in [0.0, 1.0)"),
    ],
)
def test_illegal_numeric_values_are_rejected(argv, needle):
    with pytest.raises(LoRAConfigError) as excinfo:
        validate_lora_args(_args(argv))
    assert needle in str(excinfo.value)


@pytest.mark.unit
def test_nonzero_dropout_is_rejected_for_on_policy_rl():
    """A stochastic LoRA forward would desynchronise training from the rollout policy."""
    with pytest.raises(LoRAConfigError) as excinfo:
        validate_lora_args(_args(["--use-lora", "--lora-dropout", "0.05"]))
    message = str(excinfo.value)
    assert "on-policy" in message and "rollout policy" in message


# --------------------------------------------------------------------------- #
# target selection
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_preset_and_explicit_targets_are_mutually_exclusive():
    argv = ["--use-lora", "--lora-target-preset", "dense_mlp", "--lora-target-modules", "linear_fc1$"]
    with pytest.raises(LoRAConfigError, match="only ONE of"):
        validate_lora_args(_args(argv))


@pytest.mark.unit
def test_unknown_preset_is_rejected_with_the_available_list():
    with pytest.raises(LoRAConfigError) as excinfo:
        validate_lora_args(_args(["--use-lora", "--lora-target-preset", "llama_qkvo"]))
    assert DEFAULT_LORA_TARGET_PRESET in str(excinfo.value)


@pytest.mark.unit
def test_explicit_targets_replace_the_preset():
    config = LoRAConfig.from_args(_args(["--use-lora", "--lora-target-modules", r"linear_fc1$", r"linear_fc2$"]))
    assert config.preset is None
    assert config.target_modules == (r"linear_fc1$", r"linear_fc2$")


@pytest.mark.unit
def test_invalid_regex_is_rejected():
    with pytest.raises(LoRAConfigError, match="Invalid LoRA module regex"):
        validate_lora_args(_args(["--use-lora", "--lora-target-modules", "linear_fc1(["]))


@pytest.mark.unit
@pytest.mark.parametrize("pattern", ["visual", r"model\.visual\..*", r"\.visual\.blocks"])
def test_vision_targets_fail_fast_instead_of_being_injected(pattern):
    """V1 keeps the vision tower frozen; asking for vision LoRA must be an error."""
    with pytest.raises(LoRAConfigError, match="[Vv]ision"):
        validate_lora_args(_args(["--use-lora", "--lora-target-modules", pattern]))


@pytest.mark.unit
def test_default_excludes_filter_embedding_output_layer_and_vision():
    config = LoRAConfig.from_args(_args(["--use-lora"]))
    assert config.matches("decoder.layers.0.self_attention.linear_qkv")
    assert config.matches("language_model.decoder.layers.11.mlp.linear_fc2")
    assert not config.matches("embedding.word_embeddings")
    assert not config.matches("output_layer")
    assert not config.matches("visual.blocks.0.attn.qkv")


@pytest.mark.unit
def test_hybrid_preset_matches_projections_without_allowing_replicated_modules():
    config = LoRAConfig.from_args(_args(["--use-lora", "--lora-target-preset", "hybrid_language"]))
    assert config.matches("decoder.layers.4.self_attention.linear_attn.in_proj_qkv")
    assert config.allow_replicated_modules == ()


@pytest.mark.unit
@pytest.mark.parametrize("preset", ["mla_attention", "mla_moe_language", "mla_moe_language_all"])
def test_mla_presets_are_not_supported(preset):
    with pytest.raises(LoRAConfigError, match="Unknown --lora-target-preset"):
        validate_lora_args(_args(["--use-lora", "--lora-target-preset", preset]))


@pytest.mark.unit
def test_mla_model_is_rejected_even_with_custom_targets():
    with pytest.raises(LoRAConfigError, match="MLA models"):
        validate_lora_args(
            _args(
                ["--use-lora", "--lora-target-modules", r"self_attention\.linear_proj$"],
                multi_latent_attention=True,
            )
        )


@pytest.mark.unit
def test_moe_preset_targets_shared_experts_but_never_routed_experts():
    """``moe_language`` is the conservative preset: shared experts only."""
    config = LoRAConfig.from_args(_args(["--use-lora", "--lora-target-preset", "moe_language"]))
    assert config.matches("decoder.layers.2.mlp.shared_experts.linear_fc1")
    assert config.matches("decoder.layers.2.mlp.shared_experts.linear_fc2")
    assert config.matches("decoder.layers.2.self_attention.linear_qkv")
    assert not config.matches("decoder.layers.2.mlp.experts.linear_fc1")


@pytest.mark.unit
def test_routed_expert_preset_matches_grouped_expert_projections():
    config = LoRAConfig.from_args(_args(["--use-lora", "--lora-target-preset", "moe_routed_experts"]))
    assert config.matches("decoder.layers.2.mlp.experts.linear_fc1")
    assert config.matches("decoder.layers.2.mlp.experts.linear_fc2")
    # The router is a tiny gate whose adaptation would change expert assignment.
    assert not config.matches("decoder.layers.2.mlp.router")


@pytest.mark.unit
def test_all_moe_preset_covers_routed_and_shared_experts():
    config = LoRAConfig.from_args(_args(["--use-lora", "--lora-target-preset", "moe_language_all"]))
    assert config.matches("decoder.layers.2.mlp.experts.linear_fc1")
    assert config.matches("decoder.layers.2.mlp.shared_experts.linear_fc1")


@pytest.mark.unit
def test_user_excludes_replace_the_defaults():
    config = LoRAConfig.from_args(_args(["--use-lora", "--lora-exclude-modules", r"layers\.0\."]))
    assert not config.matches("decoder.layers.0.mlp.linear_fc1")
    assert config.matches("decoder.layers.1.mlp.linear_fc1")


@pytest.mark.unit
def test_preset_targets_never_match_adapter_parameters():
    """Guards against a second adapter being injected into a LoRA module."""
    config = LoRAConfig.from_args(_args(["--use-lora"]))
    assert not config.matches("decoder.layers.0.self_attention.linear_qkv.lora_A")
    assert not config.matches("decoder.layers.0.self_attention.linear_qkv.lora_B")


# --------------------------------------------------------------------------- #
# unsupported combinations
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"train_backend": "fsdp"}, "megatron"),
        ({"only_train_params_name_list": ["experts"]}, "only-train-params-name-list"),
        ({"freeze_params_name_list": ["embedding"]}, "freeze-params-name-list"),
        ({"use_critic": True}, "critic"),
        ({"use_opd": True}, "distillation"),
        ({"opd_teacher_load": "/tmp/teacher"}, "distillation"),
        ({"keep_old_actor": True}, "keep-old-actor"),
        ({"ref_update_interval": 4}, "ref-update-interval"),
        ({"ref_load": "/tmp/ref"}, "ref-load"),
    ],
)
def test_unsupported_combinations_fail_fast(overrides, needle):
    with pytest.raises(LoRAConfigError) as excinfo:
        validate_lora_args(_args(["--use-lora"], **overrides))
    assert needle in str(excinfo.value)


@pytest.mark.unit
def test_native_adapter_sync_is_explicitly_unimplemented():
    """Never pretend to support an SGLang API that has not been verified in the image."""
    with pytest.raises(LoRAConfigError) as excinfo:
        validate_lora_args(_args(["--use-lora", "--lora-rollout-sync-mode", "native_adapter"]))
    assert "not implemented" in str(excinfo.value)


@pytest.mark.unit
def test_merged_sync_is_accepted():
    validate_lora_args(_args(["--use-lora", "--lora-rollout-sync-mode", "merged"]))


# --------------------------------------------------------------------------- #
# adapter path / metadata
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_missing_adapter_directory_is_rejected(tmp_path: Path):
    with pytest.raises(LoRAConfigError, match="not an existing directory"):
        validate_lora_checkpoint_args(_args(["--use-lora", "--lora-load", str(tmp_path / "nope")]))


@pytest.mark.unit
def test_adapter_directory_without_config_is_rejected(tmp_path: Path):
    (tmp_path / "adapter").mkdir()
    with pytest.raises(LoRAConfigError, match=ADAPTER_CONFIG_FILE):
        validate_lora_checkpoint_args(_args(["--use-lora", "--lora-load", str(tmp_path / "adapter")]))


@pytest.mark.unit
def test_valid_adapter_directory_is_accepted(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / ADAPTER_CONFIG_FILE).write_text("{}", encoding="utf-8")
    validate_lora_checkpoint_args(_args(["--use-lora", "--lora-load", str(adapter)]))


@pytest.mark.unit
def test_base_config_hash_is_stable_and_ignores_volatile_keys(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    third = tmp_path / "c"
    for path in (first, second, third):
        path.mkdir()
    base = {"model_type": "test_decoder", "hidden_size": 4096, "num_hidden_layers": 48}
    (first / "config.json").write_text(json.dumps(base), encoding="utf-8")
    (second / "config.json").write_text(
        json.dumps({**base, "transformers_version": "4.99.0", "_name_or_path": "/elsewhere"}), encoding="utf-8"
    )
    (third / "config.json").write_text(json.dumps({**base, "hidden_size": 2048}), encoding="utf-8")

    assert compute_base_model_config_hash(first) == compute_base_model_config_hash(second)
    assert compute_base_model_config_hash(first) != compute_base_model_config_hash(third)
    assert compute_base_model_config_hash(tmp_path / "missing") is None


@pytest.mark.unit
def test_metadata_payload_round_trips_as_json(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test_decoder"}), encoding="utf-8")
    config = LoRAConfig.from_args(_args(["--use-lora"], hf_checkpoint=str(tmp_path)))
    metadata = json.loads(json.dumps(config.to_metadata()))
    assert metadata["rank"] == 64
    assert metadata["alpha"] == 128.0
    assert metadata["bias"] == "none"
    assert metadata["model_type"] == "test_decoder"
    assert metadata["target_modules"] == list(config.target_modules)


@pytest.mark.unit
def test_is_lora_param_name():
    assert is_lora_param_name("module.module.decoder.layers.0.mlp.linear_fc1.lora_A")
    assert is_lora_param_name("module.module.decoder.layers.0.mlp.linear_fc1.lora_B")
    assert not is_lora_param_name("module.module.decoder.layers.0.mlp.linear_fc1.weight")
    # The MLA architecture parameters are NOT PEFT LoRA and must never be treated as such.
    assert not is_lora_param_name("module.module.decoder.layers.0.self_attention.linear_q_a_proj.weight")


@pytest.mark.unit
def test_resume_ignores_missing_initial_adapter_directory(tmp_path):
    args = _args(["--use-lora", "--lora-load", str(tmp_path / "missing")], finetune=False)
    validate_lora_args(args)
    validate_lora_checkpoint_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("preset", ["qwen3_5_attention", "qwen3_5_mlp", "qwen3_5_language"])
def test_model_specific_presets_are_rejected(preset):
    with pytest.raises(LoRAConfigError, match="Unknown --lora-target-preset"):
        validate_lora_args(_args(["--use-lora", "--lora-target-preset", preset]))


@pytest.mark.unit
def test_hybrid_replicated_modules_require_explicit_opt_in():
    pattern = r"linear_attn\.in_proj_qkv$"
    config = LoRAConfig.from_args(
        _args(
            [
                "--use-lora",
                "--lora-target-preset",
                "hybrid_language",
                "--lora-allow-replicated-modules",
                pattern,
            ]
        )
    )
    assert config.allow_replicated_modules == (pattern,)


@pytest.mark.unit
@pytest.mark.parametrize("option", ["--save-lora", "--lora-save-merged-hf"])
@pytest.mark.parametrize("interval", [None, 0, -1])
def test_lora_export_requires_a_save_trigger(option, interval):
    args = _args(["--use-lora", option, "/tmp/export"], save="/tmp/full", save_interval=interval)
    with pytest.raises(LoRAConfigError, match="positive --save-interval"):
        validate_lora_checkpoint_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("option", ["--save-lora", "--lora-save-merged-hf"])
def test_lora_export_requires_full_checkpoint_destination(option):
    with pytest.raises(LoRAConfigError, match="requires --save"):
        validate_lora_checkpoint_args(_args(["--use-lora", option, "/tmp/export"], save_interval=1))


@pytest.mark.unit
@pytest.mark.parametrize("release_train, interval", [(False, 2), (True, None)])
def test_lora_export_accepts_periodic_or_release_train_saving(release_train, interval):
    validate_lora_checkpoint_args(
        _args(
            ["--use-lora", "--save-lora", "/tmp/adapter"],
            save="/tmp/full",
            save_interval=interval,
            release_train=release_train,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize("option", ["lora_alpha", "lora_learning_rate", "lora_weight_decay"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_lora_numeric_options_reject_nonfinite_and_negative_values(option, value):
    with pytest.raises(LoRAConfigError, match=option.replace("_", "-")):
        validate_lora_args(_args(["--use-lora"], **{option: value}))


@pytest.mark.unit
def test_lora_learning_rate_rejects_zero_but_weight_decay_accepts_it():
    with pytest.raises(LoRAConfigError, match="lora-learning-rate"):
        validate_lora_args(_args(["--use-lora"], lora_learning_rate=0.0))
    validate_lora_args(_args(["--use-lora"], lora_learning_rate=1e-5, lora_weight_decay=0.0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
