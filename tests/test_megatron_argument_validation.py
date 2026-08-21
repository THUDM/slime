import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0


def load_arguments_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    training_mod = types.ModuleType("megatron.training")
    arguments_mod = types.ModuleType("megatron.training.arguments")
    tokenizer_pkg_mod = types.ModuleType("megatron.training.tokenizer")
    tokenizer_mod = types.ModuleType("megatron.training.tokenizer.tokenizer")
    transformers_mod = types.ModuleType("transformers")

    arguments_mod.parse_args = lambda *args, **kwargs: None
    arguments_mod.validate_args = lambda args: args
    tokenizer_mod._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    transformers_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", tokenizer_pkg_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "arguments.py"
    module_name = "test_megatron_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_slime_arguments_module(monkeypatch):
    router_pkg_mod = types.ModuleType("sglang_router")
    router_launch_mod = types.ModuleType("sglang_router.launch_router")
    sglang_arguments_mod = types.ModuleType("slime.backends.sglang_utils.arguments")
    sglang_external_mod = types.ModuleType("slime.backends.sglang_utils.external")
    logging_utils_mod = types.ModuleType("slime.observability.logging_utils")

    router_launch_mod.RouterArgs = object
    sglang_arguments_mod.sglang_parse_args = lambda *args, **kwargs: None
    sglang_arguments_mod.validate_args = lambda args: args
    sglang_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_arguments_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", sglang_external_mod)
    monkeypatch.setitem(sys.modules, "slime.observability.logging_utils", logging_utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "utils" / "arguments.py"
    module_name = "test_slime_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_qwen3_6_args(**overrides):
    values = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_layers=40,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_layer_freq=[1] * 40,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        layernorm_epsilon=1e-6,
        rotary_base=10000000,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_rs_refill_cli_parser_contract(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    module.get_slime_extra_args_provider()(parser)

    defaults = parser.parse_args(["--rollout-batch-size", "1"])
    enabled = parser.parse_args(["--rollout-batch-size", "1", "--rs-batch-refill", "--rs-refill-max-rounds", "7"])

    assert defaults.rs_batch_refill is False
    assert defaults.rs_refill_max_rounds == 2
    assert defaults.rs_refill_rpc_timeout_seconds == 1800.0
    assert defaults.rs_refill_max_candidate_cache_bytes == 1 << 30
    assert enabled.rs_batch_refill is True
    assert enabled.rs_refill_max_rounds == 7
    help_text = parser.format_help()
    assert "--rs-batch-refill" in help_text
    assert "--rs-refill-max-rounds" in help_text
    assert "--rs-refill-rpc-timeout-seconds" in help_text
    assert "--rs-refill-max-candidate-cache-bytes" in help_text


def make_qwen3_6_hf_config():
    text_config = types.SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000},
    )
    return types.SimpleNamespace(text_config=text_config)


def make_allgather_cp_args(**overrides):
    values = dict(
        allgather_cp=True,
        context_parallel_size=2,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_hf_validate_all_moe_skips_dense_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    module._hf_validate_args(make_qwen3_6_args(), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_moe_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    with pytest.raises(AssertionError, match="moe_intermediate_size"):
        module._hf_validate_args(make_qwen3_6_args(moe_ffn_hidden_size=256), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_dense_intermediate_size_when_moe_has_dense_layers(monkeypatch):
    module = load_arguments_module(monkeypatch)

    args = make_qwen3_6_args(moe_layer_freq=[0] + [1] * 39)

    with pytest.raises(AssertionError, match="intermediate_size"):
        module._hf_validate_args(args, make_qwen3_6_hf_config())


@pytest.mark.unit
def test_allgather_cp_rejects_non_dsa_cp_models(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args()
    hf_config = types.SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3")

    with pytest.raises(ValueError, match="only supported for DSA attention models"):
        module._validate_allgather_cp_supported(args, hf_config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "hf_config",
    [
        types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"], model_type="deepseek_v3"),
        types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], model_type="glm"),
    ],
)
def test_allgather_cp_allows_dsa_architectures(monkeypatch, hf_config):
    module = load_arguments_module(monkeypatch)

    module._validate_allgather_cp_supported(make_allgather_cp_args(), hf_config)


@pytest.mark.unit
def test_allgather_cp_ignores_cp_size_one(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args(context_parallel_size=1)

    module._validate_allgather_cp_supported(args)


@pytest.mark.unit
def test_update_weight_disk_dir_required_for_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(update_weight_transport="disk", update_weight_disk_dir=None)

    with pytest.raises(ValueError, match="update-weight-disk-dir"):
        module.slime_validate_args(args)


def make_slime_validate_args(**overrides):
    values = dict(
        eval_config=None,
        eval_prompt_data=None,
        kl_coef=0,
        use_kl_loss=False,
        ref_load=None,
        use_opd=False,
        opd_type=None,
        opd_teacher_load=None,
        load=None,
        hf_checkpoint="/tmp/hf",
        ref_ckpt_step=None,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        start_rollout_id=None,
        eval_interval=None,
        save_interval=None,
        save=None,
        kl_loss_coef=0,
        advantage_estimator="grpo",
        normalize_advantages=False,
        use_rollout_logprobs=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        rs_batch_refill=False,
        rs_refill_max_rounds=2,
        rs_refill_rpc_timeout_seconds=1800.0,
        rs_refill_max_candidate_cache_bytes=1 << 30,
        use_rs=False,
        tis_mode="truncate",
        tis_level="token",
        tis_lower_bound=0.1,
        tis_upper_bound=10.0,
        tis_batch_normalize=False,
        rs_level="token",
        rs_lower_bound=0.8,
        rs_upper_bound=1.2,
        rs_veto_threshold=None,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        log_probs_max_tokens_per_gpu=None,
        balance_by_flops=False,
        balance_data=False,
        eps_clip_high=None,
        eps_clip=0.2,
        eval_reward_key=None,
        reward_key="reward",
        dump_details=None,
        save_debug_rollout_data=None,
        save_debug_train_data=None,
        load_debug_rollout_data=None,
        rollout_external_engine_addrs=None,
        debug_train_only=False,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        offload=False,
        offload_train=None,
        offload_rollout=None,
        debug_rollout_only=False,
        colocate=False,
        rollout_num_gpus=8,
        eval_function_path=None,
        rollout_function_path="custom.rollout",
        rollout_top_p=1.0,
        rollout_top_k=-1,
        num_steps_per_rollout=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=None,
        grpo_std_normalization=True,
        over_sampling_batch_size=None,
        num_epoch=None,
        num_rollout=1,
        rollout_global_dataset=False,
        enable_mtp_training=False,
        mtp_num_layers=None,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        custom_config_path=None,
        megatron_config_path=None,
        custom_model_provider_path=None,
        custom_megatron_init_path=None,
        custom_megatron_before_log_prob_hook_path=None,
        custom_megatron_before_train_step_hook_path=None,
        custom_convert_samples_to_train_data_path=None,
        custom_reward_post_process_path=None,
        rollout_data_postprocess_path=None,
        custom_rollout_log_function_path=None,
        dynamic_sampling_filter_path=None,
        eval_max_context_len=None,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        train_backend="megatron",
        release_train=False,
        keep_old_actor=False,
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_local_checkpoint_dir=None,
        update_weight_mode="full",
        update_weights_interval=1,
        update_weight_start_version=0,
        ref_update_interval=None,
        rollout_temperature=1.0,
        context_parallel_size=1,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        fp8=None,
        fp8_param_gather=False,
        fp4=None,
        te_precision_config_file=None,
        kitchen_config_file=None,
        kitchen_recipe_number=None,
        megatron_deepgemm_forward_layers=None,
        megatron_deepgemm_forward_modules=None,
        megatron_deepgemm_moe_forward_layers=None,
        megatron_deepgemm_moe_forward_modules=None,
        moe_input_jitter_eps=None,
        moe_router_force_load_balancing=False,
        moe_expert_capacity_factor=None,
        moe_router_load_balancing_type="aux_loss",
        compute_advantages_and_returns=True,
        custom_advantage_function_path=None,
        loss_type="policy_loss",
        use_opsm=False,
        use_rollout_entropy=False,
        partial_rollout=False,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_rs_refill_args(**overrides):
    values = dict(
        rs_batch_refill=True,
        use_tis=True,
        use_rs=True,
        rs_level="geometric",
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        global_batch_size=4,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=1024,
    )
    values.update(overrides)
    return make_slime_validate_args(**values)


@pytest.mark.unit
def test_rs_refill_accepts_the_supported_exact_configuration(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)

    args = make_rs_refill_args()
    module.slime_validate_args(args)

    assert args.start_rollout_id == 0
    assert args.finetune is True


@pytest.mark.unit
def test_rs_refill_accepts_explicitly_disabled_fp_quantization(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)

    module.slime_validate_args(make_rs_refill_args(fp8=False, fp4=False))


@pytest.mark.unit
def test_rs_refill_accepts_a_read_only_custom_rollout_logger(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)

    module.slime_validate_args(make_rs_refill_args(custom_rollout_log_function_path="custom.log"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_backend": "fsdp"}, "train-backend megatron"),
        ({"colocate": True}, "disaggregated"),
        ({"rollout_global_dataset": False}, "rollout-global-dataset"),
        ({"update_weights_interval": 2}, "update-weights-interval 1"),
        ({"start_rollout_id": 3}, "starting at rollout 0"),
        ({"global_batch_size": 8}, "one optimizer step"),
        ({"custom_model_provider_path": "custom.model"}, "custom-model-provider-path"),
        ({"custom_megatron_init_path": "custom.init"}, "custom-megatron-init-path"),
        ({"update_weight_transport": "disk"}, "full weight updates over NCCL"),
        ({"rollout_top_k": 8}, "rollout-top-k -1"),
        ({"use_tis": False}, "requires --use-tis"),
        ({"custom_tis_function_path": "custom.tis"}, "built-in TIS/RS"),
        ({"use_rs": False}, "requires use_rs"),
        ({"rs_level": "token"}, "rs_level: sequence or geometric"),
        ({"tis_mode": "mask"}, "tis_mode: truncate or clip"),
        ({"tis_batch_normalize": True}, "tis_batch_normalize"),
        ({"attention_dropout": 0.1}, "attention-dropout 0"),
        ({"fp8": "hybrid"}, "non-FP8 Megatron training"),
        ({"fp8_param_gather": True}, "fp8-param-gather"),
        ({"fp4": "nvfp4"}, "non-FP4 Megatron training"),
        ({"te_precision_config_file": "/tmp/te.yaml"}, "Transformer Engine precision config"),
        ({"kitchen_config_file": "/tmp/kitchen.yaml"}, "Kitchen quantization"),
        ({"kitchen_recipe_number": 1}, "Kitchen quantization"),
        ({"megatron_deepgemm_forward_layers": [0]}, "DeepGEMM FP8 forward"),
        ({"megatron_deepgemm_forward_modules": ["linear_qkv"]}, "DeepGEMM FP8 forward"),
        ({"megatron_deepgemm_moe_forward_layers": [0]}, "DeepGEMM FP8 forward"),
        ({"megatron_deepgemm_moe_forward_modules": ["mlp.experts"]}, "DeepGEMM FP8 forward"),
        ({"moe_input_jitter_eps": 0.1}, "moe-input-jitter-eps"),
        ({"moe_router_force_load_balancing": True}, "moe-router-force-load-balancing"),
        ({"moe_expert_capacity_factor": 1.25}, "moe-expert-capacity-factor"),
        ({"moe_router_load_balancing_type": "sinkhorn"}, "Sinkhorn MoE routing"),
        ({"moe_router_load_balancing_type": ["aux_loss", "sinkhorn"]}, "Sinkhorn MoE routing"),
        ({"context_parallel_size": 2}, "context-parallel-size 1"),
        ({"partial_rollout": True}, "partial-rollout"),
        ({"dynamic_sampling_filter_path": "custom.filter"}, "dynamic-sampling-filter-path"),
        (
            {"rollout_function_path": "slime.rollout.fully_async_rollout.generate_rollout_fully_async"},
            "fully-async rollout queue",
        ),
        ({"rs_refill_max_rounds": -1}, "max-rounds"),
        ({"rs_refill_max_rounds": 1.5}, "max-rounds"),
        ({"rs_refill_max_rounds": False}, "max-rounds"),
        ({"rs_refill_rpc_timeout_seconds": 0}, "rpc-timeout"),
        ({"rs_refill_rpc_timeout_seconds": float("inf")}, "rpc-timeout"),
        ({"rs_refill_max_candidate_cache_bytes": 0}, "candidate-cache"),
        ({"rs_refill_max_candidate_cache_bytes": False}, "candidate-cache"),
        ({"rs_lower_bound": 2.0, "rs_upper_bound": 1.0}, "finite RS bounds"),
    ],
)
def test_rs_refill_rejects_unsupported_configurations(monkeypatch, overrides, message):
    module = load_slime_arguments_module(monkeypatch)

    with pytest.raises(ValueError, match=message):
        module.slime_validate_args(make_rs_refill_args(**overrides))


@pytest.mark.unit
def test_rs_refill_rejects_checkpoint_resume(monkeypatch, tmp_path):
    module = load_slime_arguments_module(monkeypatch)
    checkpoint = tmp_path / "actor"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("3", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint resume is not implemented"):
        module.slime_validate_args(make_rs_refill_args(load=str(checkpoint), start_rollout_id=4))


@pytest.mark.unit
@pytest.mark.parametrize("megatron_to_hf_mode", ["raw", "bridge"])
def test_slime_validate_args_preserves_explicit_start_rollout_id(monkeypatch, megatron_to_hf_mode):
    """``--start-rollout-id`` is only a fallback when the user did not set it.

    Both the bridge and the raw branch reset it when there is no resumable
    Megatron checkpoint, which is exactly the case an explicit value is for.
    """
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(start_rollout_id=100, megatron_to_hf_mode=megatron_to_hf_mode)

    module.slime_validate_args(args)

    assert args.start_rollout_id == 100


@pytest.mark.unit
@pytest.mark.parametrize("megatron_to_hf_mode", ["raw", "bridge"])
def test_slime_validate_args_defaults_start_rollout_id_to_zero(monkeypatch, megatron_to_hf_mode):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(start_rollout_id=None, megatron_to_hf_mode=megatron_to_hf_mode)

    module.slime_validate_args(args)

    assert args.start_rollout_id == 0


@pytest.mark.unit
def test_slime_validate_args_rejects_equal_debug_data_paths(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        save_debug_rollout_data="/tmp/debug_{rollout_id}.pt",
        save_debug_train_data="/tmp/debug_{rollout_id}.pt",
    )

    with pytest.raises(ValueError, match="--save-debug-train-data must not be equal"):
        module.slime_validate_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_slime_validate_args_rejects_non_positive_rollout_temperature(monkeypatch, temperature):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(rollout_temperature=temperature)

    with pytest.raises(ValueError, match="--rollout-temperature must be > 0"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=True, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_slime_validate_args_preserves_larger_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        colocate=True,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        rollout_num_gpus=12,
    )

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 12
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_slime_validate_args_preserves_zero_rollout_gpus_without_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=False, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.actor_num_gpus_per_node == 8
    assert args.actor_num_nodes == 1
    assert args.offload_train is False
    assert args.offload_rollout is False


@pytest.mark.unit
def test_update_weight_delta_requires_disk_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="nccl",
        update_weight_local_checkpoint_dir="/local/ckpt",
    )

    with pytest.raises(ValueError, match="requires --update-weight-transport=disk"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir="/local/ckpt",
        colocate=True,
    )

    with pytest.raises(ValueError, match="not supported with --colocate"):
        module.slime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_requires_local_checkpoint_dir(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir=None,
    )

    with pytest.raises(ValueError, match="requires --update-weight-local-checkpoint-dir"):
        module.slime_validate_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
