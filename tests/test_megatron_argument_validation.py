import argparse
import importlib.util
import runpy
import shlex
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0


@pytest.mark.parametrize("mode", ["save", "async_save", "load"])
@pytest.mark.parametrize("optimizer", ["cpu", "gpu"])
def test_checkpoint_e2e_launch_has_valid_save_configuration(monkeypatch, tmp_path, mode, optimizer):
    from slime.utils import external_utils

    commands = []
    launcher = types.ModuleType("slime.utils.external_utils.command_utils")
    launcher.execute_train = lambda **kwargs: commands.append(kwargs["train_args"])
    launcher.get_default_wandb_args = lambda _: ""
    monkeypatch.setitem(sys.modules, launcher.__name__, launcher)
    monkeypatch.setattr(external_utils, "command_utils", launcher, raising=False)
    test = runpy.run_path(str(Path(__file__).with_name("test_qwen3_4B_ckpt.py")))
    directory = str(tmp_path / "checkpoint with spaces")
    test["execute"](mode, optimizer=optimizer, checkpoint_dir=directory)
    [command] = commands

    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--load")
    parser.add_argument("--ckpt-step", type=int)
    parser.add_argument("--async-save", action="store_true")
    args, _ = parser.parse_known_args(shlex.split(command))

    assert args.save == directory
    # Megatron requires a positive save interval whenever --save is set,
    # including when straw uses the directory to track a restored branch.
    assert args.save_interval is not None and args.save_interval > 0
    assert args.async_save == (mode == "async_save")
    if mode == "load":
        assert args.load == directory and args.ckpt_step == 1
    else:
        assert args.load is None


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
        ckpt_format="torch_dist",
        flush_cache_interval=1,
        rollout_data_transport="object-store",
        rollout_data_dir=None,
        rollout_queue_lease_seconds=300,
        rollout_queue_segment_mib=None,
        rollout_io_concurrency=4,
        use_distributed_post=False,
        data_source_path=None,
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
        num_steps_per_rollout=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=None,
        grpo_std_normalization=True,
        over_sampling_batch_size=None,
        num_epoch=None,
        num_rollout=1,
        enable_mtp_training=False,
        mtp_num_layers=None,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        custom_config_path=None,
        eval_max_context_len=None,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        train_backend="megatron",
        release_train=False,
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_local_checkpoint_dir=None,
        update_weight_mode="full",
        rollout_temperature=1.0,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_trainer_fault_tolerance_saves_reshardable_optimizer(monkeypatch, tmp_path):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        use_fault_tolerance=True,
        save_debug_rollout_data=str(tmp_path / "rollout_{rollout_id}.pt"),
        ckpt_format="torch_dist",
        ckpt_fully_parallel_save=False,
        dist_ckpt_optim_fully_reshardable=False,
    )
    module.slime_validate_args(args)
    assert args.ckpt_fully_parallel_save
    assert args.dist_ckpt_optim_fully_reshardable


@pytest.mark.parametrize(
    "options,match",
    [
        ({"no_save_optim": True}, "optimizer and RNG"),
        ({"no_save_rng": True}, "optimizer and RNG"),
        ({"ckpt_format": "torch"}, "torch_dist"),
        ({"save_debug_rollout_data": "one-file.pt"}, "unique"),
    ],
)
def test_trainer_fault_tolerance_rejects_unrecoverable_checkpoints(monkeypatch, tmp_path, options, match):
    module = load_slime_arguments_module(monkeypatch)
    values = dict(
        use_fault_tolerance=True,
        save_debug_rollout_data=str(tmp_path / "rollout_{rollout_id}.pt"),
        ckpt_format="torch_dist",
    )
    values.update(options)
    with pytest.raises(ValueError, match=match):
        module.slime_validate_args(make_slime_validate_args(**values))


@pytest.mark.parametrize(
    "rollout_path",
    [
        "slime.rollout.sglang_rollout.generate_rollout",
        "slime.rollout.fully_async_rollout.generate_rollout_fully_async",
        "custom.rollout",
    ],
)
def test_pipeline_rl_defaults_to_fully_async_with_separate_eval(monkeypatch, rollout_path):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(flush_cache_interval=0, rollout_function_path=rollout_path)
    module.slime_validate_args(args)
    expected = (
        "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
        if "sglang_rollout" in rollout_path
        else rollout_path
    )
    assert args.rollout_function_path == expected
    assert args.eval_function_path == "slime.rollout.sglang_rollout.generate_rollout"


@pytest.mark.parametrize(
    "options",
    [{"colocate": True}, {"offload_rollout": True}, {"debug_train_only": True}, {"debug_rollout_only": True}],
)
def test_pipeline_rl_rejects_incompatible_lifecycles(monkeypatch, options):
    module = load_slime_arguments_module(monkeypatch)
    with pytest.raises(ValueError, match="flush-cache-interval"):
        module.slime_validate_args(make_slime_validate_args(flush_cache_interval=0, **options))


def test_flush_cache_interval_defaults_to_existing_behavior(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    parser = module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    assert parser.parse_args(["--rollout-batch-size", "1"]).flush_cache_interval == 1
    assert parser.parse_args(["--rollout-batch-size", "1", "--flush-cache-interval", "0"]).flush_cache_interval == 0
    args = make_slime_validate_args(rollout_function_path="slime.rollout.sglang_rollout.generate_rollout")
    module.slime_validate_args(args)
    assert args.rollout_function_path == args.eval_function_path == "slime.rollout.sglang_rollout.generate_rollout"


@pytest.mark.parametrize("interval", [-1, -100])
def test_negative_flush_cache_interval_disables_training_flush(monkeypatch, interval):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        flush_cache_interval=interval, rollout_function_path="slime.rollout.sglang_rollout.generate_rollout"
    )
    module.slime_validate_args(args)
    assert args.rollout_function_path == "slime.rollout.fully_async_rollout.generate_rollout_fully_async"


def test_distributed_fully_async_is_opt_in(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    parser = module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    defaults = parser.parse_args(["--rollout-batch-size", "1"])
    assert defaults.rollout_function_path == "slime.rollout.sglang_rollout.generate_rollout"
    assert defaults.data_source_path is None
    assert defaults.rollout_data_transport == "object-store"
    assert defaults.rollout_data_dir is None
    assert not defaults.rollout_queue_online_gc
    path = "slime.data.queue_data_source.QueueDataSource"
    enabled = parser.parse_args(["--rollout-batch-size", "1", "--data-source-path", path])
    assert enabled.data_source_path == path


@pytest.mark.parametrize("transport", ["object-store", "nixl", "straw"])
def test_rollout_transport_selects_source_and_only_straw_needs_storage(monkeypatch, tmp_path, transport):
    module = load_slime_arguments_module(monkeypatch)
    parser = module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    parsed = parser.parse_args(["--rollout-batch-size", "1", "--rollout-data-transport", transport])
    args = make_slime_validate_args(rollout_data_transport=parsed.rollout_data_transport)
    if transport == "straw":
        with pytest.raises(ValueError, match="--rollout-data-dir or --save"):
            module.slime_validate_args(args)
        args.save = str(tmp_path)
    module.slime_validate_args(args)
    if transport == "straw":
        assert args.data_source_path == "slime.data.queue_data_source.QueueDataSource"
        assert args.rollout_data_dir == str(tmp_path / "rollout_data")
    else:
        assert args.data_source_path == "slime.data.data_source.RolloutDataSourceWithBuffer"
        assert args.rollout_data_dir is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("segment_mib", [None, 1, 1024, 0, -1])
def test_straw_pack_size_is_optional_but_explicit_values_must_be_positive(monkeypatch, tmp_path, segment_mib):
    module = load_slime_arguments_module(monkeypatch)
    parser = module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    options = ["--rollout-batch-size", "1"]
    if segment_mib is not None:
        options += ["--rollout-queue-segment-mib", str(segment_mib)]
    parsed = parser.parse_args(options)
    assert parsed.rollout_queue_segment_mib == segment_mib
    args = make_slime_validate_args(
        rollout_data_transport="straw",
        rollout_data_dir=str(tmp_path),
        rollout_queue_segment_mib=parsed.rollout_queue_segment_mib,
    )
    if segment_mib is not None and segment_mib <= 0:
        with pytest.raises(ValueError, match="rollout-queue-segment-mib.*positive"):
            module.slime_validate_args(args)
    else:
        module.slime_validate_args(args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout_queue_online_gc": True},
        {"data_source_path": "slime.data.queue_data_source.QueueDataSource"},
    ],
)
def test_queue_options_require_straw_transport(monkeypatch, overrides):
    module = load_slime_arguments_module(monkeypatch)
    with pytest.raises(ValueError, match="requires --rollout-data-transport straw"):
        module.slime_validate_args(make_slime_validate_args(**overrides))


def test_global_dataset_flag_is_removed(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    parser = module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1"])
    assert not hasattr(args, "rollout_global_dataset")
    with pytest.raises(SystemExit):
        parser.parse_args(["--rollout-batch-size", "1", "--disable-rollout-global-dataset"])
    # Epoch-based scheduling no longer depends on this attribute.
    module.slime_validate_args(make_slime_validate_args(num_epoch=2, num_rollout=None))


@pytest.mark.parametrize(
    "flag",
    [
        "--rollout-queue-resume",
        "--rollout-queue-fork",
        "--rollout-queue-max-pending",
        "--rollout-queue-max-inflight",
    ],
)
def test_removed_queue_flags_are_rejected(monkeypatch, flag):
    module = load_slime_arguments_module(monkeypatch)
    parser = module.get_slime_extra_args_provider()(argparse.ArgumentParser())
    defaults = parser.parse_args(["--rollout-batch-size", "1"])
    assert not hasattr(defaults, flag[2:].replace("-", "_"))
    with pytest.raises(SystemExit):
        parser.parse_args(["--rollout-batch-size", "1", flag])


@pytest.mark.unit
def test_slime_validate_args_preserves_explicit_start_rollout_id(monkeypatch):
    """``--start-rollout-id`` is only a fallback when the user did not set it."""
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(start_rollout_id=100)

    module.slime_validate_args(args)

    assert args.start_rollout_id == 100


@pytest.mark.unit
def test_slime_validate_args_defaults_start_rollout_id_to_zero(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(start_rollout_id=None)

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


@pytest.mark.unit
def test_force_fp8_ue8m0_scale_argument(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    module.get_slime_extra_args_provider()(parser)

    defaults = parser.parse_args(["--rollout-batch-size", "1"])
    configured = parser.parse_args(["--rollout-batch-size", "1", "--force-fp8-ue8m0-scale"])

    assert defaults.force_fp8_ue8m0_scale is False
    assert configured.force_fp8_ue8m0_scale is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
