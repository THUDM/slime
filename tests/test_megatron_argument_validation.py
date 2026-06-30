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
    logging_utils_mod = types.ModuleType("slime.utils.logging_utils")

    router_launch_mod.RouterArgs = object
    sglang_arguments_mod.sglang_parse_args = lambda *args, **kwargs: None
    sglang_arguments_mod.validate_args = lambda args: args
    sglang_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.arguments", sglang_arguments_mod)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", sglang_external_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", logging_utils_mod)

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
    args = types.SimpleNamespace(
        update_weight_transport="disk",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
    )

    with pytest.raises(ValueError, match="update-weight-disk-dir"):
        module._resolve_update_weight_disk_dir(args)


@pytest.mark.unit
def test_update_weight_disk_dir_normalizes_delta_alias(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_transport="disk",
        update_weight_disk_dir=None,
        update_weight_delta_dir="/shared/delta",
    )

    with pytest.warns(UserWarning, match="will be removed in a future release"):
        module._resolve_update_weight_disk_dir(args)

    assert args.update_weight_disk_dir == "/shared/delta"
    assert args.update_weight_delta_dir == "/shared/delta"


@pytest.mark.unit
def test_update_weight_disk_dir_backfills_legacy_delta_field(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/updates",
        update_weight_delta_dir=None,
    )

    module._resolve_update_weight_disk_dir(args)

    assert args.update_weight_disk_dir == "/shared/updates"
    assert args.update_weight_delta_dir == "/shared/updates"


@pytest.mark.unit
def test_update_weight_disk_dir_rejects_conflicting_alias(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/full",
        update_weight_delta_dir="/shared/delta",
    )

    with pytest.raises(ValueError, match="deprecated alias"):
        module._resolve_update_weight_disk_dir(args)


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
        megatron_to_hf_mode="raw",
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
        train_memory_margin_bytes=0,
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
        rollout_global_dataset=False,
        enable_mtp_training=False,
        mtp_num_layers=None,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        custom_config_path=None,
        eval_max_context_len=None,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        train_backend="megatron",
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        update_weight_mode="full",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


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
def test_slime_validate_args_rejects_mooncake_with_external_rollout_before_discovery(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    called = False

    def _apply_external_engine_info_to_args(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(module, "apply_external_engine_info_to_args", _apply_external_engine_info_to_args)
    args = make_slime_validate_args(
        update_weight_transport="mooncake",
        rollout_external_engine_addrs=["127.0.0.1:30000"],
    )

    with pytest.raises(ValueError, match="slime-managed SGLang"):
        module.slime_validate_args(args)

    assert called is False


@pytest.mark.unit
def test_update_weight_delta_rejects_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="delta",
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=True,
    )

    with pytest.raises(ValueError, match="not supported with --colocate"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_unknown_transport(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="delta",
        update_weight_transport="tensor",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
    )

    with pytest.raises(ValueError, match="supports only --update-weight-transport=nccl or disk"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_full_allows_mooncake_without_disk_dir(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
    )

    module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_multi_gpu_rollout_engine(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_num_gpus_per_engine=2,
    )

    with pytest.raises(ValueError, match="one GPU per rollout engine"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_sglang_dp(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=2,
    )

    with pytest.raises(ValueError, match="SGLang DP size 1"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_sglang_ep(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_ep_size=2,
    )

    with pytest.raises(ValueError, match="SGLang expert parallel size 1"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_sglang_dp_attention(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_enable_dp_attention=True,
    )

    with pytest.raises(ValueError, match="SGLang DP attention"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_megatron_pipeline_parallelism(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        pipeline_model_parallel_size=2,
    )

    with pytest.raises(ValueError, match="Megatron pipeline model parallel size 1"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_actor_megatron_config_pipeline_parallelism_override(monkeypatch, tmp_path):
    module = load_slime_arguments_module(monkeypatch)
    config_path = tmp_path / "megatron.yaml"
    config_path.write_text(
        """
megatron:
  - role: actor
    overrides:
      pipeline_model_parallel_size: 2
""",
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        pipeline_model_parallel_size=1,
    )

    with pytest.raises(ValueError, match="Megatron pipeline model parallel size 1"):
        module.parse_megatron_role_args(args, str(config_path), role="actor")


@pytest.mark.unit
def test_update_weight_mooncake_rejects_sglang_config_multi_gpu_group(monkeypatch, tmp_path):
    module = load_slime_arguments_module(monkeypatch)
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        """
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 2
        num_gpus_per_engine: 2
""",
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_config=str(config_path),
        hf_checkpoint="/actor",
    )

    with pytest.raises(ValueError, match="one GPU per rollout engine"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
@pytest.mark.parametrize(
    "override_yaml, message",
    [
        ("dp_size: 2", "SGLang DP size 1"),
        ("data-parallel-size: 2", "SGLang DP size 1"),
        ("tp-size: 2", "one GPU per rollout engine"),
        ("tensor_parallel_size: 2", "one GPU per rollout engine"),
        ("pp_size: 2", "one GPU per rollout engine"),
        ("pipeline-parallel-size: 2", "one GPU per rollout engine"),
        ("ep_size: 2", "SGLang expert parallel size 1"),
        ("expert-parallel-size: 2", "SGLang expert parallel size 1"),
        ("enable_dp_attention: true", "SGLang DP attention"),
    ],
)
def test_update_weight_mooncake_rejects_sglang_config_parallelism_overrides(
    monkeypatch, tmp_path, override_yaml, message
):
    module = load_slime_arguments_module(monkeypatch)
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        f"""
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 1
        overrides:
          {override_yaml}
""",
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_config=str(config_path),
        hf_checkpoint="/actor",
    )

    with pytest.raises(ValueError, match=message):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_allows_inferred_frozen_sglang_config_multi_gpu_group(monkeypatch, tmp_path):
    module = load_slime_arguments_module(monkeypatch)
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        """
sglang:
  - name: actor
    server_groups:
      - worker_type: regular
        num_gpus: 1
  - name: ref
    model_path: /ref
    server_groups:
      - worker_type: regular
        num_gpus: 2
        num_gpus_per_engine: 2
""",
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_config=str(config_path),
        hf_checkpoint="/actor",
    )

    module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_non_p2p_metadata_server(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="127.0.0.1:2379",
    )

    with pytest.raises(ValueError, match="metadata-server=P2PHANDSHAKE"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_mooncake_until_delta_carrier_exists(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="delta",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
    )

    with pytest.raises(ValueError, match="delta.*mooncake.*not supported"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_reserved_rpc_port_base(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        mooncake_rpc_port_base=18000,
    )

    with pytest.raises(ValueError, match="rpc-port-base.*reserved"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_reserved_buffer_count(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        mooncake_buffer_count=2,
    )

    with pytest.raises(ValueError, match="buffer-count.*reserved.*must be 1"):
        module._validate_update_weight_args(args)


@pytest.mark.unit
def test_update_weight_mooncake_rejects_external_rollout(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="mooncake",
        update_weight_disk_dir=None,
        update_weight_delta_dir=None,
        colocate=False,
        mooncake_metadata_server="P2PHANDSHAKE",
        rollout_external=False,
        rollout_external_engine_addrs=["127.0.0.1:30000"],
    )

    with pytest.raises(ValueError, match="slime-managed SGLang"):
        module._validate_update_weight_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
