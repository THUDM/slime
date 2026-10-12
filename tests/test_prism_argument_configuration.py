"""Real slime CLI/configuration contracts; MCore parsing/validation are CPU boundaries.

The optional artifact test loads its actual AutoConfig and the published config.json,
without constructing a model, loading weights, or claiming CUDA/TE validation.
"""

import argparse
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from slime_plugins.models import prism

NUM_GPUS = 0
REPO = Path(__file__).resolve().parents[1]


def _slime_cli_provider():
    # Run the production argument registrations without importing the unrelated
    # SGLang server/runtime dependencies of slime.utils.arguments.
    source = REPO / "slime/utils/arguments.py"
    names = {"reset_arg", "get_slime_extra_args_provider"}
    functions = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"argparse": argparse, "json": json}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(source), "exec"), namespace
    )
    return namespace["get_slime_extra_args_provider"]()


def _load_backend(monkeypatch, argv, validate=None):
    """Stub only missing MCore entry points; keep real Prism and transformers."""
    training = ModuleType("megatron.training.arguments")
    tokenizer = ModuleType("megatron.training.tokenizer.tokenizer")

    def parse_args(*, extra_args_provider, ignore_unknown_args):
        assert ignore_unknown_args
        parser = argparse.ArgumentParser()
        for name in (
            "num-layers",
            "hidden-size",
            "ffn-hidden-size",
            "num-attention-heads",
            "kv-channels",
            "num-query-groups",
        ):
            parser.add_argument(f"--{name}", type=int)
        for name, default in (("seq-length", None), ("max-position-embeddings", None), ("vocab-size", None)):
            parser.add_argument(f"--{name}", type=int, default=default)
        parser.add_argument("--rotary-base", type=float, default=10000.0)
        parser.add_argument("--layernorm-epsilon", type=float, default=1e-5)
        parser.add_argument("--attention-dropout", type=float, default=0.1)
        parser.add_argument("--hidden-dropout", type=float, default=0.1)
        parser.add_argument("--fp16", action="store_true")
        parser.add_argument("--tokenizer-model")
        parser.add_argument("--tokenizer-type")
        parser.add_argument("--context-parallel-size", type=int, default=1)
        parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
        parser.set_defaults(
            position_embedding_type="learned_absolute",
            multi_latent_attention=False,
            rope_type=None,
            decoder_first_pipeline_num_layers=None,
            decoder_last_pipeline_num_layers=None,
        )
        parser = extra_args_provider(parser)
        args, _ = parser.parse_known_args(["--rollout-batch-size", "1", *argv])
        return args

    training.parse_args = parse_args
    training.validate_args = validate or (lambda args: args)
    tokenizer._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    for name in ("megatron", "megatron.training", "megatron.training.tokenizer"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, training.__name__, training)
    monkeypatch.setitem(sys.modules, tokenizer.__name__, tokenizer)
    path = REPO / "slime/backends/megatron_utils/arguments.py"
    spec = importlib.util.spec_from_file_location("prism_argument_configuration_backend", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_prism_config():
    return {
        "model_type": "moss_tts_prism",
        "dtype": "bfloat16",
        "n_vq": 3,
        "speech_vocab_size": 1024,
        "text_pad_idx": 151643,
        "audio_start_token_id": 151652,
        "prism_loop_block_ranges": [],
        "prism_loop_block_repetitions": [],
        "prism_rvq_heads_per_execution_unit": [1, 2],
        "prism_input_rvq_channels": [1, 2, 3],
        "embedding_head_usage": "untied",
        "language_config": {
            "model_type": "qwen3",
            "hidden_act": "silu",
            "attention_bias": False,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "num_hidden_layers": 2,
            "vocab_size": 151700,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        },
    }


def _write_config(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _prism_dispatch(monkeypatch, module):
    # Synthetic artifacts have no remote config class. The actual artifact test
    # below intentionally keeps AutoConfig.from_pretrained unpatched.
    monkeypatch.setattr(
        module,
        "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(model_type="moss_tts_prism")),
    )


def _assert_architecture(args, spec):
    assert args.num_layers == spec.topology.num_layers
    assert args.hidden_size == spec.hidden_size
    assert args.ffn_hidden_size == spec.ffn_hidden_size
    assert args.num_attention_heads == spec.num_attention_heads
    assert args.num_query_groups == spec.num_query_groups
    assert args.kv_channels == spec.kv_channels
    assert args.padded_vocab_size == spec.text_vocab_size
    assert args.max_position_embeddings == spec.max_position_embeddings
    assert args.n_vq == spec.topology.n_vq
    assert args.audio_start_token_id == spec.audio_start_token_id
    assert args.prism_mlp_only_layers == spec.topology.prism_mlp_only_layers
    assert args.prism_rvq_heads_per_execution_unit == spec.topology.prism_rvq_heads_per_execution_unit
    assert args.prism_input_rvq_channels == spec.input_rvq_channels
    assert args.rotary_base == spec.rotary_base
    assert args.layernorm_epsilon == spec.norm_epsilon
    assert args.position_embedding_type == "rope"
    assert args.rotary_percent == 1.0
    assert args.normalization == "RMSNorm"
    assert args.qk_layernorm and args.swiglu and args.group_query_attention
    assert args.attention_dropout == args.hidden_dropout == 0.0
    assert not args.add_bias_linear and not args.add_qkv_bias


@pytest.mark.parametrize(
    "stale_cli",
    [[], ["--num-layers", "99", "--hidden-size", "1", "--padded-vocab-size", "12", "--max-position-embeddings", "32"]],
)
def test_prism_configures_before_defaults_and_megatron_validation(monkeypatch, tmp_path, stale_cli):
    checkpoint = _write_config(tmp_path, _synthetic_prism_config())
    spec = prism.read_prism_spec(checkpoint)
    events = []

    def validate(args):
        events.append("validate")
        _assert_architecture(args, spec)
        assert args.bf16 and not args.fp16

    module = _load_backend(monkeypatch, ["--hf-checkpoint", checkpoint, *stale_cli], validate)
    _prism_dispatch(monkeypatch, module)
    set_defaults = module._set_default_megatron_args

    def defaults(args):
        events.append("defaults")
        _assert_architecture(args, spec)
        return set_defaults(args)

    monkeypatch.setattr(module, "_set_default_megatron_args", defaults)
    monkeypatch.setattr(
        module, "_hf_validate_args", lambda *_: pytest.fail("Prism must use its own nested artifact architecture")
    )
    args = module.megatron_parse_args(_slime_cli_provider())
    module.validate_args(args)
    assert events == ["defaults", "validate"]
    assert args.variable_seq_lengths
    assert args.tokenizer_model == checkpoint and args.tokenizer_type == "HuggingFaceTokenizer"


@pytest.mark.parametrize(
    "stale_cli",
    [[], ["--num-layers", "1", "--hidden-size", "64", "--attention-dropout", "0.2", "--hidden-dropout", "0.3"]],
)
def test_real_prism_artifact_configures_actual_cli(monkeypatch, stale_cli):
    configured = os.environ.get("SLIME_TEST_PRISM_HF_CHECKPOINT")
    checkpoint = Path(configured) if configured else REPO.parent / "artifacts/prism17b-v04/hf-iter-0100000"
    if not (checkpoint / "config.json").is_file():
        pytest.skip("set SLIME_TEST_PRISM_HF_CHECKPOINT to a published Prism HF artifact")
    spec = prism.read_prism_spec(checkpoint)
    module = _load_backend(
        monkeypatch,
        [
            "--hf-checkpoint",
            str(checkpoint),
            "--custom-model-provider-path",
            "slime_plugins.models.prism.model_provider",
            "--custom-megatron-log-prob-function-path",
            "slime_plugins.models.prism.extract_policy_log_probs",
            "--prism-pretrained-checkpoint",
            "native-run",
            "--prism-pretrained-step",
            "100000",
            "--actor-num-nodes",
            "2",
            "--actor-num-gpus-per-node",
            "4",
            *stale_cli,
        ],
        lambda args: _assert_architecture(args, spec),
    )
    # This is the real transformers loader, including the artifact's remote
    # configuration class. No model weights are read.
    hf_config = module.AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True)
    assert hf_config.model_type == "moss_tts_prism"
    args = module.megatron_parse_args(_slime_cli_provider())
    module.validate_args(args)
    _assert_architecture(args, spec)
    assert args.attention_dropout == args.hidden_dropout == 0.0
    assert args.custom_model_provider_path == "slime_plugins.models.prism.model_provider"
    assert args.custom_megatron_log_prob_function_path == "slime_plugins.models.prism.extract_policy_log_probs"
    assert args.prism_pretrained_checkpoint == "native-run" and args.prism_pretrained_step == 100000
    assert args.world_size == 8 and args.rank == 0
    assert args.max_position_embeddings == 40960 and args.seq_length == 4096
    assert args.bf16 and args.use_distributed_optimizer


@pytest.mark.parametrize("hidden_size", [16, 64])
def test_regular_hf_cli_retains_existing_validation(monkeypatch, tmp_path, hidden_size):
    config = _synthetic_prism_config()["language_config"]
    checkpoint = _write_config(tmp_path, config)
    module = _load_backend(
        monkeypatch,
        [
            "--hf-checkpoint",
            checkpoint,
            "--num-layers",
            "2",
            "--hidden-size",
            str(hidden_size),
            "--ffn-hidden-size",
            "32",
            "--num-attention-heads",
            "4",
            "--layernorm-epsilon",
            "1e-6",
            "--rotary-base",
            "1000000",
        ],
    )
    monkeypatch.setattr(
        prism, "configure_prism_args", lambda *_: pytest.fail("ordinary HF models must not be Prism-configured")
    )
    if hidden_size != 16:
        with pytest.raises(AssertionError, match="hidden_size"):
            module.megatron_parse_args(_slime_cli_provider())
        return
    args = module.megatron_parse_args(_slime_cli_provider())
    assert args.hidden_size == 16 and args.num_layers == 2
    assert args.attention_dropout == args.hidden_dropout == 0.1
    assert args.max_position_embeddings == 4096
    assert args.custom_model_provider_path is None
    assert not hasattr(args, "n_vq")


@pytest.mark.parametrize("checkpoint,skip", [("missing-checkpoint", True), (None, False)])
def test_skip_and_no_hf_paths_keep_generic_defaults(monkeypatch, checkpoint, skip):
    module = _load_backend(monkeypatch, ["--hf-checkpoint", checkpoint] if checkpoint else [])
    monkeypatch.setattr(
        module,
        "AutoConfig",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: pytest.fail("HF config must not be loaded")),
    )
    monkeypatch.setattr(prism, "configure_prism_args", lambda *_: pytest.fail("Prism configure must not be called"))
    args = module.megatron_parse_args(_slime_cli_provider(), skip_hf_validate=skip)
    assert args.seq_length == args.max_position_embeddings == 4096
    assert not hasattr(args, "n_vq")


def test_prism_keeps_existing_allgather_cp_guard(monkeypatch, tmp_path):
    checkpoint = _write_config(tmp_path, _synthetic_prism_config())
    module = _load_backend(
        monkeypatch, ["--hf-checkpoint", checkpoint, "--allgather-cp", "--context-parallel-size", "2"]
    )
    _prism_dispatch(monkeypatch, module)
    with pytest.raises(ValueError, match="only supported for DSA"):
        module.megatron_parse_args(_slime_cli_provider())


def test_prism_invalid_architecture_fails_before_megatron_defaults(monkeypatch, tmp_path):
    config = _synthetic_prism_config()
    config["language_config"]["num_hidden_layers"] = 3
    checkpoint = _write_config(tmp_path, config)
    module = _load_backend(monkeypatch, ["--hf-checkpoint", checkpoint])
    _prism_dispatch(monkeypatch, module)
    monkeypatch.setattr(
        module, "_set_default_megatron_args", lambda *_: pytest.fail("invalid Prism config must fail before defaults")
    )
    with pytest.raises(ValueError, match="physical topology"):
        module.megatron_parse_args(_slime_cli_provider())
