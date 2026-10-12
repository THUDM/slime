"""CPU probability/causality contracts; these do not claim TE or NCCL validation."""

import ast
import inspect
from dataclasses import dataclass
from textwrap import dedent
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slime_plugins.models import prism

NUM_GPUS = 0


def _topology(counts=(1, 2), ranges=(), repetitions=(), reapply=False):
    return prism.resolve_prism_topology(
        SimpleNamespace(
            n_vq=sum(counts),
            prism_loop_block_ranges=ranges,
            prism_loop_block_repetitions=repetitions,
            prism_rvq_heads_per_execution_unit=counts,
            prism_mlp_only_layers=(),
            reapply_rvq_conditioning_in_loop=reapply,
        )
    )


def _carrier(frames=1, natural_stop=True, padding=0):
    length = frames + 1
    codes = torch.tensor([[1, 0, 1]] * frames, dtype=torch.long).reshape(frames, 3)
    audio = torch.cat((torch.zeros(1, 3, dtype=torch.long), codes))
    successors = torch.cat((codes, torch.zeros(1, 3, dtype=torch.long)))
    edges = torch.arange(length) < frames
    stop_mask = edges.clone()
    stops = torch.zeros(length, dtype=torch.long)
    if natural_stop:
        stop_mask[-1], stops[-1] = True, 1
    return {
        "input_ids": torch.tensor([[151652, *([151643] * frames), *([0] * padding)]]),
        "audio_codes": audio,
        "item_kind": torch.tensor([0, *([1] * frames)]),
        "audio_role": torch.tensor([0, *([2] * frames)]),
        "successor_audio_codes": successors,
        "successor_audio_mask": edges,
        "target_history_retention_mask": None,
        "policy_code_mask": edges.unsqueeze(-1).expand(-1, 3),
        "policy_stop_mask": stop_mask,
        "policy_stop_actions": stops,
    }


def _prepare(carrier, boundaries=None, audio_start_token_id=151652):
    return prism.prepare_policy_inputs(
        **carrier,
        n_vq=3,
        speech_vocab_size=2,
        text_vocab_size=151700,
        text_pad_idx=151643,
        audio_start_token_id=audio_start_token_id,
        cu_seqlens=boundaries,
    )


@pytest.mark.parametrize("frames,natural", [(0, True), (0, False), (1, True), (1, False), (3, True), (3, False)])
def test_real_frame_and_terminal_action_rows(frames, natural):
    batch = _prepare(_carrier(frames, natural))
    assert batch["code_mask"].sum().item() == 3 * frames
    assert batch["stop_mask"].sum().item() == frames + natural
    assert batch["stop_actions"][0, -1].item() == int(natural)
    assert not batch["successor_mask"][0, -1]
    # The first new frame is scored at the TEXT/NONE audio_start input.
    assert batch["item_kind"][0, 0] == 0
    if frames:
        assert batch["code_mask"][0, 0].all()


def test_padding_and_reference_conditioning_are_independent_of_actions():
    carrier = _carrier(frames=1, padding=2)
    batch = _prepare(carrier, torch.tensor([0, 2, 4], dtype=torch.int32))
    assert batch["tokens"].shape == (1, 4, 4)
    assert not batch["code_mask"][0, 2:].any()
    assert batch["retention"][0, 2:].all()
    # A known reference edge still transports codes although no action exists.
    carrier["policy_code_mask"] = torch.zeros(2, 3, dtype=torch.bool)
    carrier["policy_stop_mask"] = torch.zeros(2, dtype=torch.bool)
    batch = _prepare(carrier)
    assert batch["successor_mask"][0, 0]
    assert batch["successor_codes"][0, 0].tolist() == [1, 0, 1]


def test_successor_edge_cannot_cross_packed_boundary():
    with pytest.raises(ValueError, match="cross a packed sequence"):
        _prepare(_carrier(), torch.tensor([0, 1, 2], dtype=torch.int32))


class _Embedding(nn.Module):
    def __init__(self, scale, text=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.text = text

    def forward(self, ids, _position=None):
        if self.text:
            values = ids.eq(151652).float()
        else:
            values = ids.float().masked_fill(ids.eq(2), 0)
        return (values * self.scale).transpose(0, 1).unsqueeze(-1)


class _Layer(nn.Module):
    def forward(self, hidden_states, context=None, **_kwargs):
        return hidden_states, context


class _Head(nn.Module):
    def forward(self, hidden):
        return torch.cat((torch.zeros_like(hidden), hidden), dim=-1), None


def _toy_model(monkeypatch):
    # Exercise the migrated model's complete carrier -> embedding -> site
    # execution -> scoring path with CPU projections, without faking TE support.
    model = prism.MossTTSPrismModel.__new__(prism.MossTTSPrismModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        n_vq=3,
        speech_vocab_size=2,
        speech_pad_index=2,
        text_pad_idx=151643,
        audio_start_token_id=151652,
        prism_input_rvq_channels=(1, 2, 3),
    )
    model.vocab_size = 151700
    model.share_embeddings_and_output_weights = False
    model.prism_topology = _topology()
    model.text_embedding = _Embedding(1, text=True)
    model.audio_embeddings = nn.ModuleList([_Embedding(2), _Embedding(5), _Embedding(7)])
    model.audio_lm_heads = nn.ModuleList([_Head(), _Head(), _Head()])
    model.prediction_norms = nn.ModuleList([nn.Identity(), nn.Identity()])
    model.stop_head = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.stop_head.weight.copy_(torch.tensor([[0.0], [1.0]]))
    model.decoder = nn.Module()
    model.decoder.layers = nn.ModuleList([_Layer(), _Layer()])
    model.rotary_pos_emb = lambda *_args, **_kwargs: None
    model.rotary_pos_emb.get_rotary_seq_len = lambda *_args: 2
    monkeypatch.setattr(
        prism.tensor_parallel,
        "vocab_parallel_cross_entropy",
        lambda logits, targets: -logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1),
        raising=False,
    )
    return model


def _forward(model, carrier):
    length = carrier["input_ids"].shape[1]
    return model(
        **carrier,
        position_ids=None,
        attention_mask=None,
        packed_seq_params=SimpleNamespace(qkv_format="thd", cu_seqlens_q=torch.tensor([0, length])),
    )


def test_co_site_heads_do_not_condition_on_each_other(monkeypatch):
    model = _toy_model(monkeypatch)
    carrier = _carrier()
    original = _forward(model, carrier)
    original[0, 0, 2].backward()
    assert model.audio_embeddings[0].scale.grad.abs() > 0
    assert model.audio_embeddings[1].scale.grad is None or model.audio_embeddings[1].scale.grad == 0
    # RVQ2 and RVQ3 share site 2. Changing RVQ2 must not change RVQ3's
    # conditional score at this row, while changing prior-site RVQ1 must.
    carrier["successor_audio_codes"][0, 1] = 1
    carrier["audio_codes"][1, 1] = 1
    same_site_changed = _forward(model, carrier)
    torch.testing.assert_close(original[0, 0, 2], same_site_changed[0, 0, 2])
    carrier["successor_audio_codes"][0, 0] = 0
    carrier["audio_codes"][1, 0] = 0
    prior_site_changed = _forward(model, carrier)
    assert not torch.isclose(original[0, 0, 2], prior_site_changed[0, 0, 2])
    # Analytical state is 1 at RVQ1 and 1+E1(1)=3 at both later heads.
    expected = torch.tensor(
        [
            -torch.nn.functional.softplus(torch.tensor(-1.0)),
            -torch.nn.functional.softplus(torch.tensor(3.0)),
            -torch.nn.functional.softplus(torch.tensor(-3.0)),
        ]
    )
    torch.testing.assert_close(original[0, 0, :3], expected)


def test_zero_frame_stop_scores_and_backpropagates(monkeypatch):
    model = _toy_model(monkeypatch)
    output = _forward(model, _carrier(frames=0))
    torch.testing.assert_close(output[0, 0, -1], -torch.nn.functional.softplus(torch.tensor(-1.0)))
    output[0, 0, -1].backward()
    assert model.stop_head.weight.grad.abs().sum() > 0


def test_loop_reuses_physical_layers_and_anchor_with_gradients():
    topology = _topology(counts=(1, 1), ranges=((0, 1),), repetitions=(2,), reapply=True)
    weight = torch.tensor(2.0, requires_grad=True)
    initial = torch.tensor([1.0])
    seen = []

    def apply(start, stop, value):
        seen.append((start, stop))
        return value * weight

    output = prism._execute_prism_schedule(
        topology, initial, apply_range=apply, read_site=lambda *_args: torch.tensor([3.0])
    )
    # first=2; second=(previous 2 + anchor 1 + delta 3)*2=12
    torch.testing.assert_close(output, torch.tensor([12.0]))
    output.sum().backward()
    torch.testing.assert_close(weight.grad, torch.tensor(8.0))
    assert seen == [(0, 1), (0, 1)]


def test_extractor_keeps_terminal_stop_separate_and_omits_tail_padding():
    output = torch.arange(4 * 4, dtype=torch.float32).reshape(1, 4, 4)
    mask = torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.bool)
    batch = {"total_lengths": [2, 1], "multimodal_train_inputs": {"policy_code_mask": mask}}
    result = prism.extract_policy_log_probs(SimpleNamespace(use_rollout_entropy=False), batch, output)
    assert [value.shape for value in result["policy_code_logprobs"]] == [(2, 3), (1, 3)]
    assert result["policy_stop_logprobs"][1].tolist() == [11]
    assert result["log_probs"][0].tolist() == [3]
    assert result["log_probs"][1].numel() == 0
    with pytest.raises(ValueError, match="entropy was requested"):
        prism.extract_policy_log_probs(SimpleNamespace(use_rollout_entropy=True), batch, output)


def test_entropy_and_selected_probability_match_categorical_gradient():
    logits = torch.tensor([[0.5, -0.7, 1.3]], requires_grad=True)
    selected, entropy = prism.selected_logprob_and_entropy(logits, torch.tensor([1]), return_entropy=True)
    distribution = torch.distributions.Categorical(logits=logits)
    torch.testing.assert_close(selected, distribution.log_prob(torch.tensor([1])))
    torch.testing.assert_close(entropy, distribution.entropy())
    actual = torch.autograd.grad((selected + entropy).sum(), logits, retain_graph=True)[0]
    expected = torch.autograd.grad((distribution.log_prob(torch.tensor([1])) + distribution.entropy()).sum(), logits)[
        0
    ]
    torch.testing.assert_close(actual, expected)


def test_entropy_columns_follow_the_selected_score_layout(monkeypatch):
    model = _toy_model(monkeypatch)
    monkeypatch.setattr(
        prism,
        "vocab_parallel_entropy",
        lambda logits: torch.distributions.Categorical(logits=logits).entropy(),
    )
    carrier = _carrier()
    carrier["policy_return_entropy"] = True
    output = _forward(model, carrier)
    assert output.shape == (1, 2, 8)
    batch = {
        "total_lengths": [2],
        "multimodal_train_inputs": {"policy_code_mask": carrier["policy_code_mask"]},
    }
    results = prism.extract_policy_log_probs(SimpleNamespace(use_rollout_entropy=True), batch, output)
    # Real finite continue/stop log probabilities remain outside the legacy
    # Omni frame summary, while the action arrays retain both queried stops.
    assert output[0, 0, 3] < 0
    assert output[0, 1, 3] < 0
    torch.testing.assert_close(results["log_probs"][0], output[0, :1, :3].sum(-1))
    assert not torch.isclose(results["log_probs"][0][0], output[0, 0, :4].sum())
    torch.testing.assert_close(results["policy_code_entropy"][0], output[0, :, 4:7])
    torch.testing.assert_close(results["policy_stop_entropy"][0], output[0, :, 7])
    assert results["entropy"][0].shape == (1,)


def _hf_config():
    return {
        "model_type": "moss_tts_prism",
        "dtype": "bfloat16",
        "n_vq": 3,
        "speech_vocab_size": 1024,
        "text_pad_idx": 151643,
        "audio_start_token_id": 151652,
        "prism_mlp_only_layers": [],
        "prism_loop_block_ranges": [],
        "prism_loop_block_repetitions": [],
        "prism_rvq_heads_per_execution_unit": [1, 2],
        "prism_input_rvq_channels": [1, 2, 3],
        "reapply_rvq_conditioning_in_loop": False,
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


def test_artifact_config_preserves_depth_and_full_rope_semantics():
    config = _hf_config()
    spec = prism.PrismModelSpec.from_hf_config(config)
    assert spec.topology.co_site_groups == ((1,), (2, 3))
    assert spec.input_rvq_channels == (1, 2, 3)
    config["language_config"]["num_hidden_layers"] = 3
    with pytest.raises(ValueError, match="physical topology"):
        prism.PrismModelSpec.from_hf_config(config)
    config["language_config"]["num_hidden_layers"] = 2
    config["language_config"]["rope_parameters"]["rope_type"] = "linear"
    with pytest.raises(ValueError, match="unscaled RoPE"):
        prism.PrismModelSpec.from_hf_config(config)


@pytest.mark.parametrize("frames", [0, 1])
def test_carrier_uses_the_configured_audio_start_for_stop_and_successor_validation(frames):
    carrier = _carrier(frames=frames)
    carrier["input_ids"][0, 0] = 151600
    prepared = _prepare(carrier, audio_start_token_id=151600)
    assert prepared["stop_mask"][0, 0]
    assert prepared["successor_mask"][0, 0].item() == bool(frames)


def test_artifact_audio_start_must_match_the_native_moss_v2_protocol():
    config = _hf_config()
    config["audio_start_token_id"] = 151600
    with pytest.raises(ValueError, match="requires audio_start_token_id"):
        prism.PrismModelSpec.from_hf_config(config)


def test_checkpoint_selection_accepts_run_without_inspecting_training_state(tmp_path):
    import json

    component = tmp_path / "checkpoints" / "iter_0000020"
    (component / "model").mkdir(parents=True)
    (component / "model" / ".metadata").write_bytes(b"metadata")
    (component / "checkpoint.json").write_text(json.dumps({"schema_version": 2, "iteration": 20}))
    (tmp_path / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("20")
    selected, manifest = prism.select_native_checkpoint(tmp_path)
    assert selected == component / "model"
    assert manifest["iteration"] == 20
    pinned, _ = prism.select_native_checkpoint(component, 20)
    assert pinned == component / "model"
    with pytest.raises(ValueError, match="explicit native checkpoint"):
        prism.select_native_checkpoint(component, 21)


@pytest.mark.parametrize("dtype,flag", [("bfloat16", "--bf16"), ("float16", "--fp16")])
def test_generated_megatron_cli_has_ordinary_architecture_flags_before_parse(tmp_path, dtype, flag):
    import argparse
    import json

    config = _hf_config()
    config["dtype"] = dtype
    config["embedding_head_usage"] = "tied_head_authoritative"
    (tmp_path / "config.json").write_text(json.dumps(config))
    tokens = prism.prism_megatron_cli_args(tmp_path)
    parser = argparse.ArgumentParser()
    integer_flags = (
        "num-layers",
        "hidden-size",
        "ffn-hidden-size",
        "num-attention-heads",
        "num-query-groups",
        "kv-channels",
        "vocab-size",
        "padded-vocab-size",
        "max-position-embeddings",
        "rotary-base",
    )
    for name in integer_flags:
        parser.add_argument(f"--{name}", type=int)
    for name in ("layernorm-epsilon", "rotary-percent", "attention-dropout", "hidden-dropout"):
        parser.add_argument(f"--{name}", type=float)
    for name in ("normalization", "position-embedding-type", "transformer-impl"):
        parser.add_argument(f"--{name}")
    for name in (
        "swiglu",
        "qk-layernorm",
        "group-query-attention",
        "disable-bias-linear",
        "untie-embeddings-and-output-weights",
        "bf16",
        "fp16",
    ):
        parser.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(tokens)
    assert (args.num_layers, args.hidden_size, args.ffn_hidden_size) == (2, 16, 32)
    assert (args.num_attention_heads, args.num_query_groups, args.kv_channels) == (4, 2, 4)
    assert args.vocab_size == args.padded_vocab_size == 151700
    assert args.max_position_embeddings == 40960
    assert args.rotary_base == 1000000
    assert args.layernorm_epsilon == 1e-6
    assert args.normalization == "RMSNorm" and args.position_embedding_type == "rope"
    assert args.transformer_impl == "transformer_engine"
    assert args.swiglu and args.qk_layernorm and args.group_query_attention and args.disable_bias_linear
    assert args.untie_embeddings_and_output_weights  # independent of audio's own tying policy
    assert args.attention_dropout == args.hidden_dropout == 0
    assert flag in tokens
    assert "--attention-backend" not in tokens
    assert not any(token.startswith("--prism") for token in tokens)


def test_generated_megatron_cli_rejects_unrepresentable_fractional_rope_base(tmp_path):
    import json

    config = _hf_config()
    config["language_config"]["rope_parameters"]["rope_theta"] = 1000000.5
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="integer Prism RoPE base"):
        prism.prism_megatron_cli_args(tmp_path)


def test_plugin_initializer_reads_generic_source_and_pin_without_rl_progress(monkeypatch, tmp_path):
    import sys
    import types

    component = tmp_path / "iter_0000020"
    component.mkdir()
    (component / "model").mkdir()
    (component / "model" / ".metadata").write_bytes(b"DCP")
    (component / "checkpoint.json").write_text('{"schema_version":2,"iteration":20,"model_identity":{}}')
    calls = []
    checkpointing = SimpleNamespace(
        load_common_state_dict=lambda path: {"iteration": 20, "checkpoint_version": 3.0},
        load_tensors_metadata=lambda path: {"weight": SimpleNamespace(dtype=torch.bfloat16)},
        load=lambda template, path: {"model": {"weight": torch.tensor([2.0])}},
    )
    core = types.ModuleType("megatron.core")
    core.dist_checkpointing = checkpointing
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setattr(prism, "_require_megatron", lambda: None)
    monkeypatch.setattr(prism, "_align_native_shards", lambda value, metadata, layers: value)
    model = SimpleNamespace(
        prism_spec=SimpleNamespace(
            dtype="bfloat16",
            topology=SimpleNamespace(num_layers=1),
            check_native_identity=lambda identity: calls.append("identity"),
        ),
        sharded_state_dict=lambda **kwargs: {"weight": object(), "kernel._extra_state": object()},
        load_state_dict=lambda values, strict: calls.append(("loaded", values["weight"].item()))
        or SimpleNamespace(missing_keys=["kernel._extra_state"], unexpected_keys=[]),
    )
    args = SimpleNamespace(pretrained_checkpoint=str(component), ckpt_step=20, iteration=777)
    assert prism.load_prism_pretrained([model], args) is None
    assert calls == ["identity", ("loaded", 2.0)]
    assert args.iteration == 777 and args.ckpt_step == 20


@pytest.mark.parametrize("backend", ["auto", "flash", "fused"])
def test_decoder_construction_retains_native_te_attention_for_every_backend(monkeypatch, backend):
    """Exercise the real constructor's spec boundary without a CUDA dependency."""

    class TEDotProductAttention:
        pass

    @dataclass
    class Spec:
        module: object
        submodules: object

    @dataclass
    class AttentionSubmodules:
        core_attention: object

    @dataclass
    class LayerSubmodules:
        pre_mlp_layernorm: object = None
        mlp: object = None
        mlp_bda: object = None
        sharded_state_dict_keys_map: object = None
        self_attention: object = None

    @dataclass
    class BlockSubmodules:
        layer_specs: list
        layer_norm: object

    core = TEDotProductAttention
    attention = Spec(object(), AttentionSubmodules(core))
    native_spec = Spec(object(), LayerSubmodules(object(), object(), object(), {"norm": "weight"}, attention))
    norm_impl = object()
    monkeypatch.setattr(prism, "TransformerLayerSubmodules", LayerSubmodules, raising=False)
    monkeypatch.setattr(prism, "TransformerBlockSubmodules", BlockSubmodules, raising=False)
    monkeypatch.setattr(prism, "LayerNormImpl", norm_impl, raising=False)
    # This enum boundary also exercises an older attention override if one
    # accidentally returns: auto/flash must preserve the incoming TE object.
    choices = SimpleNamespace(auto="auto", flash="flash", fused="fused")
    monkeypatch.setattr(prism, "AttnBackend", choices, raising=False)
    constructor = ast.parse(dedent(inspect.getsource(prism.MossTTSPrismModel.__init__)))
    boundary = next(
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "decoder" for target in node.targets)
    )
    namespace = {
        **vars(prism),
        "self": SimpleNamespace(prism_topology=SimpleNamespace(prism_mlp_only_layers=(1,))),
        "config": SimpleNamespace(num_layers=3, attention_backend=backend),
        "transformer_layer_spec": native_spec,
        "TransformerBlock": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    isolated = ast.Module(body=[boundary], type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), "<Prism decoder construction>", "exec"), namespace)
    block = namespace["self"].decoder.spec
    assert block.layer_norm is norm_impl
    assert block.layer_specs[0] is native_spec
    assert block.layer_specs[2] is native_spec
    assert block.layer_specs[0].submodules.self_attention.submodules.core_attention is core
    assert block.layer_specs[2].submodules.self_attention.submodules.core_attention is core
    assert block.layer_specs[1].module is prism.PrismMLPOnlyLayer
    assert block.layer_specs[1].submodules.self_attention is None
    assert block.layer_specs[1].submodules.mlp is native_spec.submodules.mlp
    assert native_spec.submodules.self_attention is attention
