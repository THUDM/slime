"""CPU contracts for the real Omni trace/action-score wire schemas."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import importlib
import json
import math
import os
import sys
import time
import types
import wave

import pytest
import torch

from slime.utils.types import Sample
from slime_plugins import omni_rollout as plugin

NUM_GPUS = 0


def geometry():
    identity = dict(
        policy_family="moss_tts_prism",
        architecture="MOSS-TTS-Prism",
        config_sha256="student",
        modeling_sha256="model",
        processor_sha256="processor",
        prompt_protocol_sha256="prompt",
        n_vq=2,
        audio_vocab_size=8,
        sample_rate=24000,
        stop_semantics="pre_frame_threshold_v1",
    )
    return plugin.PrismConfig(2, 8, 32, 9, 0, 4, 24000, 128, (1,), identity)


def sample(index=7):
    return Sample(
        index=index,
        group_index=2,
        prompt="你好",
        metadata={
            "task_type": "Full VoiceClone",
            "show_language": False,
            "domain": "chinese",
            "references": [{"id": "audio1", "uri": "/shared/ref.wav"}],
            "transcript_target": "你好",
            "language": "zh",
            "reference_targets": {"timbre": "audio1"},
        },
    )


def output(frames=2, finish="stop"):
    config = geometry()
    codes = [[1, 2], [3, 4]][:frames]
    logprobs = [[-0.4, -0.5], [-0.6, -0.7]][:frames]
    stops = frames + int(finish == "stop")
    actions = [0] * frames + ([1] if finish == "stop" else [])
    probabilities = [0.2, 0.3][:frames] + ([0.9] if finish == "stop" else [])
    stop_logprobs = [math.log(p if action else 1 - p) for p, action in zip(probabilities, actions, strict=True)]
    trace = dict(
        version=2,
        model_family="moss_tts_prism",
        stages=["tts_engine"],
        request_id="request-1",
        model_identity=config.identity,
        admission_weight_version="step_1",
        logprob_semantics=plugin.LOGPROB_SEMANTICS,
        stop_semantics=plugin.STOP_SEMANTICS,
        sampling=dict(
            audio_temperature=1.0,
            audio_top_p=1.0,
            audio_top_k=-1,
            audio_repetition_penalty=1.0,
            stop_sampling=True,
            max_new_tokens=10,
        ),
        replay_inputs=dict(
            layout=plugin.LAYOUT,
            prompt_rows=[[3, 0, 0], [4, 0, 0]],
            item_kind=[0, 0],
            audio_role=[0, 0],
            successor_audio_mask=[False, False],
            successor_audio_codes=[[0, 0], [0, 0]],
            target_history_retention_mask=[[False, False]] * 2,
        ),
        action_streams=[
            dict(
                name="codes",
                stage="tts_engine",
                modality="audio",
                action_type="discrete",
                layout="time_depth_autoregressive",
                flatten_order="time_major",
                shape=[frames, 2],
                vocab_size=8,
                actions=codes,
                logprobs=logprobs,
                action_mask=[[1, 1]] * frames,
                deterministic_mask=None,
                channel_ids=[0, 1],
                channel_roles=["rvq_0", "rvq_1"],
            ),
            dict(
                name="stop_decisions",
                stage="tts_engine",
                modality="control",
                action_type="discrete",
                layout="pre_frame_autoregressive",
                flatten_order="time_major",
                shape=[stops],
                vocab_size=2,
                actions=actions,
                logprobs=stop_logprobs,
                action_mask=[1] * stops,
                deterministic_mask=None,
            ),
        ],
        non_action_outputs=[dict(name="stop_probabilities", stage="tts_engine", values=probabilities)],
        finish_reason=finish,
        total_action_count=frames * 2 + stops,
    )
    return dict(
        meta_info=dict(
            omni_rollout=trace,
            weight_version="step_1",
            finish_reason={"type": finish},
            prompt_tokens=2,
            completion_tokens=frames,
        ),
        audio=None,
    )


@pytest.mark.parametrize("frames,finish", [(2, "stop"), (2, "length"), (0, "stop")])
def test_trace_keeps_time_axis_and_exact_terminal_stop(frames, finish):
    item = sample()
    result = plugin.parse_rollout(output(frames, finish), item, geometry(), expected_weight_version="step_1")
    assert result is item and item.index == 7 and item.group_index == 2
    assert item.tokens == [3, 4] + [9] * frames
    assert item.response_length == frames
    assert item.status == (Sample.Status.COMPLETED if finish == "stop" else Sample.Status.TRUNCATED)
    assert len(item.rollout_log_probs) == frames
    mm = item.multimodal_train_inputs
    assert mm["audio_codes"].shape == (2 + frames, 2)
    assert mm["policy_code_mask"].sum().item() == frames * 2
    assert mm["policy_stop_mask"].sum().item() == frames + int(finish == "stop")
    assert mm["policy_stop_actions"][mm["policy_stop_mask"]].tolist() == [0] * frames + (
        [1] if finish == "stop" else []
    )
    assert mm["policy_code_mask"][-1].tolist() == [False, False]
    assert mm["policy_stop_mask"][-1].item() == (finish == "stop")
    if frames:
        assert mm["target_history_retention_mask"][-1].tolist() == [True, False]
        torch.testing.assert_close(mm["successor_audio_codes"][1 : 1 + frames], mm["audio_codes"][2:])
        assert mm["audio_role"][2:].tolist() == [plugin.TARGET] * frames
    item._validate_response_metadata_lengths()
    restored = Sample.from_dict(item.to_dict())
    assert restored.response_length == frames
    torch.testing.assert_close(restored.multimodal_train_inputs["policy_stop_mask"], mm["policy_stop_mask"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda response: response["meta_info"]["omni_rollout"].update(version=1),
        lambda response: response["meta_info"].update(weight_version="different"),
        lambda response: response["meta_info"]["omni_rollout"]["sampling"].update(audio_temperature=0.7),
        lambda response: response["meta_info"]["omni_rollout"]["action_streams"][0].update(channel_ids=[1, 0]),
        lambda response: response["meta_info"]["omni_rollout"]["action_streams"][0].update(
            logprobs=[[float("nan")] * 2] * 2
        ),
        lambda response: response["meta_info"]["omni_rollout"]["action_streams"][1].update(actions=[0, 1, 1]),
        lambda response: response["meta_info"]["omni_rollout"]["action_streams"][1].update(logprobs=[-5.0] * 3),
        lambda response: response["meta_info"]["omni_rollout"]["replay_inputs"].update(
            successor_audio_mask=[True, False]
        ),
        lambda response: response["meta_info"]["omni_rollout"]["replay_inputs"].update(audio_role=[0, 2]),
        lambda response: response["meta_info"]["omni_rollout"]["replay_inputs"].update(
            prompt_rows=[[3, 0, 0], [5, 0, 0]]
        ),
        lambda response: response["meta_info"].update(completion_tokens=3),
    ],
)
def test_trace_rejects_wrong_identity_distribution_carrier_and_scores(mutation):
    response = output()
    mutation(response)
    with pytest.raises(ValueError):
        plugin.parse_rollout(response, sample(), geometry())


def test_prompt_payload_uses_structured_business_fields_not_tokenization():
    item = sample()
    item.metadata["global_instruction"] = {"Gender": "female"}
    args = types.SimpleNamespace(context_parallel_size=1, omni_return_audio=True, omni_model_name="prism")
    payload = plugin.build_generate_payload(args, item, {"max_new_tokens": 12})
    assert "input_ids" not in payload
    assert payload["prompt"]["script"] == "你好"
    assert payload["prompt"]["task_type"] == "Full VoiceClone"
    assert payload["prompt"]["references"] == item.metadata["references"]
    assert json.loads(payload["prompt"]["global_instruction"]) == {"Gender": "female"}
    assert "transcript_target" not in payload["prompt"]
    assert payload["sampling_params"]["stop_sampling"] is True
    assert payload["return_omni_rollout"] and payload["return_audio"]
    with pytest.raises(ValueError, match="temperature"):
        plugin.build_generate_payload(args, item, {"max_new_tokens": 12, "temperature": 0.7})


@pytest.mark.parametrize(
    "params,expected", [({"seed": 0}, 0), ({"sampling_seed": 321}, 321), ({"seed": 321, "sampling_seed": 321}, 321)]
)
def test_seed_aliases_keep_the_public_nonnegative_integer(params, expected):
    args = types.SimpleNamespace()
    payload = plugin.build_generate_payload(args, sample(), {"max_new_tokens": 12, **params})
    assert payload["sampling_params"]["seed"] == expected
    assert type(payload["sampling_params"]["seed"]) is int
    assert "sampling_seed" not in payload["sampling_params"]


@pytest.mark.parametrize(
    "params",
    [
        {"seed": 2, "sampling_seed": 3},
        {"seed": True},
        {"sampling_seed": True},
        {"seed": -1},
        {"sampling_seed": -1},
        {"seed": 1.5},
        {"sampling_seed": 1.5},
    ],
)
def test_seed_aliases_reject_conflicting_or_invalid_values(params):
    with pytest.raises(ValueError):
        plugin.build_generate_payload(types.SimpleNamespace(), sample(), {"max_new_tokens": 12, **params})


def _native_rollout_for_seed_tests():
    try:
        return importlib.import_module("slime.rollout.sglang_rollout")
    except ModuleNotFoundError as error:
        pytest.skip(f"Original rollout integration needs optional dependency {error.name}")


def test_original_deterministic_group_forwards_sampling_seed_to_omni(monkeypatch):
    rollout = _native_rollout_for_seed_tests()
    args = types.SimpleNamespace(sglang_enable_deterministic_inference=True, group_rm=False)
    state = types.SimpleNamespace(aborted=False, group_sampling_seeds=[321, 322])
    monkeypatch.setattr(rollout, "GenerateState", lambda args: state)
    observed = []

    async def capture_generate(args, item, sampling_params, *, evaluation=False):
        observed.append(plugin.build_generate_payload(args, item, sampling_params)["sampling_params"]["seed"])
        return item

    monkeypatch.setattr(rollout, "generate_and_rm", capture_generate)
    group = [sample(0), sample(1)]
    result = asyncio.run(rollout.generate_and_rm_group(args, group, {"max_new_tokens": 12}))
    assert result == group and observed == [321, 322]


def test_original_deterministic_eval_forwards_sampling_seed_to_omni(monkeypatch):
    from slime.utils.eval_config import EvalDatasetConfig

    rollout = _native_rollout_for_seed_tests()
    args = types.SimpleNamespace(
        sglang_enable_deterministic_inference=True,
        group_rm=False,
        rollout_seed=321,
        hf_checkpoint="fixture",
        multimodal_keys=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        eval_min_new_tokens=None,
        eval_reward_key=None,
        reward_key=None,
    )
    config = EvalDatasetConfig(
        name="omni-seed",
        path="unused",
        n_samples_per_eval_prompt=2,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_response_len=12,
    )
    cache_key = config.cache_key + (args.hf_checkpoint, False, None, None)
    monkeypatch.setattr(rollout, "EVAL_PROMPT_DATASET", {cache_key: types.SimpleNamespace(samples=[sample()])})

    async def capture_generate(args, item, sampling_params, *, evaluation=False):
        assert evaluation
        item.metadata["forwarded_seed"] = plugin.build_generate_payload(args, item, sampling_params)[
            "sampling_params"
        ]["seed"]
        item.reward = 0.0
        item.status = Sample.Status.COMPLETED
        return item

    monkeypatch.setattr(rollout, "generate_and_rm", capture_generate)
    result = asyncio.run(rollout.eval_rollout_single_dataset(args, 0, config))
    assert [item.metadata["forwarded_seed"] for item in result[config.name]["samples"]] == [321, 322]


def teacher_binding():
    identity = copy.deepcopy(geometry().identity)
    return dict(endpoint="http://teacher", weight_sha256="a" * 64, model_identity=identity)


def teacher_output(request, binding):
    return dict(
        version=1,
        sample_id=request["sample_id"],
        input_sha256=plugin.teacher_request_hash(request),
        code_logprobs=[[-0.2, -0.3] for _ in request["codes"]],
        stop_probabilities=[0.25] * (len(request["codes"]) + int(request["finish_reason"] == "stop")),
        finish_reason=request["finish_reason"],
        temperature=1.0,
        weight_version="frozen_teacher",
        teacher_weight_sha256=binding["weight_sha256"],
        model_identity=binding["model_identity"],
        logprob_semantics=plugin.LOGPROB_SEMANTICS,
        stop_semantics="pre_frame_threshold_v1",
    )


def test_cross_config_teacher_requires_verified_replay_assets(monkeypatch):
    from dataclasses import replace

    hashes = {"tokenizer.json": "one", "tokenizer_config.json": "two", "special_tokens_map.json": "three"}
    student_config = replace(geometry(), tokenizer_sha256=hashes)
    item = plugin.parse_rollout(output(), sample(), student_config)
    request, binding = plugin.teacher_request(item), teacher_binding()
    binding["model_identity"]["config_sha256"] = "different-backbone"
    result = teacher_output(request, binding)
    with pytest.raises(ValueError, match="requires local checkpoint"):
        plugin.attach_teacher_scores(item, request, result, binding)
    binding["checkpoint"] = "teacher-fixture"
    teacher_config = replace(student_config, identity=binding["model_identity"])
    monkeypatch.setattr(plugin, "_config", lambda path: teacher_config)
    assert plugin.attach_teacher_scores(item, request, result, binding) is item
    monkeypatch.setattr(plugin, "_config", lambda path: replace(teacher_config, audio_start_token_id=5))
    with pytest.raises(ValueError, match="incompatible geometry"):
        plugin.attach_teacher_scores(item, request, result, binding)


@pytest.mark.parametrize("frames,finish", [(2, "stop"), (2, "length"), (0, "stop")])
def test_teacher_scores_exact_student_prefix_and_actual_stop_actions(frames, finish):
    item = plugin.parse_rollout(output(frames, finish), sample(), geometry())
    request, binding = plugin.teacher_request(item), teacher_binding()
    result = teacher_output(request, binding)
    assert request["temperature"] == 1.0 and type(request["temperature"]) is float
    assert request["prompt_rows"] == [[3, 0, 0], [4, 0, 0]]
    assert "stop_decisions" not in request
    plugin.attach_teacher_scores(item, request, result, binding)
    mm = item.multimodal_train_inputs
    expected = [math.log(0.75)] * frames + ([math.log(0.25)] if finish == "stop" else [])
    torch.testing.assert_close(mm["policy_teacher_stop_logprobs"][mm["policy_stop_mask"]], torch.tensor(expected))
    assert item.teacher_log_probs == pytest.approx([-0.5] * frames)
    assert not mm["policy_teacher_code_logprobs"].requires_grad


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_sha256", "wrong"),
        ("teacher_weight_sha256", "b" * 64),
        ("temperature", 0.7),
        ("sample_id", "wrong"),
        ("version", 2),
    ],
)
def test_teacher_rejects_stale_or_mismatched_scores(field, value):
    item = plugin.parse_rollout(output(), sample(), geometry())
    request, binding = plugin.teacher_request(item), teacher_binding()
    result = teacher_output(request, binding)
    result[field] = value
    with pytest.raises(ValueError, match="differs"):
        plugin.attach_teacher_scores(item, request, result, binding)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_teacher_invalid_probabilities_are_rejected_without_target_clipping(probability):
    item = plugin.parse_rollout(output(), sample(), geometry())
    request, binding = plugin.teacher_request(item), teacher_binding()
    result = teacher_output(request, binding)
    result["stop_probabilities"][0] = probability
    with pytest.raises(ValueError):
        plugin.attach_teacher_scores(item, request, result, binding)


@pytest.mark.parametrize(
    "position,probability,raises",
    [
        pytest.param(0, 0.0, False, id="continue-certain"),
        pytest.param(0, 1.0, True, id="continue-zero-probability"),
        pytest.param(1, 1.0, False, id="stop-certain"),
        pytest.param(1, 0.0, True, id="stop-zero-probability"),
    ],
)
def test_teacher_endpoint_probability_is_conditioned_on_the_sampled_action(position, probability, raises):
    item = plugin.parse_rollout(output(frames=1), sample(), geometry())
    request, binding = plugin.teacher_request(item), teacher_binding()
    result = teacher_output(request, binding)
    result["stop_probabilities"][position] = probability
    if raises:
        with pytest.raises(ValueError, match="zero probability.*stable logprobs"):
            plugin.attach_teacher_scores(item, request, result, binding)
    else:
        plugin.attach_teacher_scores(item, request, result, binding)
        mm = item.multimodal_train_inputs
        selected = mm["policy_teacher_stop_logprobs"][mm["policy_stop_mask"]]
        assert selected[position].item() == 0.0
        assert torch.isfinite(selected).all()


def test_real_hf_artifact_protocol_from_environment():
    """Repeat with PRISM_HF_TEST_ARTIFACT=<HF directory>; no weights are loaded."""
    checkpoint = os.environ.get("PRISM_HF_TEST_ARTIFACT")
    if not checkpoint:
        pytest.skip("Set PRISM_HF_TEST_ARTIFACT to inspect a real exported HF artifact")
    config = plugin.PrismConfig.from_path(checkpoint)
    response = output()
    trace = response["meta_info"]["omni_rollout"]
    trace["model_identity"] = config.identity
    q = config.n_vq
    replay = trace["replay_inputs"]
    replay["prompt_rows"] = [[3] + [0] * q, [config.audio_start_token_id] + [0] * q]
    replay["successor_audio_codes"] = [[0] * q for _ in range(2)]
    replay["target_history_retention_mask"] = [[False] * q for _ in range(2)]
    codes = trace["action_streams"][0]
    codes.update(
        shape=[2, q],
        vocab_size=config.speech_vocab_size,
        actions=[[index % config.speech_vocab_size for index in range(q)] for _ in range(2)],
        logprobs=[[-1.0] * q for _ in range(2)],
        action_mask=[[1] * q for _ in range(2)],
        channel_ids=list(range(q)),
        channel_roles=[f"rvq_{index}" for index in range(q)],
    )
    trace["total_action_count"] = 2 * q + 3
    item = plugin.parse_rollout(response, sample(), config)
    assert item.tokens == [3, config.audio_start_token_id, config.text_pad_idx, config.text_pad_idx]
    assert item.rollout_log_probs == [-float(q), -float(q)]
    assert item.multimodal_train_inputs["audio_codes"].shape == (4, q)
    assert item.multimodal_train_inputs["policy_code_mask"].sum().item() == 2 * q
    assert set(config.tokenizer_sha256) == {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}


def test_teacher_domain_hook_reuses_score_endpoint_and_rejects_unmapped_domain(monkeypatch):
    item = plugin.parse_rollout(output(), sample(), geometry())
    binding = teacher_binding()
    args = types.SimpleNamespace(
        omni_teacher_config=dict(models={"a": binding}, by_domain={"chinese": "a"}, default_teacher="a")
    )
    calls = []

    async def fake_post(url, payload, **kwargs):
        calls.append((url, kwargs))
        return {"version": 1, "results": [teacher_output(payload["samples"][0], binding)]}

    monkeypatch.setattr(plugin, "post", fake_post)
    assert asyncio.run(plugin.score_teacher(args, item)) is item
    assert calls[0][0] == "http://teacher/score_actions" and calls[0][1]["max_retries"] == 1
    assert item.metadata["omni_teacher_key"] == "a"
    item.metadata["domain"] = "unmapped"
    with pytest.raises(ValueError, match="Unmapped"):
        asyncio.run(plugin.score_teacher(args, item))
    assert len(calls) == 1


def named_teacher_binding():
    binding = teacher_binding()
    binding.pop("endpoint")
    binding["deployment"] = "teacher_a"
    return binding


def named_teacher_args(binding, configuration=None):
    return types.SimpleNamespace(
        omni_teacher_config={"models": {"a": binding}, "by_domain": {"chinese": "a"}},
        omni_model_name="student",
        sglang_model_routers={"teacher_a": ("127.0.0.1", 30001)},
        sglang_config=configuration,
        rollout_health_check_timeout=0.2,
    )


def worker(url="http://worker-a", *, health="healthy", routable=True, disabled=False):
    return dict(url=url, health_state=health, routable=routable, disabled=disabled, model=None)


def test_named_teacher_discovers_live_workers_then_scores_directly(monkeypatch):
    from slime.backends.sglang_utils.sglang_config import ModelConfig, SglangConfig

    binding = named_teacher_binding()
    configuration = SglangConfig(models=[ModelConfig(name="teacher_a", update_weights=False)])
    args = named_teacher_args(binding, configuration)
    gets, posts = [], []

    async def fake_get(url, **kwargs):
        gets.append(url)
        if url.endswith("/workers"):
            return {
                "workers": [
                    worker(),
                    worker("http://worker-b"),
                    worker("http://disabled", disabled=True),
                    worker("http://dead", health="dead"),
                ]
            }
        assert url in ("http://worker-a/health", "http://worker-b/health")
        return {"running": True}

    async def fake_post(url, payload, **kwargs):
        assert url in ("http://worker-a/score_actions", "http://worker-b/score_actions")
        posts.append(url)
        return {"version": 1, "results": [teacher_output(payload["samples"][0], binding)]}

    monkeypatch.setattr(plugin, "get", fake_get)
    monkeypatch.setattr(plugin, "post", fake_post)
    for _ in range(2):
        item = plugin.parse_rollout(output(), sample(), geometry())
        asyncio.run(plugin.score_teacher(args, item))
        assert item.metadata["omni_teacher_deployment"] == "teacher_a"
        assert item.metadata["omni_teacher_endpoint"] + "/score_actions" == posts[-1]
        assert "policy_teacher_stop_logprobs" in item.multimodal_train_inputs
    assert posts[0] == posts[1]  # Stable group/sample dispatch, no global mutable round-robin state.
    assert gets.count("http://127.0.0.1:30001/workers") == 2


@pytest.mark.parametrize(
    "binding",
    [
        {"endpoint": "http://teacher", "deployment": "teacher_a"},
        {},
        {"endpoint": ""},
    ],
)
def test_teacher_binding_requires_one_valid_endpoint_or_deployment(monkeypatch, binding):
    monkeypatch.setattr(plugin, "get", lambda *args, **kwargs: pytest.fail("Invalid binding made an HTTP request"))
    with pytest.raises(ValueError):
        asyncio.run(plugin._teacher_endpoint(named_teacher_args(binding), sample(), binding))


@pytest.mark.parametrize("role", ["student-name", "update-true", "inferred-student", "unconfigured", "missing-router"])
def test_named_teacher_rejects_student_or_unconfigured_models_before_http(monkeypatch, role):
    from slime.backends.sglang_utils.sglang_config import ModelConfig, SglangConfig

    binding = named_teacher_binding()
    configuration = SglangConfig(models=[ModelConfig(name="teacher_a", update_weights=False)])
    args = named_teacher_args(binding, configuration)
    if role == "student-name":
        args.omni_model_name = "teacher_a"
    elif role == "update-true":
        configuration.models[0].update_weights = True
    elif role == "inferred-student":
        configuration.models[0].update_weights = None
        args.hf_checkpoint = "student-checkpoint"
        args.rollout_num_gpus_per_engine = 1
    elif role == "unconfigured":
        configuration.models = []
    else:
        args.sglang_model_routers = {}
    monkeypatch.setattr(plugin, "get", lambda *args, **kwargs: pytest.fail("Invalid role made an HTTP request"))
    with pytest.raises(ValueError):
        asyncio.run(plugin._teacher_endpoint(args, sample(), binding))


def test_named_teacher_resolves_explicit_frozen_model_from_yaml(monkeypatch, tmp_path):
    path = tmp_path / "deployments.yaml"
    path.write_text("sglang:\n- name: teacher_a\n  model_path: frozen-checkpoint\n  update_weights: false\n")
    binding = named_teacher_binding()
    args = named_teacher_args(binding, str(path))

    async def fake_get(url, **kwargs):
        return {"workers": [worker()]} if url.endswith("/workers") else {"running": True}

    monkeypatch.setattr(plugin, "get", fake_get)
    assert asyncio.run(plugin._teacher_endpoint(args, sample(), binding)) == "http://worker-a"


@pytest.mark.parametrize(
    "workers",
    [[], [worker(health="unhealthy")], [worker(routable=False)], [worker(disabled=True)], [worker(routable=1)]],
)
def test_named_teacher_rejects_router_workers_that_are_not_healthy_and_routable(monkeypatch, workers):
    async def fake_get(url, **kwargs):
        assert url.endswith("/workers")
        return {"workers": workers}

    monkeypatch.setattr(plugin, "get", fake_get)
    binding = named_teacher_binding()
    with pytest.raises(ValueError, match="no healthy routable"):
        asyncio.run(plugin._teacher_endpoint(named_teacher_args(binding), sample(), binding))


def test_named_teacher_probes_registration_health_and_uses_another_live_replica(monkeypatch):
    probes = []

    async def fake_get(url, **kwargs):
        if url.endswith("/workers"):
            return {"workers": [worker(), worker("http://worker-b")]}
        probes.append(url)
        # Make the first selected registered worker stale, then accept the other.
        return {"running": len(probes) > 1}

    monkeypatch.setattr(plugin, "get", fake_get)
    binding = named_teacher_binding()
    selected = asyncio.run(plugin._teacher_endpoint(named_teacher_args(binding), sample(), binding))
    assert len(probes) == 2 and selected + "/health" == probes[-1]


@pytest.mark.parametrize("mismatch", ["identity", "checksum"])
def test_named_teacher_keeps_frozen_identity_and_weight_pins_strict(monkeypatch, mismatch):
    binding = named_teacher_binding()
    args = named_teacher_args(binding)

    async def fake_get(url, **kwargs):
        return {"workers": [worker()]} if url.endswith("/workers") else {"running": True}

    async def fake_post(url, payload, **kwargs):
        result = teacher_output(payload["samples"][0], binding)
        if mismatch == "identity":
            result["model_identity"] = {**result["model_identity"], "config_sha256": "wrong"}
        else:
            result["teacher_weight_sha256"] = "b" * 64
        return {"version": 1, "results": [result]}

    monkeypatch.setattr(plugin, "get", fake_get)
    monkeypatch.setattr(plugin, "post", fake_post)
    item = plugin.parse_rollout(output(), sample(), geometry())
    with pytest.raises(ValueError, match="identity, input hash, checksum"):
        asyncio.run(plugin.score_teacher(args, item))
    assert "policy_teacher_stop_logprobs" not in item.multimodal_train_inputs


def test_cancellation_drains_the_owned_request_before_returning(monkeypatch):
    state = []

    async def fake_post(*args, **kwargs):
        await asyncio.sleep(0.01)
        state.append("completed")
        return {}

    monkeypatch.setattr(plugin, "post", fake_post)

    async def exercise():
        args = types.SimpleNamespace(omni_request_timeout_s=1.0, omni_cancel_drain_timeout_s=0.1)
        task = asyncio.create_task(plugin._request(args, "http://router/generate", {}))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state == ["completed"]

    asyncio.run(exercise())


def test_cancellation_cleanup_does_not_wait_forever_for_remote_transport(monkeypatch):
    async def slow_post(*args, **kwargs):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # A remote transport may not stop when its local waiter cancels.
            await asyncio.sleep(10)

    monkeypatch.setattr(plugin, "post", slow_post)

    async def exercise():
        args = types.SimpleNamespace(omni_request_timeout_s=1.0, omni_cancel_drain_timeout_s=0.01)
        task = asyncio.create_task(plugin._request(args, "http://router/generate", {}))
        await asyncio.sleep(0)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert time.monotonic() - started < 0.5

    asyncio.run(exercise())


def test_generate_uses_router_and_preserves_sample_identity(monkeypatch, tmp_path):
    item = sample()
    response = output()
    blob = io.BytesIO()
    with wave.open(blob, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 32)
    response["audio"] = dict(data=base64.b64encode(blob.getvalue()).decode(), format="wav", sample_rate=24000)
    args = types.SimpleNamespace(hf_checkpoint="fixture", omni_audio_dir=str(tmp_path), omni_model_name="prism")
    calls = []

    async def fake_post(url, payload, **kwargs):
        calls.append((url, payload))
        return copy.deepcopy(response)

    monkeypatch.setattr(plugin, "post", fake_post)
    monkeypatch.setattr(plugin, "_config", lambda path: geometry())
    monkeypatch.setitem(
        sys.modules,
        "slime.rollout.sglang_rollout",
        types.SimpleNamespace(get_model_url=lambda args, model: f"http://python-router/{model}/generate"),
    )
    assert asyncio.run(plugin.generate(args, item, {"max_new_tokens": 12})) is item
    assert calls[0][0] == "http://python-router/prism/generate"
    assert item.index == 7 and item.group_index == 2
    assert item.metadata["omni_audio_path"].endswith(".wav")
    assert item.metadata["omni_prompt"]["transcript_target"] == "你好"
    assert plugin.generate.abort_mode == "request"
