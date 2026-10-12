"""Prism Omni rollout and frozen-teacher adapters for slime's existing Sample.

The transformer time axis stays one dimensional. Typed audio carriers and
selected actions are multimodal tensors; no RL objective lives in this module.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import math
import os
import uuid
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from slime.utils.http_utils import _wrap_ipv6, get, post
from slime.utils.types import Sample

# mossLite.models.MossTTSPrism.run_policy: PrismItemKind / PrismAudioRole.
TEXT, AUDIO = 0, 1
NONE, REFERENCE, TARGET = 0, 1, 2
LAYOUT = "moss_prism_typed_v1"
LOGPROB_SEMANTICS = "temperature_scaled_full_vocab_v1"
STOP_SEMANTICS = "pre_frame_bernoulli_v1"
REPLAY_FIELDS = (
    "layout",
    "prompt_rows",
    "item_kind",
    "audio_role",
    "successor_audio_mask",
    "successor_audio_codes",
    "target_history_retention_mask",
)


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _tensor(value, shape, dtype, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    flat = value
    if len(shape) == 2:
        if any(not isinstance(row, list) or len(row) != shape[1] for row in value):
            raise ValueError(f"{name} has invalid row widths")
        flat = [item for row in value for item in row]
    expected = bool if dtype == torch.bool else int if dtype == torch.long else (int, float)
    if any(type(item) not in (expected if isinstance(expected, tuple) else (expected,)) for item in flat):
        raise ValueError(f"{name} has invalid scalar types")
    result = torch.tensor(value, dtype=dtype)
    if shape[0] == 0 and value == []:
        result = result.reshape(shape)
    if tuple(result.shape) != tuple(shape):
        raise ValueError(f"{name} shape {tuple(result.shape)} != {shape}")
    if dtype.is_floating_point and not torch.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite scores")
    return result


@dataclass(frozen=True)
class PrismConfig:
    n_vq: int
    speech_vocab_size: int
    text_vocab_size: int
    text_pad_idx: int
    pad_token_id: int
    audio_start_token_id: int
    sample_rate: int
    context_length: int
    input_rvq_channels: tuple[int, ...]
    identity: dict[str, Any]
    tokenizer_sha256: dict[str, str] | None = None

    def replay_geometry(self):
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in vars(self).items()
            if name not in ("identity", "tokenizer_sha256")
        }

    @classmethod
    def from_path(cls, checkpoint):
        root = Path(checkpoint)
        raw = (root / "config.json").read_bytes()
        config = json.loads(raw)
        if config.get("model_type") != "moss_tts_prism":
            raise ValueError("Omni rollout requires a moss_tts_prism inference artifact")
        language = config["language_config"]
        depth = _integer(config["n_vq"], "n_vq", minimum=1)
        channels = config["prism_input_rvq_channels"]
        if (
            not isinstance(channels, list)
            or not channels
            or channels != sorted(set(channels))
            or any(type(channel) is not int or not 1 <= channel <= depth for channel in channels)
        ):
            raise ValueError("prism_input_rvq_channels must be a sorted one-based channel subset")
        identity = {
            "policy_family": "moss_tts_prism",
            "architecture": "MOSS-TTS-Prism",
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "n_vq": depth,
            "audio_vocab_size": config["speech_vocab_size"],
            "sample_rate": config["sampling_rate"],
            "stop_semantics": "pre_frame_threshold_v1",
        }
        for key, filename in (
            ("modeling_sha256", "modeling_moss_tts.py"),
            ("processor_sha256", "processing_moss_tts.py"),
            ("prompt_protocol_sha256", "prompt_protocol.py"),
        ):
            identity[key] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        return cls(
            depth,
            _integer(config["speech_vocab_size"], "speech_vocab_size", minimum=1),
            _integer(language["vocab_size"], "vocab_size", minimum=1),
            _integer(config["text_pad_idx"], "text_pad_idx"),
            _integer(config["pad_token_id"], "pad_token_id"),
            _integer(config["audio_start_token_id"], "audio_start_token_id"),
            _integer(config["sampling_rate"], "sampling_rate", minimum=1),
            _integer(language["max_position_embeddings"], "max_position_embeddings", minimum=1),
            tuple(channels),
            identity,
            {
                filename: hashlib.sha256((root / filename).read_bytes()).hexdigest()
                for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
            },
        )


@lru_cache(maxsize=8)
def _config(checkpoint):
    return PrismConfig.from_path(checkpoint)


def _prompt(sample):
    metadata = sample.metadata or {}
    source = metadata.get("omni_prompt", {})
    if not isinstance(source, dict):
        raise ValueError("omni_prompt metadata must be an object")
    source = {**metadata, **source}
    if isinstance(sample.prompt, dict):
        source = {**source, **sample.prompt}
    script = source.get("script", sample.prompt)
    if not isinstance(script, str) or not script.strip():
        raise ValueError("Prism script must be a nonempty string")
    task = source.get("task_type")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Prism task_type must be explicit prompt metadata")
    if type(source.get("show_language")) is not bool:
        raise ValueError("Prism show_language must be explicit prompt metadata")
    if source["show_language"] and not source.get("language"):
        raise ValueError("show_language requires a language")
    instruction = source.get("global_instruction")
    if isinstance(instruction, dict):
        instruction = json.dumps(instruction, ensure_ascii=False, separators=(",", ":"))
    if instruction is not None and not isinstance(instruction, str):
        raise ValueError("global_instruction must be a JSON object or string")
    references = source.get("references", [])
    if not isinstance(references, list) or any(
        not isinstance(reference, dict)
        or not reference.get("id")
        or not (reference.get("uri") or reference.get("codes_uri"))
        for reference in references
    ):
        raise ValueError("references require explicit id and uri/codes_uri")
    control = source.get("tokens_control", False)
    if type(control) is not bool:
        raise ValueError("tokens_control must be Boolean")
    count = source.get("global_tokens")
    if control:
        _integer(count, "global_tokens", minimum=1)
    elif count is not None:
        raise ValueError("global_tokens requires tokens_control")
    prompt = dict(
        script=script,
        task_type=task,
        show_language=source["show_language"],
        language=source.get("language") if source["show_language"] else None,
        global_instruction=instruction,
        references=copy.deepcopy(references),
        tokens_control=control,
        global_tokens=count,
    )
    return prompt, source


def build_generate_payload(args, sample, sampling_params):
    """Send the structured task to Omni; its processor owns prompt rendering."""
    if getattr(args, "partial_rollout", False) or sample.response_length:
        raise ValueError("Prism baseline supports complete single-turn requests only")
    if getattr(args, "context_parallel_size", 1) != 1:
        raise ValueError("Prism Omni baseline requires context_parallel_size=1")
    for name, default in (("temperature", 1.0), ("top_p", 1.0), ("top_k", -1), ("repetition_penalty", 1.0)):
        if sampling_params.get(name, default) != default:
            raise ValueError(f"Prism baseline requires {name}={default}")
    if sampling_params.get("min_p", 0.0) not in (None, 0.0):
        raise ValueError("Prism baseline requires min_p=0 for full sampling support")
    if sampling_params.get("stop_sampling", True) is not True:
        raise ValueError("Prism baseline requires sampled stop")
    budget = _integer(sampling_params.get("max_new_tokens"), "max_new_tokens", minimum=1)
    prompt, _ = _prompt(sample)
    params = dict(
        temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, max_new_tokens=budget, stop_sampling=True
    )
    # slime group/eval workers use sampling_seed; Omni's public wire uses seed.
    seed = sampling_params.get("seed")
    sampling_seed = sampling_params.get("sampling_seed")
    if seed is not None:
        seed = _integer(seed, "seed")
    if sampling_seed is not None:
        sampling_seed = _integer(sampling_seed, "sampling_seed")
    if seed is not None and sampling_seed is not None and seed != sampling_seed:
        raise ValueError("seed and sampling_seed must agree when both are specified")
    if seed is not None or sampling_seed is not None:
        params["seed"] = seed if seed is not None else sampling_seed
    model = getattr(args, "omni_model_name", "prism")
    return dict(
        model=model,
        prompt=prompt,
        sampling_params=params,
        stream=False,
        return_logprob=True,
        return_omni_rollout=True,
        return_audio=getattr(args, "omni_return_audio", True),
        response_format="wav",
        metadata={"sample_index": sample.index, "group_index": sample.group_index},
    )


def _validate_prompt_carrier(replay, config):
    if not isinstance(replay, dict) or replay.get("layout") != LAYOUT:
        raise ValueError("Prism trace requires the typed replay layout")
    rows = replay.get("prompt_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prism prompt_rows must be nonempty")
    p, q = len(rows), config.n_vq
    carrier = _tensor(rows, (p, q + 1), torch.long, "prompt_rows")
    kind = _tensor(replay.get("item_kind"), (p,), torch.long, "item_kind")
    role = _tensor(replay.get("audio_role"), (p,), torch.long, "audio_role")
    edge = _tensor(replay.get("successor_audio_mask"), (p,), torch.bool, "successor_audio_mask")
    successor = _tensor(replay.get("successor_audio_codes"), (p, q), torch.long, "successor_audio_codes")
    retention = _tensor(
        replay.get("target_history_retention_mask"), (p, q), torch.bool, "target_history_retention_mask"
    )
    if ((kind != TEXT) & (kind != AUDIO)).any() or not torch.equal(kind, role):
        raise ValueError("Prism prompt may contain text and reference audio only")
    if ((carrier[:, 0] < 0) | (carrier[:, 0] >= config.text_vocab_size)).any():
        raise ValueError("Prism text token is outside its vocabulary")
    if ((carrier[:, 1:] < 0) | (carrier[:, 1:] >= config.speech_vocab_size)).any():
        raise ValueError("Prism reference codes are outside their vocabulary")
    if carrier[kind == TEXT, 1:].any() or (carrier[kind == AUDIO, 0] != config.text_pad_idx).any():
        raise ValueError("Prism typed input disagrees with its item kind")
    expected_edge = torch.cat((kind[1:] == AUDIO, torch.tensor([False])))
    expected_successor = torch.zeros_like(successor)
    indices = expected_edge.nonzero().flatten()
    expected_successor[indices] = carrier[indices + 1, 1:]
    if not torch.equal(edge, expected_edge) or not torch.equal(successor, expected_successor):
        raise ValueError("Reference successor edges differ from the exact prompt carrier")
    if (carrier[(kind == TEXT) & expected_edge, 0] != config.audio_start_token_id).any():
        raise ValueError("Reference audio must follow its audio-start anchor")
    if (
        carrier[-1, 0] != config.audio_start_token_id
        or carrier[-1, 1:].any()
        or kind[-1] != TEXT
        or role[-1] != NONE
        or edge[-1]
    ):
        raise ValueError("Prism prompt must end at the target audio-start anchor")
    return carrier, kind, role, edge, successor, retention


def parse_rollout(response, sample, config, *, max_frames=None, expected_weight_version=None):
    """Populate the existing Sample without treating RVQ channels as time steps."""
    meta = response.get("meta_info", {})
    trace = meta.get("omni_rollout")
    if (
        not isinstance(trace, dict)
        or type(trace.get("version")) is not int
        or trace["version"] != 2
        or trace.get("model_family") != "moss_tts_prism"
        or trace.get("stages") != ["tts_engine"]
    ):
        raise ValueError("Prism requires a schema-v2 tts_engine trace")
    version = meta.get("weight_version")
    if (
        not isinstance(version, str)
        or not version
        or trace.get("admission_weight_version") != version
        or expected_weight_version is not None
        and version != expected_weight_version
    ):
        raise ValueError("Prism rollout crossed or mismatched a policy version")
    if trace.get("model_identity") != config.identity:
        raise ValueError("Prism rollout model identity differs from the inference artifact")
    if trace.get("logprob_semantics") != LOGPROB_SEMANTICS or trace.get("stop_semantics") != STOP_SEMANTICS:
        raise ValueError("Prism rollout probability or stopping semantics differ")
    sampling = trace.get("sampling", {})
    required = dict(
        audio_temperature=1.0, audio_top_p=1.0, audio_top_k=-1, audio_repetition_penalty=1.0, stop_sampling=True
    )
    if any(sampling.get(key) != value for key, value in required.items()):
        raise ValueError("Prism trace must use full-support temperature-one sampling and sampled stop")
    streams = trace.get("action_streams")
    if not isinstance(streams, list) or [stream.get("name") for stream in streams] != ["codes", "stop_decisions"]:
        raise ValueError("Prism trace must contain RVQ and sampled-stop streams")
    code_stream, stop_stream = streams
    actions = code_stream.get("actions")
    if not isinstance(actions, list):
        raise ValueError("Prism codes must be an array")
    f, q = len(actions), config.n_vq
    finish = trace.get("finish_reason")
    if finish not in ("stop", "length") or f == 0 and finish != "stop":
        raise ValueError("Prism requires natural stop or a nonempty length truncation")
    if meta.get("finish_reason", {}).get("type") != finish:
        raise ValueError("Prism finish reason differs between trace and response")
    if max_frames is not None and f > max_frames:
        raise ValueError("Prism generated more frames than requested")
    for stream, layout, modality, shape, vocabulary in (
        (code_stream, "time_depth_autoregressive", "audio", [f, q], config.speech_vocab_size),
        (stop_stream, "pre_frame_autoregressive", "control", [f + int(finish == "stop")], 2),
    ):
        if (
            stream.get("stage") != "tts_engine"
            or stream.get("modality") != modality
            or stream.get("action_type") != "discrete"
            or stream.get("layout") != layout
            or stream.get("flatten_order") != "time_major"
            or stream.get("shape") != shape
            or stream.get("vocab_size") != vocabulary
            or stream.get("deterministic_mask") is not None
        ):
            raise ValueError("Prism action stream geometry differs")
    if (
        code_stream.get("channel_ids") != list(range(q))
        or code_stream.get("channel_roles") != [f"rvq_{index}" for index in range(q)]
        or code_stream.get("action_mask") != [[1] * q for _ in range(f)]
    ):
        raise ValueError("Prism RVQ action mask or channel order differs")
    codes = _tensor(actions, (f, q), torch.long, "codes")
    code_scores = _tensor(code_stream.get("logprobs"), (f, q), torch.float32, "code logprobs")
    stops = f + int(finish == "stop")
    stop_actions = _tensor(stop_stream.get("actions"), (stops,), torch.long, "stop decisions")
    stop_scores = _tensor(stop_stream.get("logprobs"), (stops,), torch.float32, "stop logprobs")
    if (
        stop_stream.get("action_mask") != [1] * stops
        or stop_actions.tolist() != [0] * f + ([1] if finish == "stop" else [])
        or ((codes < 0) | (codes >= config.speech_vocab_size)).any()
        or (code_scores > 1e-5).any()
        or (stop_scores > 1e-5).any()
    ):
        raise ValueError("Prism actions or selected behavior scores are invalid")
    observations = trace.get("non_action_outputs")
    if (
        not isinstance(observations, list)
        or len(observations) != 1
        or observations[0].get("name") != "stop_probabilities"
        or observations[0].get("stage") != "tts_engine"
    ):
        raise ValueError("Prism requires its stop-probability observation")
    probability = _tensor(observations[0].get("values"), (stops,), torch.float32, "stop probabilities")
    if ((probability < 0) | (probability > 1)).any() or not torch.allclose(
        stop_scores.exp(), torch.where(stop_actions.bool(), probability, 1 - probability), atol=2e-5, rtol=2e-4
    ):
        raise ValueError("Prism sampled-stop scores disagree with the observed distribution")
    if trace.get("total_action_count") != f * q + stops:
        raise ValueError("Prism action count differs")
    carrier, kind, role, edge, successor, retention = _validate_prompt_carrier(trace.get("replay_inputs"), config)
    p, t = len(carrier), len(carrier) + f
    if t > config.context_length:
        raise ValueError("Prism replay exceeds its context window")
    if meta.get("prompt_tokens") != p or meta.get("completion_tokens") != f:
        raise ValueError("Prism usage must count typed prompt rows and completed frames")
    if not isinstance(trace.get("request_id"), str) or not trace["request_id"]:
        raise ValueError("Prism trace request_id is missing")
    tail = torch.cat((torch.full((f, 1), config.text_pad_idx, dtype=torch.long), codes), dim=1)
    carrier = torch.cat((carrier, tail))
    kind = torch.cat((kind, torch.full((f,), AUDIO, dtype=torch.long)))
    role = torch.cat((role, torch.full((f,), TARGET, dtype=torch.long)))
    edge = torch.cat((edge, torch.zeros(f, dtype=torch.bool)))
    successor = torch.cat((successor, torch.zeros((f, q), dtype=torch.long)))
    target_retention = torch.zeros((f, q), dtype=torch.bool)
    target_retention[:, torch.tensor(config.input_rvq_channels) - 1] = True
    retention = torch.cat((retention, target_retention))
    code_positions = torch.arange(f) + p - 1
    stop_positions = torch.arange(stops) + p - 1
    edge[code_positions] = True
    successor[code_positions] = codes
    code_mask = torch.zeros((t, q), dtype=torch.bool)
    stop_mask = torch.zeros(t, dtype=torch.bool)
    code_mask[code_positions] = True
    stop_mask[stop_positions] = True
    aligned_stops = torch.zeros(t, dtype=torch.long)
    aligned_stops[stop_positions] = stop_actions
    aligned_code_scores = torch.zeros((t, q), dtype=torch.float32)
    aligned_stop_scores = torch.zeros(t, dtype=torch.float32)
    aligned_code_scores[code_positions] = code_scores
    aligned_stop_scores[stop_positions] = stop_scores
    sample.tokens = carrier[:, 0].tolist()
    sample.response_length = f
    sample.response = ""  # Waveform and transcript target are separate artifacts.
    sample.loss_mask = [1] * f
    sample.rollout_log_probs = code_scores.sum(-1).tolist()
    sample.weight_versions = [version]
    sample.status = Sample.Status.COMPLETED if finish == "stop" else Sample.Status.TRUNCATED
    sample.multimodal_train_inputs = {
        "audio_codes": carrier[:, 1:],
        "item_kind": kind,
        "audio_role": role,
        "successor_audio_mask": edge,
        "successor_audio_codes": successor,
        "target_history_retention_mask": retention,
        "policy_code_mask": code_mask,
        "policy_stop_mask": stop_mask,
        "policy_stop_actions": aligned_stops,
        "policy_rollout_code_logprobs": aligned_code_scores,
        "policy_rollout_stop_logprobs": aligned_stop_scores,
        "policy_group_advantage": torch.zeros(1, dtype=torch.float32),
    }
    prompt, source = _prompt(sample)
    sample.metadata = dict(sample.metadata or {})
    sample.metadata.update(
        omni_rollout=copy.deepcopy(trace),
        omni_frames=f,
        omni_prompt_rows=p,
        omni_finish_reason=finish,
        omni_policy_version=version,
        policy_version=version,
        omni_replay_geometry=config.replay_geometry(),
        omni_tokenizer_sha256=config.tokenizer_sha256,
        omni_prompt={
            **prompt,
            **{key: source[key] for key in ("transcript_target", "language", "reference_targets") if key in source},
        },
    )
    return sample


async def _request(args, url, payload, *, headers=None):
    """Drain a locally owned HTTP task on cancellation, within a fixed bound.

    Draining the client is not a claim that the server stopped its GPU request.
    Generation/teacher POST is not known to be idempotent, so use one attempt.
    """
    timeout = float(getattr(args, "omni_request_timeout_s", 600.0))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("omni_request_timeout_s must be finite and positive")
    drain = float(getattr(args, "omni_cancel_drain_timeout_s", min(timeout, 30.0)))
    if not math.isfinite(drain) or drain <= 0:
        raise ValueError("omni_cancel_drain_timeout_s must be finite and positive")
    task = asyncio.create_task(post(url, payload, max_retries=1, headers=headers, timeout=timeout))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=drain)
        except Exception:
            task.cancel()
            # asyncio.wait does not wait indefinitely for a transport/remote
            # actor that ignores cancellation. Consume its eventual exception.
            await asyncio.wait({task}, timeout=min(drain, 1.0))
            task.add_done_callback(lambda finished: None if finished.cancelled() else finished.exception())
        raise


def _write_audio(response, args, sample, config, required):
    audio = response.get("audio")
    if sample.response_length == 0:
        if audio is not None:
            raise ValueError("A zero-frame sampled stop must not return waveform")
        sample.metadata["omni_audio_path"] = None
        return
    if not required:
        if audio is not None:
            raise ValueError("An audio-free request unexpectedly returned waveform")
        sample.metadata["omni_audio_path"] = None
        return
    root = getattr(args, "omni_audio_dir", None)
    if not root:
        raise ValueError("omni_audio_dir is required when returning waveform")
    if not isinstance(audio, dict) or audio.get("format") != "wav" or audio.get("sample_rate") != config.sample_rate:
        raise ValueError("Omni must return matching WAV audio")
    directory = Path(root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"sample-{sample.index}-{uuid.uuid4().hex}.wav"
    if audio.get("data") is not None:
        limit = int(getattr(args, "omni_max_audio_bytes", 64 * 1024 * 1024))
        if len(audio["data"]) > (limit + 2) // 3 * 4:
            raise ValueError("Omni waveform exceeds the configured byte budget")
        data = base64.b64decode(audio["data"], validate=True)
    elif audio.get("path") is not None:
        data = Path(audio["path"]).read_bytes()
    else:
        raise ValueError("Omni waveform requires data or a shared path")
    if not data or data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise ValueError("Omni waveform is not WAV bytes")
    with wave.open(io.BytesIO(data), "rb") as source:
        if (
            source.getframerate() != config.sample_rate
            or source.getnframes() < 1
            or source.getsampwidth() != 2
            or source.getnchannels() not in (1, 2)
        ):
            raise ValueError("Omni WAV payload disagrees with its metadata or PCM encoding")
        expected_bytes = source.getnframes() * source.getnchannels() * source.getsampwidth()
        if len(source.readframes(source.getnframes())) != expected_bytes:
            raise ValueError("Omni WAV payload is truncated")
    output.write_bytes(data)
    sample.metadata["omni_audio_path"] = str(output)


async def generate(args, sample, sampling_params, *, evaluation=False):
    from slime.rollout.sglang_rollout import get_model_url

    if sample.status not in (Sample.Status.PENDING, Sample.Status.ABORTED):
        raise ValueError(f"Prism generation requires a pending sample, got {sample.status}")
    payload = build_generate_payload(args, sample, sampling_params)
    config = _config(str(args.hf_checkpoint))
    headers = None
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": sample.session_id}
    response = await _request(args, get_model_url(args, payload["model"]), payload, headers=headers)
    parse_rollout(
        response,
        sample,
        config,
        max_frames=payload["sampling_params"]["max_new_tokens"],
        expected_weight_version=sample.metadata.get(
            "omni_expected_weight_version", sample.metadata.get("policy_version")
        ),
    )
    _write_audio(response, args, sample, config, payload["return_audio"])
    return sample


generate.abort_mode = "request"


def _load_teacher_config(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        raise ValueError("omni_teacher_config must be a JSON/YAML path or object")
    if value.lstrip().startswith("{"):
        return json.loads(value)
    content = Path(value).read_text()
    if Path(value).suffix.lower() == ".json":
        return json.loads(content)
    import yaml

    return yaml.safe_load(content)


def teacher_request(sample):
    trace = sample.metadata["omni_rollout"]
    sample_id = f"{sample.group_index}:{sample.index}:{trace['request_id']}"
    return dict(
        version=1,
        sample_id=sample_id,
        **{name: copy.deepcopy(trace["replay_inputs"][name]) for name in REPLAY_FIELDS},
        codes=copy.deepcopy(trace["action_streams"][0]["actions"]),
        finish_reason=trace["finish_reason"],
        temperature=1.0,
    )


def teacher_request_hash(request):
    normalized = {key: value for key, value in request.items() if key != "sample_id"}
    normalized["temperature"] = float(normalized["temperature"])
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def attach_teacher_scores(sample, request, result, binding):
    """Convert teacher Bernoulli probabilities into the student's sampled action scores."""
    identity = binding.get("model_identity")
    student = sample.metadata["omni_rollout"]["model_identity"]
    if not isinstance(identity, dict) or {k: v for k, v in identity.items() if k != "config_sha256"} != {
        k: v for k, v in student.items() if k != "config_sha256"
    }:
        raise ValueError("Prism teacher binding has incompatible replay identity")
    if identity.get("config_sha256") != student.get("config_sha256"):
        checkpoint = binding.get("checkpoint") or binding.get("model_path")
        if not checkpoint:
            raise ValueError("A different-config teacher requires local checkpoint assets for replay compatibility")
        teacher_config = _config(str(checkpoint))
        if (
            teacher_config.identity != identity
            or teacher_config.replay_geometry() != sample.metadata.get("omni_replay_geometry")
            or not sample.metadata.get("omni_tokenizer_sha256")
            or teacher_config.tokenizer_sha256 != sample.metadata["omni_tokenizer_sha256"]
        ):
            raise ValueError("Prism teacher checkpoint has incompatible geometry, special tokens or tokenizer")
    checksum = binding.get("weight_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Prism teacher requires a pinned weight_sha256")
    if type(result.get("version")) is not int or type(result.get("temperature")) not in (int, float):
        raise ValueError("Prism teacher response version or temperature has invalid scalar types")
    expected = dict(
        version=1,
        sample_id=request["sample_id"],
        input_sha256=teacher_request_hash(request),
        temperature=1.0,
        finish_reason=request["finish_reason"],
        model_identity=identity,
        teacher_weight_sha256=checksum,
        logprob_semantics=LOGPROB_SEMANTICS,
        stop_semantics="pre_frame_threshold_v1",
    )
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Prism teacher response identity, input hash, checksum or semantics differs")
    if not isinstance(result.get("weight_version"), str) or not result["weight_version"]:
        raise ValueError("Prism teacher weight version is missing")
    f, q = sample.response_length, student["n_vq"]
    stops = f + int(request["finish_reason"] == "stop")
    scores = _tensor(result.get("code_logprobs"), (f, q), torch.float32, "teacher code logprobs")
    probabilities = _tensor(result.get("stop_probabilities"), (stops,), torch.float64, "teacher stop probabilities")
    if (scores > 1e-5).any() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Teacher selected scores are invalid or stop probabilities are outside [0, 1]")
    mm = sample.multimodal_train_inputs
    actions = mm["policy_stop_actions"][mm["policy_stop_mask"]]
    selected = torch.where(actions.bool(), probabilities.log(), torch.log1p(-probabilities)).float()
    if not torch.isfinite(selected).all():
        raise ValueError("Selected teacher stop action has zero probability; the scorer must return stable logprobs")
    code_result = torch.zeros_like(mm["policy_rollout_code_logprobs"])
    stop_result = torch.zeros_like(mm["policy_rollout_stop_logprobs"])
    code_result[mm["policy_code_mask"].any(-1)] = scores
    stop_result[mm["policy_stop_mask"]] = selected
    mm["policy_teacher_code_logprobs"] = code_result
    mm["policy_teacher_stop_logprobs"] = stop_result
    sample.teacher_log_probs = scores.sum(-1).tolist()
    sample.metadata["omni_teacher_identity"] = copy.deepcopy(identity)
    sample.metadata["omni_teacher_weight_sha256"] = checksum
    sample.metadata["omni_teacher_weight_version"] = result["weight_version"]
    return sample


def _validate_frozen_deployment(args, deployment):
    """Reject trainable named models before discovering any teacher worker."""
    if not isinstance(deployment, str) or not deployment:
        raise ValueError("Prism teacher deployment must be a nonempty model name")
    if deployment == getattr(args, "omni_model_name", "student"):
        raise ValueError("A Prism teacher deployment cannot be the student model")
    configuration = getattr(args, "sglang_config", None)
    if configuration is None:
        # External/pre-resolved deployments may expose only their router map.
        # Pinned response identity and checksum remain mandatory below.
        return
    if isinstance(configuration, (str, Path)):
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        configuration = SglangConfig.from_yaml(str(configuration))
    models = getattr(configuration, "models", None)
    if models is None:
        raise ValueError("Prism teacher requires a model-aware sglang_config")
    matches = [model for model in models if model.name == deployment]
    if len(matches) != 1:
        raise ValueError(f"Prism teacher deployment {deployment!r} must name one configured model")
    model = matches[0]
    if model.update_weights is None:
        model.resolve(args)
    if model.update_weights is not False:
        raise ValueError(f"Prism teacher deployment {deployment!r} must have update_weights=false")


async def _teacher_endpoint(args, sample, binding):
    """Discover a named teacher's live worker; the router has no score API."""
    selected = [name for name in ("endpoint", "deployment") if binding.get(name) is not None]
    if len(selected) != 1:
        raise ValueError("Prism teacher binding must specify exactly one of endpoint or deployment")
    if selected[0] == "endpoint":
        endpoint = binding["endpoint"]
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
            raise ValueError("Prism teacher endpoint must be HTTP(S)")
        return endpoint.rstrip("/")
    deployment = binding["deployment"]
    _validate_frozen_deployment(args, deployment)
    routers = getattr(args, "sglang_model_routers", None) or {}
    if deployment not in routers:
        raise ValueError(f"Prism teacher deployment {deployment!r} has no registered router")
    address = routers[deployment]
    if not isinstance(address, (tuple, list)) or len(address) != 2:
        raise ValueError("Named Omni model router must provide its (host, port)")
    host, port = address
    port = _integer(port, "teacher router port", minimum=1)
    timeout = float(getattr(args, "rollout_health_check_timeout", 10.0))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Teacher health timeout must be finite and positive")
    router = f"http://{_wrap_ipv6(str(host))}:{port}"
    body = await asyncio.wait_for(get(router + "/workers", timeout=timeout), timeout=timeout)
    if not isinstance(body, dict) or not isinstance(body.get("workers"), list):
        raise ValueError("Omni teacher router /workers returned an invalid worker list")
    workers = sorted(
        {
            worker["url"].rstrip("/")
            for worker in body["workers"]
            if isinstance(worker, dict)
            and worker.get("health_state") == "healthy"
            and worker.get("routable") is True
            and worker.get("disabled") is False
            and worker.get("model") in (None, deployment)
            and isinstance(worker.get("url"), str)
            and worker["url"].startswith(("http://", "https://"))
        }
    )
    if not workers:
        raise ValueError(f"Omni teacher deployment {deployment!r} has no healthy routable workers")
    digest = hashlib.sha256(json.dumps([sample.group_index, sample.index]).encode()).digest()
    start = int.from_bytes(digest[:8], "little") % len(workers)
    for offset in range(len(workers)):
        worker = workers[(start + offset) % len(workers)]
        try:
            health = await asyncio.wait_for(get(worker + "/health", timeout=timeout), timeout=timeout)
        except Exception:
            continue
        if isinstance(health, dict) and health.get("running") is True:
            return worker
    raise ValueError(f"Omni teacher deployment {deployment!r} has no reachable running workers")


async def score_teacher(args, sample, **kwargs):
    """Per-sample hook: route an explicit domain to one frozen Prism teacher."""
    if not getattr(args, "omni_teacher_config", None):
        return sample
    config = _load_teacher_config(args.omni_teacher_config)
    models, routes = config.get("models", {}), config.get("by_domain", {})
    domain = sample.metadata.get("domain")
    key = config.get("default_teacher") if domain is None else routes.get(domain)
    if key not in models:
        raise ValueError(f"Unmapped Prism teacher domain {domain!r}")
    binding = models[key]
    if not isinstance(binding, dict):
        raise ValueError("Prism teacher binding must be an object")
    endpoint = await _teacher_endpoint(args, sample, binding)
    token_name = binding.get("auth_token_env")
    token = os.environ.get(token_name) if token_name else None
    if token_name and not token:
        raise ValueError(f"Teacher authentication environment {token_name!r} is missing")
    request = teacher_request(sample)
    response = await _request(
        args,
        endpoint + "/score_actions",
        {"samples": [request]},
        headers={"Authorization": f"Bearer {token}"} if token else None,
    )
    if (
        not isinstance(response, dict)
        or type(response.get("version")) is not int
        or response["version"] != 1
        or not isinstance(response.get("results"), list)
        or len(response["results"]) != 1
        or not isinstance(response["results"][0], dict)
    ):
        raise ValueError("Prism teacher response version or result count differs")
    attach_teacher_scores(sample, request, response["results"][0], binding)
    sample.metadata["omni_teacher_key"] = key
    sample.metadata["omni_teacher_endpoint"] = endpoint
    if "deployment" in binding:
        sample.metadata["omni_teacher_deployment"] = binding["deployment"]
    return sample
