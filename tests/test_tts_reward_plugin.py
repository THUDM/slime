"""CPU-only TTS reward business and frozen HTTP protocol contracts."""

import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime_plugins import tts_reward  # noqa: E402

NUM_GPUS = 0


def config(**weights):
    return {
        "services": {name: {"endpoint": f"http://{name}/prefix", "expected_fingerprint": name} for name in weights},
        "weights": weights,
    }


def sample(tmp_path, **overrides):
    candidate, reference = tmp_path / "candidate.wav", tmp_path / "reference.wav"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")
    metadata = {
        "omni_frames": 3,
        "omni_audio_path": str(candidate),
        "omni_prompt": {
            "transcript_target": "你好，世界！",
            "language": "zh-CN",
            "references": [{"id": "voice", "uri": str(reference)}],
            "reference_targets": {"timbre": "voice"},
            "global_instruction": '{"Speed": "slow", "Gender": "female"}',
        },
    }
    metadata.update(overrides)
    return SimpleNamespace(metadata=metadata)


class Service:
    def __init__(self):
        self.calls = []
        self.identity = {
            "wer": {"fingerprint": "wer", "model": "Qwen3-ASR-1.7B"},
            "sim": {"fingerprint": "sim", "service_task": "speaker_embedding", "embedding_dimension": 2},
            "judge_clap": {
                "fingerprint": "judge_clap",
                "supported_categories": ["Gender", "Speed"],
                "query_versions": [tts_reward.QUERY_VERSION],
            },
        }
        self.text = "你好，世。"
        self.candidate_vector = [0.6, 0.8]
        self.reference_vector = [1.0, 0.0]
        self.mutate = lambda body: body

    async def get(self, url, *, timeout=None):
        name = url.split("/")[2]
        self.calls.append(("get", url, timeout))
        return {"protocol": tts_reward.PROTOCOL, "identity": copy.deepcopy(self.identity[name])}

    async def post(self, url, payload, *, max_retries, timeout):
        name = url.split("/")[2]
        self.calls.append(("post", url, copy.deepcopy(payload), max_retries, timeout))
        if name == "wer":
            result = {"text": self.text}
        elif name == "sim":
            is_reference = payload["request_id"].endswith("reference")
            result = {
                "embedding": self.reference_vector if is_reference else self.candidate_vector,
                "dimension": 2,
                "normalized": True,
            }
        else:
            result = {"scores": [0.4, 0.8]}
        return self.mutate(
            {
                "protocol": tts_reward.PROTOCOL,
                "identity": copy.deepcopy(self.identity[name]),
                "request_id": payload["request_id"],
                "result": result,
            }
        )


@pytest.fixture
def service(monkeypatch):
    backend = Service()
    monkeypatch.setattr(tts_reward, "get", backend.get)
    monkeypatch.setattr(tts_reward, "post", backend.post)
    return backend


def test_all_rewards_preserve_raw_and_use_frozen_protocol(tmp_path, service):
    item = sample(tmp_path)
    args = SimpleNamespace(omni_reward_config=json.dumps(config(wer=0.6, sim=0.4, judge_clap=0.25)))
    total = asyncio.run(tts_reward.reward(args, item))
    assert total == pytest.approx(0.6 * 0.8 + 0.4 * 0.6 + 0.25 * 0.6)
    outcome = item.metadata["omni_reward"]
    assert outcome["components"]["wer"]["raw"] == 0.25
    assert outcome["components"]["wer"]["details"]["reference_tokens"] == 4
    assert outcome["components"]["sim"]["details"]["purpose"] == "timbre"
    assert outcome["components"]["judge_clap"]["details"]["queries"] == ["Gender: female", "Speed: slow"]
    posts = [call for call in service.calls if call[0] == "post"]
    assert all(call[3:] == (1, 120) for call in posts)
    asr_payload = next(call[2] for call in posts if "wer/prefix/score" in call[1])
    assert set(asr_payload) == {"request_id", "audio"}
    judge_payload = next(call[2] for call in posts if "judge_clap/prefix/score" in call[1])
    assert judge_payload["query_version"] == "moss_tts_v2_revision6_instruction_v1"
    assert sum(call[0] == "get" for call in service.calls) == 3


@pytest.mark.parametrize(
    "language,target,hypothesis,raw",
    [
        ("en", "Ａ dog, runs!", "a dog", 1 / 3),
        ("Japanese", "今日は", "今日", 1 / 3),
        ("ko", "안녕", "안", 0.5),
        ("zh", "你", "你好世界", 3),
    ],
)
def test_inverse_error_multilingual_and_unclipped_raw(tmp_path, service, language, target, hypothesis, raw):
    item = sample(tmp_path)
    item.metadata["omni_prompt"].update(language=language, transcript_target=target)
    service.text = hypothesis
    value = asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=1)), item))
    assert item.metadata["omni_reward"]["components"]["wer"]["raw"] == pytest.approx(raw)
    assert value == pytest.approx(1 / (1 + raw))


def test_batch_preserves_sample_order_and_sim_reference_cache(tmp_path, service):
    first, second = sample(tmp_path), sample(tmp_path)
    args = SimpleNamespace(omni_reward_config=config(sim=1))

    async def run():
        assert await tts_reward.reward(args, [first]) == pytest.approx([0.6])
        return await tts_reward.reward(args, [second, first])

    assert asyncio.run(run()) == pytest.approx([0.6, 0.6])
    references = [call for call in service.calls if call[0] == "post" and call[2]["request_id"].endswith("reference")]
    assert len(references) == 1
    assert sum(call[0] == "get" for call in service.calls) == 1


def test_prism_task_reference_mapping_and_metadata_target_fields(tmp_path, service):
    item = sample(tmp_path)
    prompt = item.metadata["omni_prompt"]
    prompt["task_type"] = "Full VoiceClone"
    for name in ("transcript_target", "language", "reference_targets"):
        item.metadata[name] = prompt.pop(name)
    item.metadata["reference_targets"] = {key: "voice" for key in ("timbre", "accent", "prosody", "emotion")}
    result = asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=0.6, sim=0.4)), item))
    assert result == pytest.approx(0.72)


def test_sim_never_guesses_reference_from_order(tmp_path, service):
    item = sample(tmp_path)
    del item.metadata["omni_prompt"]["reference_targets"]
    with pytest.raises(ValueError, match="explicit reference_targets"):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(sim=1)), item))
    assert service.calls == []


def test_sim_negative_cosine_zero_reward_and_invalid_reference_errors(tmp_path, service):
    item = sample(tmp_path)
    service.candidate_vector = [-1.0, 0.0]
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(sim=1)), item)) == 0
    assert item.metadata["omni_reward"]["components"]["sim"]["raw"] == -1
    service.reference_vector = [0.0, 0.0]
    with pytest.raises(tts_reward.ScoringError, match="not normalized"):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(sim=1)), item))


def test_sim_short_candidate_is_explicit_valid_low_reward(tmp_path, service):
    item = sample(tmp_path)

    def short(body):
        if body["request_id"].endswith("candidate"):
            body["result"] = {
                "status": "invalid_audio",
                "reason": "too_short",
                "num_samples": 200,
                "min_samples": 1600,
                "sample_rate": 16000,
            }
        return body

    service.mutate = short
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(sim=1)), item)) == 0
    component = item.metadata["omni_reward"]["components"]["sim"]
    assert component["status"] == "invalid_audio"
    assert component["raw"] is None
    assert component["details"]["reason"] == "too_short"


def test_zero_frame_has_no_http_and_keeps_raw_none(tmp_path, service):
    item = sample(tmp_path, omni_frames=0, omni_audio_path=None)
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=0.6, sim=0.4)), item)) == 0
    assert service.calls == []
    assert all(component["raw"] is None for component in item.metadata["omni_reward"]["components"].values())
    assert item.metadata["omni_reward"]["components"]["wer"]["details"] == {"reason": "zero_frame"}


def test_nonzero_frames_without_waveform_is_error(tmp_path, service):
    item = sample(tmp_path, omni_audio_path=None)
    with pytest.raises(ValueError, match="nonzero TTS frames"):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=1)), item))
    assert "omni_reward" not in item.metadata


@pytest.mark.parametrize("instruction", [None, "{}", '{"Speed": null}', '{"Unknown": "fast"}'])
def test_judge_requires_rendered_instruction_even_for_zero_frame(tmp_path, service, instruction):
    item = sample(tmp_path, omni_frames=0)
    item.metadata["omni_prompt"]["global_instruction"] = instruction
    with pytest.raises(ValueError):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(judge_clap=1)), item))
    assert service.calls == []


def test_instruction_normalization_and_local_span_exclusion():
    instruction = '{"global instruction":{"speed":"“slow”", "Gender":" female\\n "},"local instruction":[]}'
    assert tts_reward.instruction_queries(instruction) == ["Gender: female", "Speed: slow"]
    for invalid in (
        '{"Speed":"slow","Speed":"fast"}',
        '{"Speed":"slow", " speed ":"fast"}',
        '{"Speed":"<|reserved|>"}',
        '{"Speed":2}',
    ):
        with pytest.raises(ValueError):
            tts_reward.instruction_queries(invalid)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda body: body.update(protocol="wrong"), "protocol"),
        (lambda body: body.update(request_id="wrong"), "request_id"),
        (lambda body: body["identity"].update(model="other"), "identity changed"),
        (lambda body: body.update(result=None), "result object"),
    ],
)
def test_response_contract_errors_never_become_rewards(tmp_path, service, mutation, error):
    item = sample(tmp_path)

    def corrupt(body):
        mutation(body)
        return body

    service.mutate = corrupt
    with pytest.raises(tts_reward.ScoringError, match=error):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=1)), item))
    assert item.metadata["omni_reward"]["status"] == "error"
    assert "total" not in item.metadata["omni_reward"]


def test_configured_fingerprint_is_checked_before_post(tmp_path, service):
    args = SimpleNamespace(omni_reward_config=config(wer=1))
    args.omni_reward_config["services"]["wer"]["expected_fingerprint"] = "different-model"
    with pytest.raises(tts_reward.ScoringError, match="fingerprint"):
        asyncio.run(tts_reward.reward(args, sample(tmp_path)))
    assert not any(call[0] == "post" for call in service.calls)


def test_component_failure_waits_for_other_audio_readers(tmp_path, service, monkeypatch):
    finished = []
    original = service.post

    async def post(url, payload, **kwargs):
        if "wer/prefix" in url:
            raise RuntimeError("ASR unavailable")
        await asyncio.sleep(0.01)
        result = await original(url, payload, **kwargs)
        finished.append(payload["request_id"])
        return result

    monkeypatch.setattr(tts_reward, "post", post)
    item = sample(tmp_path)
    with pytest.raises(RuntimeError, match="ASR unavailable"):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=config(wer=0.6, sim=0.4)), item))
    assert len(finished) == 2
    assert item.metadata["omni_reward"]["components"]["sim"]["raw"] == pytest.approx(0.6)


def test_http_timeout_is_bounded(tmp_path, service, monkeypatch):
    async def post(url, payload, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(tts_reward, "post", post)
    settings = config(wer=1)
    settings["services"]["wer"]["timeout_s"] = 0.01
    with pytest.raises(TimeoutError):
        asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=settings), sample(tmp_path)))


def test_json_file_configuration_and_explicit_upload(tmp_path, service):
    settings = config(wer=1)
    settings["services"]["wer"].update(audio_transport="upload", timeout_s=15, max_retries=2)
    path = tmp_path / "rewards.json"
    path.write_text(json.dumps(settings))
    item = sample(tmp_path)
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=str(path)), item)) == pytest.approx(0.8)
    call = next(call for call in service.calls if call[0] == "post")
    assert call[3:] == (3, 15)
    assert call[2]["audio"] == {"data_b64": "Y2FuZGlkYXRl", "suffix": ".wav"}


def test_yaml_configuration(tmp_path, service):
    pytest.importorskip("yaml")
    path = tmp_path / "rewards.yaml"
    path.write_text(
        "services:\n  wer:\n    endpoint: http://wer/prefix\n    expected_fingerprint: wer\nweights:\n  wer: 1\n"
    )
    assert asyncio.run(tts_reward.reward(SimpleNamespace(omni_reward_config=path), sample(tmp_path))) == pytest.approx(
        0.8
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
