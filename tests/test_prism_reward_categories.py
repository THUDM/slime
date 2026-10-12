"""CPU contracts for example-owned, mixed-category Prism rewards."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.prism import rewards
from test_tts_reward_plugin import Service, config, sample

from slime_plugins import tts_reward

NUM_GPUS = 0


@pytest.fixture
def service(monkeypatch):
    backend = Service()
    monkeypatch.setattr(tts_reward, "get", backend.get)
    monkeypatch.setattr(tts_reward, "post", backend.post)
    return backend


def arguments(**weights):
    return SimpleNamespace(omni_reward_config=config(**(weights or {"wer": 1, "sim": 1, "judge_clap": 1})))


def categorized_sample(tmp_path, category, **metadata):
    item = sample(tmp_path, reward_category=category, domain="asmr", **metadata)
    # Identical domain/task type across categories exercises independent routing.
    item.metadata["omni_prompt"]["task_type"] = "Full VoiceClone"
    item.metadata["omni_prompt"]["reference_targets"] = {
        key: "voice" for key in ("timbre", "accent", "prosody", "emotion")
    }
    return item


@pytest.mark.parametrize(
    "category,expected,components",
    [
        ("base", 0.8, {"wer"}),
        ("voiceclone", 1.4, {"wer", "sim"}),
        ("instration", 1.4, {"wer", "judge_clap"}),
    ],
)
def test_category_selects_only_its_components(tmp_path, service, category, expected, components):
    item = categorized_sample(tmp_path, category)
    result = asyncio.run(rewards.reward(arguments(), item))
    assert result == pytest.approx(expected)
    outcome = item.metadata["omni_reward"]
    assert outcome["category"] == category
    assert set(outcome["components"]) == components
    assert set(outcome["weights"]) == components
    assert {call[1].split("/")[2] for call in service.calls} == components
    assert item.metadata["domain"] == "asmr"
    assert item.metadata["omni_prompt"]["task_type"] == "Full VoiceClone"


def test_mixed_batch_keeps_order_and_shares_frozen_service_identity(tmp_path, service):
    args = arguments()
    items = [categorized_sample(tmp_path, category) for category in ("instration", "base", "voiceclone", "base")]
    result = asyncio.run(rewards.reward(args, items))
    assert result == pytest.approx([1.4, 0.8, 1.4, 0.8])
    assert [item.metadata["omni_reward"]["category"] for item in items] == ["instration", "base", "voiceclone", "base"]
    assert sum(call[0] == "get" and "/wer/" in call[1] for call in service.calls) == 1


@pytest.mark.parametrize(
    "function,category,expected",
    [
        (rewards.reward_base, "base", 0.8),
        (rewards.reward_voiceclone, "voiceclone", 1.4),
        (rewards.reward_instration, "instration", 1.4),
    ],
)
def test_category_functions_are_standalone_batch_hooks(tmp_path, service, function, category, expected):
    items = [sample(tmp_path), sample(tmp_path)]
    assert asyncio.run(function(arguments(), items)) == pytest.approx([expected, expected])
    assert all(item.metadata["omni_reward"]["category"] == category for item in items)


def test_base_uses_wer_only_config_and_does_not_mutate_legacy_cache(tmp_path, service):
    args = arguments(wer=1)
    sentinel = args._omni_reward_pipeline = object()
    item = categorized_sample(tmp_path, "base")
    prompt = item.metadata["omni_prompt"]
    prompt.pop("references")
    prompt.pop("reference_targets")
    prompt.pop("global_instruction")
    assert asyncio.run(rewards.reward(args, item)) == pytest.approx(0.8)
    assert args._omni_reward_pipeline is sentinel
    assert args.omni_reward_config == config(wer=1)


@pytest.mark.parametrize(
    "category,unused",
    [
        ("base", ("references", "global_instruction")),
        ("voiceclone", ("global_instruction",)),
        ("instration", ("references",)),
    ],
)
def test_unused_reward_inputs_are_not_required(tmp_path, service, category, unused):
    item = categorized_sample(tmp_path, category)
    for field in unused:
        item.metadata["omni_prompt"].pop(field)
    asyncio.run(rewards.reward(arguments(), item))


def test_configured_coefficients_are_preserved_without_sum_normalization(tmp_path, service):
    args = arguments(wer=0.6, sim=0.4, judge_clap=0.2)
    items = [categorized_sample(tmp_path, category) for category in ("base", "voiceclone", "instration")]
    assert asyncio.run(rewards.reward(args, items)) == pytest.approx([0.48, 0.72, 0.6])
    assert items[1].metadata["omni_reward"]["weights"] == {"wer": 0.6, "sim": 0.4}


@pytest.mark.parametrize("category", [None, "", "instruction", "unknown", False, ["base"]])
def test_missing_or_unknown_category_fails_before_http(tmp_path, service, category):
    item = categorized_sample(tmp_path, category)
    with pytest.raises(ValueError, match="metadata.reward_category"):
        asyncio.run(rewards.reward(arguments(), item))
    assert service.calls == []


def test_invalid_category_in_batch_is_rejected_before_any_scoring(tmp_path, service):
    items = [categorized_sample(tmp_path, "base"), categorized_sample(tmp_path, "unknown")]
    with pytest.raises(ValueError, match="metadata.reward_category"):
        asyncio.run(rewards.reward(arguments(), items))
    assert service.calls == []


def test_mixed_categories_in_one_grpo_group_fail_before_http(tmp_path, service):
    items = [categorized_sample(tmp_path, category) for category in ("base", "voiceclone")]
    for item in items:
        item.group_index = 7
    with pytest.raises(ValueError, match="GRPO prompt group"):
        asyncio.run(rewards.reward(arguments(), items))
    assert service.calls == []


def test_same_category_in_one_grpo_group_is_valid(tmp_path, service):
    items = [categorized_sample(tmp_path, "base") for _ in range(2)]
    for item in items:
        item.group_index = 7
    assert asyncio.run(rewards.reward(arguments(wer=1), items)) == pytest.approx([0.8, 0.8])


@pytest.mark.parametrize(
    "category,field,message",
    [
        ("base", "transcript_target", "transcript_target"),
        ("voiceclone", "references", "references"),
        ("instration", "global_instruction", "JudgeCLAP"),
    ],
)
def test_selected_reward_inputs_remain_strict(tmp_path, service, category, field, message):
    item = categorized_sample(tmp_path, category)
    item.metadata["omni_prompt"].pop(field)
    with pytest.raises(ValueError, match=message):
        asyncio.run(rewards.reward(arguments(), item))
    assert service.calls == []
    assert "omni_reward" not in item.metadata


@pytest.mark.parametrize(
    "weight,error",
    [(0, ValueError), (-1, ValueError), (float("nan"), tts_reward.ScoringError), (True, tts_reward.ScoringError)],
)
def test_required_component_cannot_be_disabled_or_have_invalid_weight(tmp_path, service, weight, error):
    args = arguments(wer=1, sim=weight)
    with pytest.raises(error, match="weight"):
        asyncio.run(rewards.reward(args, categorized_sample(tmp_path, "voiceclone")))
    assert service.calls == []


def test_missing_component_weight_or_service_is_explicit(tmp_path, service):
    item = categorized_sample(tmp_path, "voiceclone")
    with pytest.raises(ValueError, match="component weights.*sim"):
        asyncio.run(rewards.reward(arguments(wer=1), item))
    args = arguments(wer=1, sim=1)
    del args.omni_reward_config["services"]["sim"]
    with pytest.raises(ValueError, match="missing reward services.*sim"):
        asyncio.run(rewards.reward(args, item))
    assert service.calls == []


@pytest.mark.parametrize("category", ["base", "voiceclone", "instration"])
def test_zero_frame_keeps_selected_category_outcomes_without_http(tmp_path, service, category):
    item = categorized_sample(tmp_path, category, omni_frames=0, omni_audio_path=None)
    assert asyncio.run(rewards.reward(arguments(), item)) == 0
    outcome = item.metadata["omni_reward"]
    assert outcome["category"] == category
    assert all(component["details"] == {"reason": "zero_frame"} for component in outcome["components"].values())
    assert service.calls == []


def test_service_concurrency_limit_is_shared_across_categories(tmp_path, service, monkeypatch):
    args = arguments()
    args.omni_reward_config["services"]["wer"]["max_inflight"] = 1
    active = maximum = 0
    original = service.post

    async def post(url, payload, **kwargs):
        nonlocal active, maximum
        if "/wer/" not in url:
            return await original(url, payload, **kwargs)
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01)
            return await original(url, payload, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(tts_reward, "post", post)
    items = [categorized_sample(tmp_path, category) for category in ("base", "voiceclone", "instration")]
    asyncio.run(rewards.reward(args, items))
    assert maximum == 1
    assert sum(call[0] == "get" and "/wer/" in call[1] for call in service.calls) == 1


def test_batch_failure_waits_for_other_categories_and_keeps_error_details(tmp_path, service, monkeypatch):
    finished = []
    original = service.post

    async def post(url, payload, **kwargs):
        if "/sim/" in url:
            raise RuntimeError("speaker service unavailable")
        if "/judge_clap/" in url:
            await asyncio.sleep(0.01)
            finished.append("judge_clap")
        return await original(url, payload, **kwargs)

    monkeypatch.setattr(tts_reward, "post", post)
    items = [categorized_sample(tmp_path, category) for category in ("voiceclone", "instration", "base")]
    with pytest.raises(RuntimeError, match="speaker service unavailable"):
        asyncio.run(rewards.reward(arguments(), items))
    assert finished == ["judge_clap"]
    assert items[0].metadata["omni_reward"]["status"] == "error"
    assert items[0].metadata["omni_reward"]["category"] == "voiceclone"
    assert "total" not in items[0].metadata["omni_reward"]
    assert all(item.metadata["omni_reward"]["status"] == "success" for item in items[1:])


def test_voiceclone_reference_cache_is_reused(tmp_path, service):
    args = arguments(wer=1, sim=1)

    async def run():
        for _ in range(2):
            await rewards.reward(args, categorized_sample(tmp_path, "voiceclone"))

    asyncio.run(run())
    references = [call for call in service.calls if call[0] == "post" and call[2]["request_id"].endswith("reference")]
    assert len(references) == 1


@pytest.mark.parametrize("format", ["inline_json", "json_file", "yaml_file"])
def test_existing_reward_config_formats_work(tmp_path, service, format):
    settings = config(wer=1)
    value = json.dumps(settings)
    if format != "inline_json":
        path = tmp_path / ("rewards.json" if format == "json_file" else "rewards.yaml")
        if format == "yaml_file":
            import yaml

            value = yaml.safe_dump(settings)
        path.write_text(value)
        value = str(path)
    args = SimpleNamespace(omni_reward_config=value)
    assert asyncio.run(rewards.reward(args, categorized_sample(tmp_path, "base"))) == pytest.approx(0.8)


def test_original_rm_hooks_load_category_router_for_samples_and_batches(tmp_path, service):
    from slime.rollout.rm_hub import async_rm, batched_async_rm

    args = arguments()
    args.custom_rm_path = "examples.prism.rewards.reward"
    items = [categorized_sample(tmp_path, category) for category in ("base", "voiceclone", "instration")]
    for item in items:
        item.custom_rm_path = None

    async def run():
        first = await async_rm(args, items[0])
        batch = await batched_async_rm(args, items)
        return first, batch

    first, batch = asyncio.run(run())
    assert first == pytest.approx(0.8)
    assert batch == pytest.approx([0.8, 1.4, 1.4])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
