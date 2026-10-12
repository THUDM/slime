"""Example-owned reward categories for ``--custom-rm-path``.

Samples select ``metadata.reward_category`` independently of their domain and
Prism task type. Component coefficients and frozen service bindings retain the
existing ``omni_reward_config`` schema. The common TTS clients own scoring,
identity checks, timeouts, audio handling, and component failure propagation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy

from slime_plugins import tts_reward


class _CategoryRewards:
    def __init__(self, config):
        if not isinstance(config, Mapping) or set(config) != {"services", "weights"}:
            raise ValueError("omni_reward_config requires services and weights mappings")
        services, weights = config["services"], config["weights"]
        if not isinstance(services, Mapping) or not isinstance(weights, Mapping):
            raise ValueError("reward services and weights must be mappings")
        if set(weights) - {"wer", "sim", "judge_clap"}:
            raise ValueError("supported reward components are wer, sim, and judge_clap")
        self.weights = {name: tts_reward._finite(value, f"{name} weight") for name, value in weights.items()}
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("reward weights must be nonnegative")
        self.options = deepcopy(dict(services))
        self.pipelines = {}
        self.clients = {}

    def pipeline(self, category, components):
        if category not in self.pipelines:
            missing = set(components) - self.weights.keys()
            if missing:
                raise ValueError(f"reward category {category!r} requires component weights: {sorted(missing)}")
            weights = {name: self.weights[name] for name in components}
            if any(weight <= 0 for weight in weights.values()):
                raise ValueError(f"reward category {category!r} requires positive weights for {components}")
            pipeline = tts_reward._Rewards({"services": self.options, "weights": weights})
            # All categories share each service's identity cache and semaphore.
            # Constructing a category is synchronous, before any scoring awaits.
            pipeline.services = {
                name: self.clients.setdefault(name, client) for name, client in pipeline.services.items()
            }
            self.pipelines[category] = pipeline
        return self.pipelines[category]


def _state(args):
    state = getattr(args, "_prism_category_rewards", None)
    if state is None:
        state = _CategoryRewards(tts_reward._config(getattr(args, "omni_reward_config", None)))
        args._prism_category_rewards = state
    return state


async def _gather_scores(coroutines):
    outcomes = await asyncio.gather(*coroutines, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return outcomes


async def _score_one(args, sample, category, components):
    if not isinstance(getattr(sample, "metadata", None), dict):
        raise ValueError("Prism category rewards require sample.metadata")
    sample.metadata.pop("omni_reward", None)
    pipeline = _state(args).pipeline(category, components)
    try:
        return await pipeline.score(sample)
    finally:
        outcome = sample.metadata.get("omni_reward")
        if isinstance(outcome, dict):
            outcome["category"] = category


async def _score_category(args, sample, category, components):
    if isinstance(sample, list):
        return await _gather_scores(_score_one(args, item, category, components) for item in sample)
    return await _score_one(args, sample, category, components)


async def reward_base(args, sample, **kwargs):
    """Base: WER/CER only."""
    return await _score_category(args, sample, "base", ("wer",))


async def reward_voiceclone(args, sample, **kwargs):
    """Voice clone: WER/CER plus speaker similarity."""
    return await _score_category(args, sample, "voiceclone", ("wer", "sim"))


async def reward_instruction(args, sample, **kwargs):
    """Instruction: WER/CER plus global JudgeCLAP instruction scoring."""
    return await _score_category(args, sample, "instruction", ("wer", "judge_clap"))


CATEGORY_REWARDS = {
    "base": reward_base,
    "voiceclone": reward_voiceclone,
    "instruction": reward_instruction,
}


def _category(sample):
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Prism category rewards require sample.metadata")
    category = metadata.get("reward_category")
    if not isinstance(category, str) or category not in CATEGORY_REWARDS:
        raise ValueError(f"metadata.reward_category must be one of {tuple(CATEGORY_REWARDS)}; got {category!r}")
    return category


async def reward(args, sample, **kwargs):
    """Dispatch a sample or ordered mixed-category batch to its reward function."""
    if isinstance(sample, list):
        categories = [_category(item) for item in sample]
        group_categories = {}
        for item, category in zip(sample, categories, strict=True):
            group = getattr(item, "group_index", None)
            if group is not None and group_categories.setdefault(group, category) != category:
                raise ValueError("All candidates in a GRPO prompt group must use the same reward_category")
        return await _gather_scores(
            CATEGORY_REWARDS[category](args, item, **kwargs) for item, category in zip(sample, categories, strict=True)
        )
    return await CATEGORY_REWARDS[_category(sample)](args, sample, **kwargs)
