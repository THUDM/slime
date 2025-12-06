"""RLVE reward function - uses environment verifiers for deterministic scoring."""

import logging
from typing import Any

from slime.utils.types import Sample

from .rlve_prompt_provider import RLVE_AVAILABLE, get_provider

if RLVE_AVAILABLE:
    from Gym.environments import identifier2environment

logger = logging.getLogger(__name__)


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> dict[str, Any]:
    """Compute reward using RLVE environment verifier."""
    metadata = sample.metadata
    env_id = metadata["env_id"]
    config = metadata["config"]
    answer_markers = metadata.get("answer_markers", ("<answer>", "</answer>"))
    difficulty_index = metadata.get("difficulty_index")

    if not RLVE_AVAILABLE:
        return _stub_reward(sample, config)

    env_class = identifier2environment[env_id]
    env = env_class(answer_markers=answer_markers)
    env.set_config(config)

    result = env.verifier(sample.response)

    provider = get_provider()
    n_samples_per_prompt = 1
    if args is not None and hasattr(args, "n_samples_per_prompt"):
        try:
            n_samples_per_prompt = int(args.n_samples_per_prompt)
        except (TypeError, ValueError):
            n_samples_per_prompt = 1
    provider.update_controller(
        env_id,
        result.get("accuracy", 0.0),
        difficulty_index,
        n_samples_per_prompt,
    )

    # Apply format_coef bonus (matches RLVE's tinker integration)
    format_coef = provider.config.get("format_coef", 0.0)
    base_reward = result.get("reward", 0.0)
    format_score = result.get("format_score", 0)
    total_reward = base_reward + format_coef * format_score

    logger.debug(
        f"[{env_id}] reward={total_reward:.2f} "
        f"(base={base_reward:.2f}, format={format_score}, coef={format_coef}) "
        f"accuracy={result.get('accuracy', 0.0)}"
    )

    return {
        "reward": total_reward,
        "accuracy": result.get("accuracy", 0.0),
        "format_score": format_score,
        "env_id": env_id,
    }


def _stub_reward(sample: Sample, config: dict[str, Any]) -> dict[str, Any]:
    """Stub reward for CI testing."""
    response = sample.response or ""
    expected = config.get("answer", 0)

    accuracy = 0
    format_score = 0

    if "<answer>" in response and "</answer>" in response:
        format_score = 1
        try:
            start = response.index("<answer>") + len("<answer>")
            end = response.index("</answer>", start)
            answer_text = response[start:end].strip()
            if str(expected) in answer_text:
                accuracy = 1
        except ValueError:
            pass

    return {
        "reward": float(accuracy),
        "accuracy": accuracy,
        "format_score": format_score,
        "env_id": "StubAddition",
    }
