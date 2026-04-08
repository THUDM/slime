from __future__ import annotations

from slime.rollout.rm_hub.instruction_following_reward import instruction_rule_scores
from slime.rollout.rm_hub.reward_text import clean_response


async def reward_func(args, sample, **kwargs):
    del args, kwargs
    if sample.status == sample.Status.TRUNCATED:
        return 0.0

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    prompt_text = metadata.get("prompt_text") or sample.prompt or ""
    prompt_text = prompt_text if isinstance(prompt_text, str) else str(prompt_text)
    response = clean_response(sample.response or "")
    if not response:
        return 0.0

    scores = instruction_rule_scores(metadata, prompt_text, response)
    if not scores:
        return 0.0
    return 1.0 if all(score == 1.0 for score in scores) else 0.0
