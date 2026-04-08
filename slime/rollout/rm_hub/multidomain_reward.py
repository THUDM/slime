from __future__ import annotations

from typing import Any, Callable

from slime.rollout.rm_hub.instruction_following_reward import compute_instruction_following_reward
from slime.rollout.rm_hub.reward_text import (
    clean_response,
    compute_stem_mcqa_reward,
    compute_structured_json_reward,
)
from slime.rollout.rm_hub.tool_call_reward import (
    TOOL_CALL_REWARD_TYPES,
    compute_tool_call_reward,
    parse_predicted_calls,
)


def compute_generic_reward(
    metadata: dict[str, Any],
    prompt_text: str,
    response: str,
    *,
    parse_predicted_calls_fn: Callable[[str, list[dict[str, Any]], str], list[dict[str, Any]]] = parse_predicted_calls,
    ifbench_rule_scores_fn: Callable[[dict[str, Any], str, str, bool], list[float] | None] | None = None,
    ifeval_rule_scores_fn: Callable[[dict[str, Any], str, str, bool], list[float] | None] | None = None,
) -> float:
    reward_type = str(metadata.get("reward_type", "")).strip()
    cleaned = clean_response(response)
    if not cleaned:
        return 0.0

    if reward_type == "bfcl_official":
        raise RuntimeError(
            "BFCL official evaluation is not supported through the generic multidomain reward router. "
            "Use the BFCL official runner."
        )

    if reward_type == "stem_mcqa":
        return compute_stem_mcqa_reward(cleaned, metadata.get("answer", ""))

    if reward_type in TOOL_CALL_REWARD_TYPES:
        return compute_tool_call_reward(
            metadata,
            cleaned,
            parse_predicted_calls_fn=parse_predicted_calls_fn,
        )

    if reward_type == "structured_json_schema":
        schema = metadata.get("schema") if isinstance(metadata.get("schema"), dict) else {}
        return compute_structured_json_reward(cleaned, schema)

    if reward_type in {"instruction_following_soft", "instruction_following_strict", "ifeval"}:
        return compute_instruction_following_reward(
            metadata,
            prompt_text,
            cleaned,
            ifbench_rule_scores_fn=ifbench_rule_scores_fn,
            ifeval_rule_scores_fn=ifeval_rule_scores_fn,
        )

    return 0.0
