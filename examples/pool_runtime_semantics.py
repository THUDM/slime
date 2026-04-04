from __future__ import annotations

import json
from hashlib import sha1
from typing import Any


STEM_SYSTEM_PROMPTS = [
    "You are a careful STEM reasoning assistant. Solve the problem step by step. If options are provided, end with the best option and a concise justification.",
    "You are an expert science and quantitative reasoning assistant. Reason carefully, avoid unsupported assumptions, and give the final answer clearly. If this is multiple choice, state the correct option explicitly.",
]

TOOL_SYSTEM_PROMPTS = [
    "You are a tool-using assistant. Use the provided tools precisely when they are useful, keep tool calls valid, and ground the final answer in the observed results.",
    "You are an assistant with access to external tools. Plan briefly, call tools only when needed, and produce a final answer that is consistent with the tool outputs.",
]

STRUCTURED_SYSTEM_PROMPTS = [
    "You are a structured output assistant. Produce an answer that exactly matches the requested schema or structure. Do not add extra keys or explanatory text.",
    "You are an information extraction assistant. Return only content that satisfies the requested structure, using valid JSON when structured output is required.",
]

INSTRUCTION_FOLLOWING_SYSTEM_PROMPTS = [
    "You are a helpful assistant. Follow the user's instruction carefully and answer clearly.",
    "You are a careful assistant. Respond directly to the user's request and keep the answer accurate and concise.",
]

PROMPT_FAMILY_TO_OPTIONS: dict[str, list[str]] = {
    "stem": STEM_SYSTEM_PROMPTS,
    "tool": TOOL_SYSTEM_PROMPTS,
    "structured": STRUCTURED_SYSTEM_PROMPTS,
    "instruction_following": INSTRUCTION_FOLLOWING_SYSTEM_PROMPTS,
}


def _stable_pick(options: list[str], payload: dict[str, Any], family: str) -> str:
    if len(options) == 1:
        return options[0]

    metadata = payload.get("metadata") or {}
    prompt = payload.get("prompt")
    try:
        prompt_repr = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    except TypeError:
        prompt_repr = str(prompt)
    key_parts = [
        family,
        str(metadata.get("dataset_name") or ""),
        str(metadata.get("record_id") or payload.get("id") or ""),
        prompt_repr[:200],
    ]
    digest = sha1("||".join(key_parts).encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _coerce_prompt_to_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        text = prompt.strip()
        return [{"role": "user", "content": text}] if text else []
    if not isinstance(prompt, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in prompt:
        if isinstance(item, dict):
            normalized.append(dict(item))
        else:
            normalized.append({"role": "user", "content": "" if item is None else str(item)})
    return normalized


def _has_system_message(prompt: list[dict[str, Any]]) -> bool:
    return bool(prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system")


def infer_prompt_family(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("metadata") or {}
    reward_type = str(metadata.get("reward_type") or "").strip().lower()
    dataset_name = str(metadata.get("dataset_name") or "").strip().lower()
    domain = str(metadata.get("domain") or "").strip().lower()

    if reward_type in {"instruction_following_soft", "instruction_following_strict", "ifeval"}:
        return "instruction_following"
    if dataset_name in {"ifbench_test", "ifeval"} or domain == "ifrl":
        return "instruction_following"
    if reward_type == "structured_json_schema":
        return "structured"
    if reward_type == "stem_mcqa" or domain == "stem":
        return "stem"
    if reward_type in {"tool_call_soft", "tool_call_strict", "tool_call", "tool_selection_strict"} or domain == "tool":
        return "tool"
    if domain == "structured":
        return "structured"
    return None


def materialize_runtime_pool_row(row: dict[str, Any]) -> dict[str, Any]:
    materialized = dict(row)
    metadata = dict(materialized.get("metadata") or {})
    prompt = _coerce_prompt_to_messages(materialized.get("prompt"))

    raw_tools = materialized.get("tools") or []
    tools = list(raw_tools) if isinstance(raw_tools, list) else []
    materialized["tools"] = tools
    if tools:
        metadata["tools"] = tools
    else:
        metadata.pop("tools", None)

    family = infer_prompt_family({"prompt": prompt, "metadata": metadata, "id": materialized.get("id")})
    if family and not _has_system_message(prompt):
        system_prompt = _stable_pick(
            PROMPT_FAMILY_TO_OPTIONS[family],
            {"prompt": prompt, "metadata": metadata, "id": materialized.get("id")},
            family,
        )
        prompt = [{"role": "system", "content": system_prompt.strip()}, *prompt]

    materialized["prompt"] = prompt
    materialized["metadata"] = metadata
    return materialized
