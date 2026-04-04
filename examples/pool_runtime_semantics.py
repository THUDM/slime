from __future__ import annotations

import json
from hashlib import sha1
from typing import Any


BENCHMARK_NATIVE_DATASETS = {
    "bfcl_v3",
    "bfcl_v3_multi_turn_base",
    "toolbench_v1_benchmark",
    "ifeval",
    "ifbench_test",
    "jsonschemabench",
    "gpqa",
    "mmlu_pro",
}

TRAIN_NATIVE_SUPERVISION_FAMILIES = {
    "function_call_single",
    "function_call_verified_multi_turn",
    "agent_trace_call_recovery",
    "tool_use_trajectory",
}


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
    if dataset_name in {"bfcl_v3", "bfcl_v3_multi_turn_base", "toolbench_v1_benchmark"}:
        return "tool"
    if dataset_name in {"gpqa", "mmlu_pro"}:
        return "stem"
    if dataset_name == "jsonschemabench":
        return "structured"
    if reward_type == "structured_json_schema":
        return "structured"
    if reward_type == "stem_mcqa" or domain == "stem":
        return "stem"
    if reward_type in {"tool_call_soft", "tool_call_strict", "tool_call", "tool_selection_strict"} or domain == "tool":
        return "tool"
    if domain == "structured":
        return "structured"
    return None


def _toolbench_allowed_tool_names(native: dict[str, Any]) -> list[str]:
    relevant = native.get("relevant_apis")
    if isinstance(relevant, list):
        names: list[str] = []
        for item in relevant:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, (list, tuple)):
                name = ""
                for value in reversed(item):
                    if isinstance(value, str) and value.strip():
                        name = value.strip()
                        break
            elif isinstance(item, dict):
                name = ""
                for key in ("api_name", "name", "tool_name", "api", "tool"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        name = value.strip()
                        break
            else:
                name = ""
            if name:
                names.append(name)
        return names
    return []


def _runtime_metadata_from_train_native(
    dataset_name: str,
    domain: str,
    record_id: Any,
    supervision_family: str,
    native: dict[str, Any],
    base_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    metadata = {
        "dataset_name": dataset_name,
        "domain": domain,
        "record_id": record_id,
        **base_metadata,
        "native": native,
    }
    metadata.pop("tools", None)
    source_fields = native.get("source_fields")
    if isinstance(source_fields, dict):
        metadata["source_fields"] = source_fields
    source_record_fields = native.get("source_record_fields")
    if isinstance(source_record_fields, dict):
        metadata["source_record_fields"] = source_record_fields

    if supervision_family in TRAIN_NATIVE_SUPERVISION_FAMILIES:
        ground_truth = native.get("ground_truth") or []
        if not ground_truth:
            return None
        metadata["reward_type"] = "tool_call_soft"
        metadata["ground_truth"] = ground_truth
        if native.get("assistant_reference") not in (None, ""):
            metadata["assistant_reference"] = str(native.get("assistant_reference"))
        for key in ("provider", "parse_mode", "raw_api_call", "accepted_for_eval", "recovery_source", "target_message_index"):
            if native.get(key) not in (None, ""):
                metadata[key] = native.get(key)
        return metadata

    return metadata


def _runtime_metadata_from_native(
    dataset_name: str,
    domain: str,
    record_id: Any,
    native: dict[str, Any],
    base_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "dataset_name": dataset_name,
        "domain": domain,
        "record_id": record_id,
        **base_metadata,
        "native": native,
    }
    metadata.pop("tools", None)

    if dataset_name == "ifbench_test":
        metadata["reward_type"] = "instruction_following_soft"
        metadata["prompt_text"] = str(native.get("prompt") or "")
        metadata["instruction_id_list"] = list(native.get("instruction_id_list") or [])
        metadata["kwargs"] = list(native.get("kwargs") or [])
        return metadata
    if dataset_name == "ifeval":
        metadata["reward_type"] = "instruction_following_strict"
        metadata["prompt_text"] = str(native.get("prompt") or "")
        metadata["instruction_id_list"] = list(native.get("instruction_id_list") or [])
        metadata["kwargs"] = list(native.get("kwargs") or [])
        return metadata
    if dataset_name == "jsonschemabench":
        metadata["reward_type"] = "structured_json_schema"
        schema = native.get("json_schema")
        metadata["schema"] = schema if isinstance(schema, dict) else {}
        return metadata
    if dataset_name == "toolbench_v1_benchmark":
        metadata["reward_type"] = "tool_selection_strict"
        metadata["allowed_tool_names"] = _toolbench_allowed_tool_names(native)
        return metadata
    if dataset_name in {"gpqa", "mmlu_pro"}:
        metadata["reward_type"] = "stem_mcqa"
        answer = native.get("answer", native.get("Correct Answer", ""))
        metadata["answer"] = "" if answer is None else str(answer).strip().upper()
        return metadata
    if dataset_name in {"bfcl_v3", "bfcl_v3_multi_turn_base"}:
        metadata["reward_type"] = "bfcl_official"
        metadata["official_eval_name"] = "bfcl"
        if native.get("test_category") not in (None, ""):
            metadata["test_category"] = str(native.get("test_category"))
        if native.get("subset") not in (None, ""):
            metadata["subset"] = native.get("subset")
        return metadata
    return metadata


def materialize_runtime_pool_row(row: dict[str, Any]) -> dict[str, Any] | None:
    materialized = dict(row)
    metadata = dict(materialized.get("metadata") or {})
    prompt = _coerce_prompt_to_messages(materialized.get("prompt"))

    raw_tools = materialized.get("tools") or []
    tools = list(raw_tools) if isinstance(raw_tools, list) else []
    materialized["tools"] = tools
    metadata.pop("tools", None)

    dataset_name = str(materialized.get("dataset_name") or metadata.get("dataset_name") or "").strip()
    domain = str(materialized.get("domain") or metadata.get("domain") or "").strip()
    record_id = materialized.get("record_id", metadata.get("record_id"))
    supervision_family = str(materialized.get("supervision_family") or "").strip()
    native = materialized.get("native")
    if dataset_name in BENCHMARK_NATIVE_DATASETS and isinstance(native, dict):
        metadata = _runtime_metadata_from_native(dataset_name, domain, record_id, native, metadata)
        materialized["dataset_name"] = dataset_name
        materialized["domain"] = domain
        materialized["record_id"] = record_id
    elif supervision_family and isinstance(native, dict):
        metadata = _runtime_metadata_from_train_native(
            dataset_name,
            domain,
            record_id,
            supervision_family,
            native,
            metadata,
        )
        if metadata is None:
            return None

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
