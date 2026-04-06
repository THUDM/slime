from __future__ import annotations

import ast
import json
from hashlib import sha1
from typing import Any


BENCHMARK_DATASETS = {
    "bfcl_v3",
    "bfcl_v3_multi_turn_base",
    "ifeval",
    "ifbench_test",
    "jsonschemabench",
    "gpqa",
    "mmlu_pro",
}

TRAIN_NATIVE_SUPERVISION_FAMILIES = {
    "function_call_single",
}


STEM_SYSTEM_PROMPTS = [
    "You are a careful STEM reasoning assistant. Solve the problem step by step. If options are provided, end with the best option and a concise justification.",
    "You are an expert science and quantitative reasoning assistant. Reason carefully, avoid unsupported assumptions, and give the final answer clearly. If this is multiple choice, state the correct option explicitly.",
]

API_CALL_CODEGEN_SYSTEM_PROMPTS = [
    "You are an API-calling code assistant. Return exactly one Python API call expression that solves the request. Do not add explanation, markdown, or extra text.",
    "You write Python API call completions. Output only the single best Python API call expression for the request, with valid arguments and no surrounding commentary.",
]

FUNCTION_CALL_SINGLE_SYSTEM_PROMPTS = [
    "You are a function-calling assistant. Choose the correct function, keep arguments valid for the provided schema, and do not fabricate tool results.",
    "You can call the provided functions. Emit the best tool call with schema-valid JSON arguments when a tool is needed, and avoid extra narrative.",
]

NEXT_ACTION_TOOL_CALL_SYSTEM_PROMPTS = [
    "You are continuing an agent workflow. Produce only the next tool call that best advances the task using the current conversation state and available tools.",
    "You are deciding the next action in a tool-using trace. Return the next valid tool call only, without a final answer or extra commentary.",
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
    "api_call_codegen": API_CALL_CODEGEN_SYSTEM_PROMPTS,
    "function_call_single": FUNCTION_CALL_SINGLE_SYSTEM_PROMPTS,
    "next_action_tool_call": NEXT_ACTION_TOOL_CALL_SYSTEM_PROMPTS,
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
    prompt_family = str(metadata.get("prompt_family") or "").strip().lower()
    reward_type = str(metadata.get("reward_type") or "").strip().lower()
    dataset_name = str(metadata.get("dataset_name") or "").strip().lower()
    domain = str(metadata.get("domain") or "").strip().lower()

    if prompt_family in PROMPT_FAMILY_TO_OPTIONS:
        return prompt_family
    if reward_type in {"instruction_following_soft", "instruction_following_strict", "ifeval"}:
        return "instruction_following"
    if dataset_name in {"ifbench_test", "ifeval"} or domain == "ifrl":
        return "instruction_following"
    if dataset_name in {"bfcl_v3", "bfcl_v3_multi_turn_base"}:
        return "tool"
    if dataset_name in {"gpqa", "mmlu_pro"}:
        return "stem"
    if dataset_name == "jsonschemabench":
        return "structured"
    if reward_type == "structured_json_schema":
        return "structured"
    if reward_type == "stem_mcqa" or domain == "stem":
        return "stem"
    if reward_type == "api_call_text":
        return "api_call_codegen"
    if reward_type == "function_call_single":
        return "function_call_single"
    if reward_type in {
        "tool_call_soft",
        "tool_call_strict",
        "tool_call",
        "tool_selection_strict",
    } or domain == "tool":
        return "tool"
    if domain == "structured":
        return "structured"
    return None


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _ensure_list(value: Any) -> list[Any]:
    parsed = _parse_json_like(value)
    return parsed if isinstance(parsed, list) else []


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _normalize_call_arguments(value: Any) -> Any:
    parsed = _parse_json_like(value)
    return parsed if parsed not in (None, "") else {}


def _normalize_ground_truth_call(name: Any, arguments: Any) -> dict[str, Any]:
    call_name = str(name or "").strip()
    args = _normalize_call_arguments(arguments)
    return {
        "name": call_name,
        "arguments": args,
        "function": {
            "name": call_name,
            "arguments": args,
        },
    }


def _extract_ground_truth_from_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for call in _ensure_list(tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            normalized.append(_normalize_ground_truth_call(function.get("name"), function.get("arguments")))
            continue
        normalized.append(_normalize_ground_truth_call(call.get("name"), call.get("arguments")))
    return [call for call in normalized if call["name"]]


def _messages_before_first_tool_call(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role == "assistant" and message.get("tool_calls"):
            break
        if role == "tool":
            break
        prompt.append({"role": role or "user", "content": message.get("content") or ""})
    return prompt


def _extract_python_call_ground_truth(api_call: Any) -> list[dict[str, Any]]:
    text = str(api_call or "").strip()
    if not text:
        return []
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return []
    if not isinstance(node, ast.Call):
        return []

    def _name(expr: ast.AST) -> str:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            base = _name(expr.value)
            return f"{base}.{expr.attr}" if base else expr.attr
        return ""

    def _literal(expr: ast.AST) -> Any:
        try:
            return ast.literal_eval(expr)
        except Exception:
            return ast.unparse(expr)

    arguments: dict[str, Any] = {}
    if node.args:
        arguments["_args"] = [_literal(arg) for arg in node.args]
    for kw in node.keywords:
        if kw.arg:
            arguments[kw.arg] = _literal(kw.value)
    call_name = _name(node.func)
    return [_normalize_ground_truth_call(call_name, arguments)] if call_name else []


def _runtime_metadata_from_train_row(
    row: dict[str, Any],
    dataset_name: str,
    domain: str,
    record_id: Any,
    supervision_family: str,
    base_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    metadata = {
        "dataset_name": dataset_name,
        "domain": domain,
        "record_id": record_id,
        **base_metadata,
        "supervision_family": supervision_family,
    }
    metadata.pop("tools", None)
    source_fields = row.get("source_fields")
    if isinstance(source_fields, dict):
        metadata["source_fields"] = source_fields
    source_record_fields = row.get("source_record_fields")
    if isinstance(source_record_fields, dict):
        metadata["source_record_fields"] = source_record_fields

    if supervision_family == "function_call_single":
        ground_truth = row.get("ground_truth") or metadata.get("ground_truth") or []
        if not ground_truth:
            ground_truth = _extract_ground_truth_from_tool_calls(
                next(
                    (
                        message.get("tool_calls")
                        for message in _ensure_list(row.get("messages"))
                        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls")
                    ),
                    [],
                )
            )
        if not ground_truth and row.get("api_call") not in (None, ""):
            ground_truth = _extract_python_call_ground_truth(row.get("api_call"))
        if not ground_truth and metadata.get("raw_api_call") not in (None, ""):
            ground_truth = _extract_python_call_ground_truth(metadata.get("raw_api_call"))
        if not ground_truth:
            return None
        reward_type = str(metadata.get("reward_type") or "").strip()
        if reward_type not in {"api_call_text", "function_call_single"}:
            reward_type = "api_call_text" if row.get("api_call") not in (None, "") else "function_call_single"
        metadata["reward_type"] = reward_type
        metadata["ground_truth"] = ground_truth
        raw_api_call = row.get("api_call")
        if raw_api_call in (None, ""):
            raw_api_call = metadata.get("raw_api_call")
        if raw_api_call not in (None, ""):
            metadata["raw_api_call"] = str(raw_api_call)
        if row.get("provider") not in (None, ""):
            metadata["provider"] = row.get("provider")
        return metadata

    return metadata


def _runtime_metadata_from_benchmark_row(
    row: dict[str, Any],
    dataset_name: str,
    domain: str,
    record_id: Any,
    base_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "dataset_name": dataset_name,
        "domain": domain,
        "record_id": record_id,
        **base_metadata,
    }
    metadata.pop("tools", None)

    if dataset_name == "ifbench_test":
        metadata["reward_type"] = "instruction_following_soft"
        metadata["prompt_text"] = str(row.get("prompt_text") or "")
        metadata["instruction_id_list"] = list(row.get("instruction_id_list") or [])
        metadata["kwargs"] = list(row.get("kwargs") or [])
        return metadata
    if dataset_name == "ifeval":
        metadata["reward_type"] = "instruction_following_strict"
        metadata["prompt_text"] = str(row.get("prompt_text") or "")
        metadata["instruction_id_list"] = list(row.get("instruction_id_list") or [])
        metadata["kwargs"] = list(row.get("kwargs") or [])
        return metadata
    if dataset_name == "jsonschemabench":
        metadata["reward_type"] = "structured_json_schema"
        schema = row.get("schema")
        metadata["schema"] = schema if isinstance(schema, dict) else {}
        return metadata
    if dataset_name in {"gpqa", "mmlu_pro"}:
        metadata["reward_type"] = "stem_mcqa"
        answer = row.get("answer", row.get("label", row.get("Correct Answer", "")))
        metadata["answer"] = "" if answer is None else str(answer).strip().upper()
        return metadata
    if dataset_name in {"bfcl_v3", "bfcl_v3_multi_turn_base"}:
        metadata["reward_type"] = "bfcl_official"
        metadata["official_eval_name"] = "bfcl"
        if row.get("test_category") not in (None, ""):
            metadata["test_category"] = str(row.get("test_category"))
        if row.get("subset") not in (None, ""):
            metadata["subset"] = row.get("subset")
        return metadata
    return metadata


def _runtime_metadata_from_row_fields(
    row: dict[str, Any],
    dataset_name: str,
    domain: str,
    record_id: Any,
    base_metadata: dict[str, Any],
) -> dict[str, Any]:
    metadata = {**base_metadata}
    if dataset_name:
        metadata["dataset_name"] = dataset_name
    if domain:
        metadata["domain"] = domain
    if record_id not in (None, ""):
        metadata["record_id"] = record_id
    metadata.pop("tools", None)

    reward_type = str(metadata.get("reward_type") or "").strip().lower()

    if reward_type in {"instruction_following_soft", "instruction_following_strict"}:
        prompt_text = row.get("prompt_text")
        if prompt_text in (None, "") and (
            row.get("instruction_id_list") not in (None, [])
            or row.get("kwargs") not in (None, [])
            or dataset_name in {"ifeval", "ifbench_test"}
        ):
            prompt = _coerce_prompt_to_messages(row.get("prompt"))
            prompt_text = prompt[0].get("content", "") if len(prompt) == 1 else ""
        if prompt_text not in (None, ""):
            metadata["prompt_text"] = str(prompt_text or "")
        if row.get("instruction_id_list") not in (None, []):
            metadata["instruction_id_list"] = list(row.get("instruction_id_list") or [])
        if row.get("kwargs") not in (None, []):
            metadata["kwargs"] = list(row.get("kwargs") or [])
        return metadata

    if reward_type == "structured_json_schema":
        schema = row.get("schema")
        if isinstance(schema, dict):
            metadata["schema"] = schema
        return metadata

    if reward_type == "stem_mcqa":
        if "answer" in row or "label" in row:
            answer = row.get("answer", row.get("label", ""))
            metadata["answer"] = "" if answer is None else str(answer).strip().upper()
        return metadata

    return metadata


def materialize_runtime_pool_row(row: dict[str, Any]) -> dict[str, Any] | None:
    materialized = dict(row)
    materialized.pop("native", None)
    metadata = dict(materialized.get("metadata") or {})
    prompt = _coerce_prompt_to_messages(materialized.get("prompt"))

    raw_tools = _parse_json_like(materialized.get("tools")) or []
    tools = list(raw_tools) if isinstance(raw_tools, list) else []
    materialized["tools"] = tools
    materialized.setdefault("label", "")
    metadata.pop("tools", None)

    dataset_name = str(materialized.get("dataset_name") or metadata.get("dataset_name") or "").strip()
    domain = str(materialized.get("domain") or metadata.get("domain") or "").strip()
    record_id = materialized.get("record_id", metadata.get("record_id"))
    supervision_family = str(materialized.get("supervision_family") or "").strip()

    if not prompt and dataset_name.startswith("apibench_"):
        prompt = [{"role": "user", "content": str(materialized.get("code") or "").strip()}]
    elif not prompt and dataset_name == "xlam_function_calling_60k":
        prompt = _messages_before_first_tool_call(_ensure_list(materialized.get("messages")))
    elif not prompt and supervision_family == "function_call_single":
        prompt = _messages_before_first_tool_call(_ensure_list(materialized.get("messages")))

    if dataset_name in BENCHMARK_DATASETS:
        metadata = _runtime_metadata_from_benchmark_row(materialized, dataset_name, domain, record_id, metadata)
        materialized["dataset_name"] = dataset_name
        materialized["domain"] = domain
        materialized["record_id"] = record_id
    elif supervision_family:
        metadata = _runtime_metadata_from_train_row(
            materialized,
            dataset_name,
            domain,
            record_id,
            supervision_family,
            metadata,
        )
        if metadata is None:
            return None
    else:
        metadata = _runtime_metadata_from_row_fields(materialized, dataset_name, domain, record_id, metadata)

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
