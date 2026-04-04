#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

sys.path.append(str(SCRIPT_DIR / "multidomain_v1"))
import prepare_multidomain_v1_data as mv1

V0_SCRIPT = SCRIPT_DIR / "multidomain_v0" / "prepare_mixed_domain_data.py"
spec = importlib.util.spec_from_file_location("prepare_mixed_domain_data_v0", V0_SCRIPT)
mv0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mv0
spec.loader.exec_module(mv0)

normalize_message = mv1.normalize_message
normalize_json_like_value = mv1.normalize_json_like_value
normalize_tool_definitions = mv1.normalize_tool_definitions
normalize_ground_truth_calls = mv1.normalize_ground_truth_calls
build_choice_prompt = mv1.build_choice_prompt

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
    "agent_function_calling_open_dataset": "agent_trace_call_recovery",
    "apibench": "function_call_single",
    "apigen_mt_5k": "function_call_verified_multi_turn",
    "toolbench_v1": "tool_use_trajectory",
    "xlam_function_calling_60k": "function_call_single",
}

EVAL_ONLY_DATASETS = {
    "bfcl_v3",
    "bfcl_v3_multi_turn_base",
    "toolbench_v1_benchmark",
    "ifeval",
    "ifbench_test",
    "gpqa",
    "mmlu_pro",
}


def _resolve_avalanche_root(project_root: Path, legacy_root: Path, env: dict[str, str]) -> Path:
    env_root = env.get("AVALANCHE_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser()
        if candidate.exists():
            return candidate.resolve()
    if legacy_root.exists():
        return legacy_root.resolve()
    for candidate in (project_root, *project_root.parents):
        if (candidate / "data" / "open_data").exists():
            return candidate.resolve()
    return project_root.resolve()


AVALANCHE_ROOT = _resolve_avalanche_root(
    project_root=PROJECT_ROOT,
    legacy_root=Path("/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche"),
    env=os.environ,
)


def dump_pool(samples: Iterable[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=out_path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    tmp_path.replace(out_path)
    print(f"✅ Saved {count} samples to {out_path}")


def _extract_record_id(row: dict[str, Any]) -> Any:
    for key in ("id", "trace_id", "session_id", "question_id", "unique_id", "uuid", "key", "Record ID"):
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def _collect_source_fields(row: dict[str, Any], *, excluded: set[str]) -> dict[str, Any]:
    source_fields: dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded or value in (None, ""):
            continue
        source_fields[key] = normalize_json_like_value(value)
    return source_fields


def _normalize_tool_definitions_for_pool(raw_tools: Any) -> list[dict[str, Any]]:
    payload = normalize_json_like_value(raw_tools)
    if not isinstance(payload, list):
        return []
    normalized: list[dict[str, Any]] = []
    for tool in payload:
        if not isinstance(tool, dict):
            continue
        copied = dict(tool)
        function = copied.get("function")
        if isinstance(function, dict):
            copied["function"] = dict(function)
            parameters = normalize_json_like_value(function.get("parameters"))
            if isinstance(parameters, dict):
                copied["function"]["parameters"] = parameters
        normalized.append(copied)
    return normalized


def _normalize_tool_call_entry(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {}
    payload = normalize_json_like_value(call)
    if not isinstance(payload, dict):
        payload = dict(call)
    function_payload = payload.get("function")
    name = ""
    arguments: Any = {}
    if isinstance(function_payload, dict):
        name = str(function_payload.get("name") or payload.get("name") or "").strip()
        arguments = function_payload.get("arguments", payload.get("arguments", {}))
    else:
        name = str(payload.get("name") or "").strip()
        arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    normalized: dict[str, Any] = {
        "name": name,
        "arguments": arguments,
        "function": {"name": name, "arguments": arguments},
    }
    if payload.get("id") not in (None, ""):
        normalized["id"] = str(payload.get("id"))
    if payload.get("type") not in (None, ""):
        normalized["type"] = str(payload.get("type"))
    return normalized


def _normalize_tool_calls_for_pool(raw_calls: Any) -> list[dict[str, Any]]:
    payload = normalize_json_like_value(raw_calls)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    normalized = [_normalize_tool_call_entry(call) for call in payload]
    return [call for call in normalized if call.get("name")]


def _normalize_prompt_message_for_pool(message: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_message(message)
    content = message.get("content")
    if isinstance(content, list):
        normalized["content"] = content
    if message.get("name") not in (None, ""):
        normalized["name"] = str(message.get("name"))
    if message.get("tool_call_id") not in (None, ""):
        normalized["tool_call_id"] = str(message.get("tool_call_id"))
    if "tool_calls" in message:
        normalized["tool_calls"] = _normalize_tool_calls_for_pool(message.get("tool_calls"))
    return normalized


def _normalize_prompt_for_pool(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        text = prompt.strip()
        return [{"role": "user", "content": text}] if text else []
    if not isinstance(prompt, list):
        return []
    normalized: list[dict[str, Any]] = []
    for message in prompt:
        if isinstance(message, dict):
            normalized.append(_normalize_prompt_message_for_pool(message))
        elif message not in (None, ""):
            normalized.append({"role": "user", "content": str(message)})
    return normalized


def _with_pool_metadata(sample: dict[str, Any], *, dataset_format: str) -> dict[str, Any]:
    sample = dict(sample)
    metadata = dict(sample.get("metadata") or {})
    metadata.setdefault("dataset_name", dataset_format)
    metadata.setdefault("domain", mv1.dataset_domain(dataset_format))
    metadata.pop("tools", None)
    sample["prompt"] = _normalize_prompt_for_pool(sample.get("prompt"))
    sample["tools"] = list(sample.get("tools") or [])
    sample["label"] = sample.get("label", "")
    sample["dataset_name"] = metadata["dataset_name"]
    sample["domain"] = metadata["domain"]
    sample["record_id"] = metadata.get("record_id")
    sample["metadata"] = metadata
    return sample


def _make_pool_sample(
    *,
    prompt: Any,
    dataset_format: str,
    metadata: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _with_pool_metadata(
        {
            "prompt": prompt,
            "label": "",
            "metadata": metadata,
            "tools": list(tools or []),
        },
        dataset_format=dataset_format,
    )


def _make_native_train_pool_sample(
    *,
    prompt: Any,
    dataset_format: str,
    record_id: Any,
    supervision_family: str,
    native: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_metadata = {
        "dataset_name": dataset_format,
        "domain": mv1.dataset_domain(dataset_format),
        "record_id": record_id,
        **(metadata or {}),
    }
    if isinstance(native.get("source_fields"), dict):
        merged_metadata.setdefault("source_fields", native["source_fields"])
    if isinstance(native.get("source_record_fields"), dict):
        merged_metadata.setdefault("source_record_fields", native["source_record_fields"])
    return _with_pool_metadata(
        {
            "prompt": prompt,
            "label": "",
            "metadata": merged_metadata,
            "tools": list(tools or []),
            "supervision_family": supervision_family,
            "native": normalize_json_like_value(native),
        },
        dataset_format=dataset_format,
    )


def _make_native_eval_pool_sample(
    *,
    prompt: Any,
    dataset_format: str,
    record_id: Any,
    native: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _with_pool_metadata(
        {
            "prompt": prompt,
            "label": "",
            "metadata": {
                "dataset_name": dataset_format,
                "domain": mv1.dataset_domain(dataset_format),
                "record_id": record_id,
            },
            "tools": list(tools or []),
            "native": normalize_json_like_value(native),
        },
        dataset_format=dataset_format,
    )


def _fallback_prompt_from_messages(messages: Any) -> list[dict[str, Any]]:
    return [_normalize_prompt_message_for_pool(message) for message in messages or [] if isinstance(message, dict)]


def _generic_question_prompt(row: dict[str, Any]) -> list[dict[str, Any]]:
    if str(row.get("question") or "").strip():
        choices = row.get("choices")
        if isinstance(choices, list) and choices:
            labeled = [(chr(65 + index), str(choice)) for index, choice in enumerate(choices)]
            return build_choice_prompt(str(row.get("question", "")), labeled)
        return [{"role": "user", "content": str(row.get("question", ""))}]
    if str(row.get("prompt") or "").strip():
        return [{"role": "user", "content": str(row.get("prompt", ""))}]
    return []


def _toolbench_benchmark_allowed_names_for_pool(raw_relevant_apis: Any) -> list[str]:
    return mv1._toolbench_benchmark_allowed_names(raw_relevant_apis)


def _augment_source_fields_from_row(sample: dict[str, Any], row: dict[str, Any], dataset_format: str) -> dict[str, Any]:
    sample = dict(sample)
    metadata = dict(sample.get("metadata") or {})
    if "source_fields" in metadata:
        sample["metadata"] = metadata
        return sample
    excluded = {
        "id",
        "trace_id",
        "session_id",
        "question",
        "choices",
        "answer",
        "prompt",
        "messages",
        "tools",
        "conversations",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
        "Question",
        "Record ID",
        "instruction_id_list",
        "kwargs",
        "json_schema",
        "instruction",
        "key",
    }
    if dataset_format == "scienceqa":
        excluded.update({"image"})
    source_fields = _collect_source_fields(row, excluded=excluded)
    if source_fields:
        metadata["source_fields"] = source_fields
    sample["metadata"] = metadata
    return sample


def _agent_html_tool_calls(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    names = re.findall(r"Call Tool\s+([A-Za-z0-9_.-]+)", text, flags=re.IGNORECASE)
    payloads = re.findall(r'<div class="div_tool_call_json">\s*(\{.*?\})\s*</div>', text, flags=re.DOTALL)
    recovered: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        try:
            arguments = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(arguments, dict):
            continue
        name = names[index] if index < len(names) else ""
        if not name:
            continue
        recovered.append(
            {
                "name": name,
                "arguments": arguments,
                "function": {"name": name, "arguments": arguments},
            }
        )
    return recovered


def _recover_agent_ground_truth(record: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None, int | None]:
    ground_truth = _normalize_tool_calls_for_pool(record.get("tool_calls"))
    if ground_truth:
        return ground_truth, "record_tool_calls", None
    messages = record.get("messages") or []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or str(message.get("role") or "").lower() != "assistant":
            continue
        ground_truth = _normalize_tool_calls_for_pool(message.get("tool_calls"))
        if ground_truth:
            return ground_truth, "assistant_tool_calls", index
        recovered = _agent_html_tool_calls(str(message.get("content") or ""))
        if recovered:
            return recovered, "html_trace", index
    return [], None, None


def _extract_agent_prompt_prefix(messages: Any, stop_index: int | None) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    upper = stop_index if stop_index is not None else len(messages)
    return [_normalize_prompt_message_for_pool(message) for message in messages[:upper] if isinstance(message, dict)]


def _convert_apigen_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    conversations = row.get("conversations") or []
    history: list[dict[str, Any]] = []
    if row.get("system"):
        history.append({"role": "system", "content": str(row.get("system"))})
    source_fields = _collect_source_fields(row, excluded={"conversations", "tools", "system"})
    samples: list[dict[str, Any]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("from", "")).lower()
        value = turn.get("value", "")
        text = "" if value is None else str(value)
        if speaker in {"human", "user"}:
            history.append({"role": "user", "content": text})
            continue
        if speaker in {"gpt", "assistant"}:
            history.append({"role": "assistant", "content": text})
            continue
        if speaker == "observation":
            history.append({"role": "tool", "content": text})
            continue
        if speaker != "function_call":
            continue
        ground_truth = _normalize_tool_calls_for_pool(value)
        if not ground_truth:
            continue
        samples.append(
            _make_native_train_pool_sample(
                prompt=list(history),
                dataset_format="apigen_mt_5k",
                record_id=_extract_record_id(row),
                supervision_family=TRAIN_NATIVE_SUPERVISION_FAMILIES["apigen_mt_5k"],
                native={"ground_truth": ground_truth, "source_fields": source_fields},
                tools=tools,
            )
        )
    return samples


def _convert_xlam_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or []
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    source_fields = _collect_source_fields(row, excluded={"messages", "tools"})
    samples: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or str(message.get("role", "")) != "assistant":
            continue
        ground_truth = _normalize_tool_calls_for_pool(message.get("tool_calls"))
        if not ground_truth:
            continue
        prompt = [_normalize_prompt_message_for_pool(previous) for previous in messages[:index] if isinstance(previous, dict)]
        samples.append(
            _make_native_train_pool_sample(
                prompt=prompt,
                dataset_format="xlam_function_calling_60k",
                record_id=(row.get("extra") or {}).get("id"),
                supervision_family=TRAIN_NATIVE_SUPERVISION_FAMILIES["xlam_function_calling_60k"],
                native={
                    "ground_truth": ground_truth,
                    "message_index": index,
                    "source_fields": source_fields,
                },
                tools=tools,
            )
        )
    return samples


def _convert_toolbench_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    conversations = mv1._normalize_toolbench_conversations(row.get("conversations") or [])
    source_fields = _collect_source_fields(row, excluded={"conversations", "tools"})
    history: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("from", "")).lower()
        text = "" if turn.get("value") is None else str(turn.get("value"))
        if speaker == "system":
            history.append({"role": "system", "content": text})
            continue
        if speaker in {"human", "user"}:
            history.append({"role": "user", "content": text})
            continue
        if speaker in {"function", "tool", "observation"}:
            history.append({"role": "tool", "content": text})
            continue
        if speaker not in {"assistant", "gpt"}:
            continue
        ground_truth = normalize_ground_truth_calls(mv1._parse_tool_calls_from_xml(text))
        if not ground_truth:
            ground_truth = normalize_ground_truth_calls(mv1._parse_toolbench_action_block(text))
        if not ground_truth:
            history.append({"role": "assistant", "content": text})
            continue
        sample_tools = tools or mv1._build_apibench_tools(ground_truth)
        samples.append(
            _make_native_train_pool_sample(
                prompt=list(history),
                dataset_format="toolbench_v1",
                record_id=row.get("id"),
                supervision_family=TRAIN_NATIVE_SUPERVISION_FAMILIES["toolbench_v1"],
                native={
                    "ground_truth": ground_truth,
                    "assistant_reference": text,
                    "source_fields": source_fields,
                },
                tools=sample_tools,
            )
        )
    return samples


def _convert_toolbench_benchmark_row_for_pool(row: dict[str, Any], *, native_eval_contract: bool = False) -> list[dict[str, Any]]:
    tools = mv1._toolbench_benchmark_tools(row.get("api_list"))
    query = str(row.get("query") or "").strip()
    if native_eval_contract:
        return [
            _make_native_eval_pool_sample(
                prompt=[{"role": "user", "content": query}] if query else [],
                dataset_format="toolbench_v1_benchmark",
                record_id=row.get("query_id"),
                native=row,
                tools=tools,
            )
        ]
    return [
        _make_pool_sample(
            prompt=[{"role": "user", "content": query}] if query else [],
            dataset_format="toolbench_v1_benchmark",
            metadata={
                "dataset_name": "toolbench_v1_benchmark",
                "domain": "tool",
                "record_id": row.get("query_id"),
                "reward_type": "tool_selection_strict",
                "allowed_tool_names": _toolbench_benchmark_allowed_names_for_pool(row.get("relevant_apis")),
            },
            tools=tools,
        )
    ]


def _convert_agent_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace_id = row.get("trace_id") or row.get("session_id")
    row_source_fields = _collect_source_fields(row, excluded={"trace_id", "session_id", "function_calls", "messages", "tools"})
    samples: list[dict[str, Any]] = []
    for record in row.get("function_calls") or []:
        if not isinstance(record, dict):
            continue
        tools = _normalize_tool_definitions_for_pool(record.get("tools") or [])
        ground_truth, recovery_source, target_message_index = _recover_agent_ground_truth(record)
        if not ground_truth:
            continue
        messages = record.get("messages") or []
        prompt = _extract_agent_prompt_prefix(messages, target_message_index)
        if not prompt:
            prompt = _fallback_prompt_from_messages(messages)
        record_source_fields = _collect_source_fields(record, excluded={"messages", "tool_calls", "tools"})
        samples.append(
            _make_native_train_pool_sample(
                prompt=prompt,
                dataset_format="agent_function_calling_open_dataset",
                record_id=trace_id,
                supervision_family=TRAIN_NATIVE_SUPERVISION_FAMILIES["agent_function_calling_open_dataset"],
                native={
                    "ground_truth": ground_truth,
                    "recovery_source": recovery_source,
                    "target_message_index": target_message_index,
                    "source_fields": row_source_fields,
                    "source_record_fields": record_source_fields,
                },
                metadata={
                    "source_fields": row_source_fields,
                    "source_record_fields": record_source_fields,
                },
                tools=tools,
            )
        )
    return samples


def _convert_bfcl_v3_row_for_pool(row: dict[str, Any], *, native_eval_contract: bool = False) -> list[dict[str, Any]]:
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    prompt = mv1._load_bfcl_prompt(row.get("turns"))
    if native_eval_contract:
        return [
            _make_native_eval_pool_sample(
                prompt=prompt,
                dataset_format="bfcl_v3",
                record_id=row.get("id"),
                native=row,
                tools=tools,
            )
        ]
    ground_truth = mv1._normalize_bfcl_ground_truth(row.get("ground_truth"))
    return [
        _make_pool_sample(
            prompt=prompt,
            dataset_format="bfcl_v3",
            metadata={
                "dataset_name": "bfcl_v3",
                "domain": "tool",
                "record_id": row.get("id"),
                "reward_type": "tool_call_soft",
                "ground_truth": ground_truth,
                "subset": row.get("subset"),
                "test_category": row.get("test_category"),
            },
            tools=tools,
        )
    ]


def _convert_bfcl_multi_turn_row_for_pool(row: dict[str, Any], *, native_eval_contract: bool = False) -> list[dict[str, Any]]:
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    prompt = [_normalize_prompt_message_for_pool(message) for message in row.get("messages") or [] if isinstance(message, dict)]
    if native_eval_contract:
        return [
            _make_native_eval_pool_sample(
                prompt=prompt,
                dataset_format="bfcl_v3_multi_turn_base",
                record_id=row.get("id"),
                native=row,
                tools=tools,
            )
        ]
    ground_truth = mv1._normalize_bfcl_ground_truth(row.get("ground_truth"))
    return [
        _make_pool_sample(
            prompt=prompt,
            dataset_format="bfcl_v3_multi_turn_base",
            metadata={
                "dataset_name": "bfcl_v3_multi_turn_base",
                "domain": "tool",
                "record_id": row.get("id"),
                "reward_type": "tool_call_soft",
                "ground_truth": ground_truth,
            },
            tools=tools,
        )
    ]


def _convert_apibench_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_text = mv1._extract_instruction_from_apibench(str(row.get("code", "")))
    try:
        tools, ground_truth, parse_mode = mv1._canonicalize_apibench_api_call(str(row.get("api_call", "")))
    except mv1.SkippableRowError:
        return []
    if not ground_truth:
        return []
    source_fields = _collect_source_fields(row, excluded={"code", "api_call"})
    return [
        _make_native_train_pool_sample(
            prompt=[{"role": "user", "content": prompt_text}] if prompt_text else [],
            dataset_format="apibench",
            record_id=_extract_record_id(row),
            supervision_family=TRAIN_NATIVE_SUPERVISION_FAMILIES["apibench"],
            native={
                "ground_truth": ground_truth,
                "provider": row.get("provider"),
                "parse_mode": parse_mode,
                "raw_api_call": row.get("api_call"),
                "accepted_for_eval": True,
                "source_fields": source_fields,
            },
            tools=tools,
        )
    ]


def _convert_native_eval_row_for_pool(row: dict[str, Any], dataset_format: str) -> list[dict[str, Any]]:
    if dataset_format == "ifbench_test":
        prompt_text = str(row.get("prompt") or "").strip()
        return [
            _make_native_eval_pool_sample(
                prompt=[{"role": "user", "content": prompt_text}] if prompt_text else [],
                dataset_format=dataset_format,
                record_id=row.get("key"),
                native=row,
            )
        ]
    if dataset_format == "ifeval":
        prompt_text = str(row.get("prompt") or "").strip()
        return [
            _make_native_eval_pool_sample(
                prompt=[{"role": "user", "content": prompt_text}] if prompt_text else [],
                dataset_format=dataset_format,
                record_id=_extract_record_id(row),
                native=row,
            )
        ]
    try:
        converted = mv1.convert_row(row, dataset_format)
    except mv1.SkippableRowError:
        converted = []
    if not converted:
        return []
    first = converted[0]
    return [
        _make_native_eval_pool_sample(
            prompt=first.get("prompt") or [],
            dataset_format=dataset_format,
            record_id=(first.get("metadata") or {}).get("record_id", _extract_record_id(row)),
            native=row,
            tools=first.get("tools") or [],
        )
    ]


def _generic_pool_fallback(row: dict[str, Any], dataset_format: str) -> list[dict[str, Any]]:
    domain = mv1.dataset_domain(dataset_format)
    metadata: dict[str, Any] = {
        "domain": domain,
        "dataset_name": dataset_format,
        "record_id": _extract_record_id(row),
    }
    prompt: Any = []

    if dataset_format in {"ifeval", "ifbench_test"}:
        prompt_text = str(row.get("prompt") or "").strip()
        prompt = [{"role": "user", "content": prompt_text}] if prompt_text else []
        metadata["reward_type"] = "instruction_following_strict" if dataset_format == "ifeval" else "instruction_following_soft"
        metadata["instruction_id_list"] = row.get("instruction_id_list") or []
        metadata["kwargs"] = row.get("kwargs") or []
        metadata["prompt_text"] = prompt_text
        source_fields = _collect_source_fields(row, excluded={"key", "prompt", "instruction_id_list", "kwargs"})
        if source_fields:
            metadata["source_fields"] = source_fields
        return [_make_pool_sample(prompt=prompt, dataset_format=dataset_format, metadata=metadata)]

    if dataset_format == "gpqa":
        converted = mv1.convert_gpqa_row(row)
        if converted:
            return [
                _augment_source_fields_from_row(
                    _with_pool_metadata(sample, dataset_format=dataset_format),
                    row,
                    dataset_format,
                )
                for sample in converted
            ]
        return []

    try:
        converted = mv1.convert_row(row, dataset_format) if dataset_format != "apibench" else []
    except mv1.SkippableRowError:
        converted = []
    if converted:
        return [
            _augment_source_fields_from_row(
                _with_pool_metadata(sample, dataset_format=dataset_format),
                row,
                dataset_format,
            )
            for sample in converted
        ]

    prompt = _generic_question_prompt(row)
    metadata["reward_type"] = ""
    return [_make_pool_sample(prompt=prompt, dataset_format=dataset_format, metadata=metadata)]


def convert_row_for_pool(row: dict[str, Any], dataset_format: str, native_eval_contract: bool = False) -> list[dict[str, Any]]:
    if native_eval_contract and dataset_format in BENCHMARK_NATIVE_DATASETS:
        if dataset_format == "toolbench_v1_benchmark":
            return _convert_toolbench_benchmark_row_for_pool(row, native_eval_contract=True)
        if dataset_format == "bfcl_v3":
            return _convert_bfcl_v3_row_for_pool(row, native_eval_contract=True)
        if dataset_format == "bfcl_v3_multi_turn_base":
            return _convert_bfcl_multi_turn_row_for_pool(row, native_eval_contract=True)
        return _convert_native_eval_row_for_pool(row, dataset_format)

    if dataset_format == "apigen_mt_5k":
        return _convert_apigen_row_for_pool(row)
    if dataset_format == "xlam_function_calling_60k":
        return _convert_xlam_row_for_pool(row)
    if dataset_format == "toolbench_v1":
        return _convert_toolbench_row_for_pool(row)
    if dataset_format == "toolbench_v1_benchmark":
        return _convert_toolbench_benchmark_row_for_pool(row)
    if dataset_format == "agent_function_calling_open_dataset":
        return _convert_agent_row_for_pool(row)
    if dataset_format == "bfcl_v3":
        return _convert_bfcl_v3_row_for_pool(row)
    if dataset_format == "bfcl_v3_multi_turn_base":
        return _convert_bfcl_multi_turn_row_for_pool(row)
    if dataset_format == "apibench":
        return _convert_apibench_row_for_pool(row)

    return _generic_pool_fallback(row, dataset_format)


def iter_pool_samples(source: Path, dataset_format: str, *, native_eval_contract: bool = False) -> Iterator[dict[str, Any]]:
    for row in mv1.iter_rows(source):
        if not isinstance(row, dict):
            continue
        for sample in convert_row_for_pool(row, dataset_format, native_eval_contract=native_eval_contract):
            yield sample


def _pool_output_paths(domain: str, filename: str) -> list[Path]:
    pool_root = AVALANCHE_ROOT / "data" / "pool" / domain
    nested = [path for path in (pool_root / "train" / filename, pool_root / "eval" / filename) if path.exists()]
    if nested:
        return nested
    root_level = pool_root / filename
    if root_level.exists():
        return [root_level]
    return [root_level]


def _use_native_eval_contract(format_name: str, source: Path) -> bool:
    if format_name in EVAL_ONLY_DATASETS:
        return True
    if format_name == "jsonschemabench":
        source_text = str(source).lower()
        return "test" in source.stem.lower() or "/test-" in source_text or "\\test-" in source_text
    return False


def process_dataset(path_str: str, format_name: str) -> None:
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"⚠️ Source missing: {path}")
        return
    domain = mv1.dataset_domain(format_name)
    native_eval_contract = _use_native_eval_contract(format_name, path)
    split = "eval" if native_eval_contract else "train"
    print(f"Processing {format_name} ({domain}/{split})...")
    samples = list(
        mv0.iter_selected_samples(
            iter_pool_samples(path, format_name, native_eval_contract=native_eval_contract),
            0,
            None,
        )
    )
    out_path = AVALANCHE_ROOT / "data" / "pool" / domain / split / f"{format_name}_{path.stem}.jsonl"
    dump_pool(samples, out_path)


def process_ifrl(path_str: str) -> None:
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"⚠️ IF-RL Source missing: {path}")
        return
    print("Processing IF-RL...")
    sys.path.append(str(SCRIPT_DIR / "if_rl"))
    import prepare_ifrl_data as ifrl

    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_text = ifrl.normalize_prompt(row.get("prompt", ""))
            source_fields = _collect_source_fields(
                row,
                excluded={"id", "prompt", "dataset", "agent_ref", "instruction_id_list", "kwargs"},
            )
            metadata = {
                "record_id": row.get("id"),
                "prompt_text": prompt_text,
                "dataset_name": row.get("dataset") or "ifrl",
                "domain": "ifrl",
                "agent_ref": row.get("agent_ref", ""),
            }
            if row.get("kwargs") is not None:
                metadata["kwargs"] = row.get("kwargs")
            if row.get("instruction_id_list") is not None:
                metadata["instruction_id_list"] = row.get("instruction_id_list")
            if source_fields:
                metadata["source_fields"] = source_fields
            samples.append(
                {
                    "prompt": [{"role": "user", "content": prompt_text}] if prompt_text else [],
                    "label": "",
                    "metadata": metadata,
                    "tools": [],
                }
            )
    out_path = AVALANCHE_ROOT / "data" / "pool" / "ifrl" / f"ifrl_{path.stem}.jsonl"
    dump_pool(samples, out_path)


DATASETS = [
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/agent_function_calling_open_dataset/deepnlp_agent_function_call_202510.json", "agent_function_calling_open_dataset"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/agent_function_calling_open_dataset/deepnlp_agent_function_call_202601.json", "agent_function_calling_open_dataset"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/apigen_mt_5k/apigen-mt_5k.json", "apigen_mt_5k"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/xlam_function_calling_60k/xlam-function-calling-60k.parquet", "xlam_function_calling_60k"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/apibench/huggingface_train.json", "apibench"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/apibench/tensorflow_train.json", "apibench"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/apibench/torchhub_train.json", "apibench"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/data/train-00000-of-00004.parquet", "toolbench_v1"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/data/train-00001-of-00004.parquet", "toolbench_v1"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/data/train-00002-of-00004.parquet", "toolbench_v1"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/data/train-00003-of-00004.parquet", "toolbench_v1"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/bfcl_v3/data/train-00000-of-00001.parquet", "bfcl_v3"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/bfcl_v3_multi_turn_base/data/train-00000-of-00001.parquet", "bfcl_v3_multi_turn_base"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g1_tool-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g1_category-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g1_instruction-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g2_category-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g2_instruction-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/tool_call/toolbench_v1/benchmark/g3_instruction-00000-of-00001.parquet", "toolbench_v1_benchmark"),
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/nemotron_structured_outputs/structured_outputs_251027_nano_v3_sdg_json_train.jsonl", "nemotron_structured_outputs"),
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/ifeval/ifeval_input_data.jsonl", "ifeval"),
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/jsonschemabench/data/train-00000-of-00001.parquet", "jsonschemabench"),
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/jsonschemabench/data/test-00000-of-00001.parquet", "jsonschemabench"),
    (f"{AVALANCHE_ROOT}/data/open_data/structured_output/ifbench_test/data/train-00000-of-00001.parquet", "ifbench_test"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/nemotron_knowledge_mcqa/data/train-00000-of-00004.parquet", "nemotron_knowledge_mcqa"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet", "ai2_arc"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/scienceqa/data/train-00000-of-00001-1028f23e353fbe3e.parquet", "scienceqa"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/openbookqa/main/train-00000-of-00001.parquet", "openbookqa"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/sciq/data/train-00000-of-00001.parquet", "sciq"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/medmcqa/data/train-00000-of-00001.parquet", "medmcqa"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/mmlu_pro/data/test-00000-of-00001.parquet", "mmlu_pro"),
    (f"{AVALANCHE_ROOT}/data/open_data/stem/gpqa/gpqa_main.csv", "gpqa"),
]


if __name__ == "__main__":
    print("🚀 Building pool data from open_data...")
    for path, fmt in DATASETS:
        process_dataset(path, fmt)
    process_ifrl(f"{AVALANCHE_ROOT}/data/raw_data/Nemotron-Cascade-2-RL-data/IF-RL/train.jsonl")
    print("🎉 Done collecting pool data!")
