#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
LEGACY_AVALANCHE_ROOT = Path("/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche")

sys.path.append(str(SCRIPT_DIR / "multidomain_v1"))
import prepare_multidomain_v1_data as mv1

V0_SCRIPT = SCRIPT_DIR / "multidomain_v0" / "prepare_mixed_domain_data.py"
spec = importlib.util.spec_from_file_location("prepare_mixed_domain_data_v0", V0_SCRIPT)
mv0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mv0
spec.loader.exec_module(mv0)


normalize_json_like_value = mv1.normalize_json_like_value
build_choice_prompt = mv1.build_choice_prompt


def _dataset_domain_for_pool(dataset_format: str) -> str:
    if dataset_format == "ifrl":
        return "ifrl"
    return mv1.dataset_domain(dataset_format)


def _resolve_avalanche_root(
    *,
    project_root: Path = PROJECT_ROOT,
    legacy_root: Path = LEGACY_AVALANCHE_ROOT,
    env: dict[str, str] | None = None,
) -> Path:
    env = os.environ if env is None else env
    configured_root = env.get("AVALANCHE_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()
    if legacy_root.exists():
        return legacy_root
    for candidate in (project_root, *project_root.parents):
        if (candidate / "data" / "open_data").exists():
            return candidate
    return project_root


AVALANCHE_ROOT = _resolve_avalanche_root()

SOURCE_FIELD_EXCLUDED_KEYS: dict[str, set[str]] = {
    "agent_function_calling_open_dataset": {"function_calls", "trace_id", "session_id"},
    "ai2_arc": {"question", "choices", "answerKey", "id"},
    "apibench": {"api_call", "provider"},
    "apigen_mt_5k": {"conversations", "system", "tools"},
    "bfcl_v3": {"ground_truth", "id", "tools", "turns"},
    "bfcl_v3_multi_turn_base": {"ground_truth", "id", "messages", "tools"},
    "gpqa": {"Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Question", "Record ID"},
    "ifeval": {"instruction_id_list", "kwargs", "key", "prompt"},
    "ifbench_test": {"instruction_id_list", "kwargs", "key", "prompt"},
    "jsonschemabench": {"instruction", "json_schema", "prompt", "unique_id"},
    "medmcqa": {"cop", "id", "opa", "opb", "opc", "opd", "question"},
    "mmlu_pro": {"answer", "options", "question", "question_id"},
    "openbookqa": {"answerKey", "choices", "id", "question_stem"},
    "scienceqa": {"answer", "choices", "question"},
    "sciq": {"correct_answer", "distractor1", "distractor2", "distractor3", "question"},
    "toolbench_v1": {"conversations", "tools"},
    "toolbench_v1_benchmark": {"api_list", "query", "query_id", "relevant_apis"},
    "xlam_function_calling_60k": {"messages", "tools"},
}


def _normalize_prompt_content_for_pool(content: Any) -> Any:
    if not isinstance(content, list):
        return "" if content is None else str(content)

    normalized_content: list[Any] = []
    for item in content:
        if isinstance(item, str):
            normalized_content.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        normalized_item = dict(item)
        if normalized_item.get("type") == "text":
            normalized_item["text"] = "" if normalized_item.get("text") is None else str(normalized_item.get("text", ""))
        normalized_content.append(normalized_item)
    return normalized_content


def _normalize_tool_call_for_pool(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None

    normalized = dict(call)
    if normalized.get("id") not in (None, ""):
        normalized["id"] = str(normalized["id"])
    if normalized.get("type") not in (None, ""):
        normalized["type"] = str(normalized["type"])

    function = normalized.get("function")
    if isinstance(function, dict):
        normalized_function = dict(function)
        normalized_function["name"] = str(function.get("name", ""))
        normalized_function["arguments"] = _normalize_json_tree(function.get("arguments", {}))
        normalized["function"] = normalized_function
        normalized["name"] = normalized_function["name"]
        normalized["arguments"] = normalized_function["arguments"]
        return normalized

    normalized["name"] = str(normalized.get("name", ""))
    normalized["arguments"] = _normalize_json_tree(normalized.get("arguments", normalized.get("parameters", {})))
    normalized["function"] = {
        "name": normalized["name"],
        "arguments": normalized["arguments"],
    }
    return normalized


def _normalize_ground_truth_calls_for_pool(raw_calls: Any) -> list[dict[str, Any]]:
    raw_calls = normalize_json_like_value(raw_calls)
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    if not isinstance(raw_calls, list):
        return []
    return [call for item in raw_calls if (call := _normalize_tool_call_for_pool(item)) is not None]


def _normalize_prompt_message_for_pool(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    normalized["role"] = str(message.get("role", "user"))
    normalized["content"] = _normalize_prompt_content_for_pool(message.get("content", ""))

    for field in ("name", "tool_call_id", "id"):
        if normalized.get(field) not in (None, ""):
            normalized[field] = str(normalized[field])

    if "tool_calls" in normalized:
        raw_tool_calls = normalized.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            normalized["tool_calls"] = [
                tool_call for item in raw_tool_calls if (tool_call := _normalize_tool_call_for_pool(item)) is not None
            ]
        else:
            normalized["tool_calls"] = []

    if "function_call" in normalized:
        function_call = _normalize_tool_call_for_pool(normalized.get("function_call"))
        if function_call is not None:
            normalized["function_call"] = function_call

    return normalized


def _normalize_tool_definition_for_pool(tool: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(tool)
    if "function" in tool and isinstance(tool["function"], dict):
        function = dict(tool["function"])
        function["name"] = str(function.get("name", ""))
        function["description"] = str(function.get("description", ""))
        function["parameters"] = _normalize_json_tree(function.get("parameters") or {"type": "object", "properties": {}})
        normalized["type"] = str(tool.get("type", "function"))
        normalized["function"] = function
        return normalized

    normalized["type"] = str(tool.get("type", "function"))
    normalized["function"] = {
        "name": str(tool.get("name", "")),
        "description": str(tool.get("description", "")),
        "parameters": _normalize_json_tree(tool.get("parameters") or {"type": "object", "properties": {}}),
    }
    return normalized


def _normalize_tool_definitions_for_pool(raw_tools: Any) -> list[dict[str, Any]]:
    raw_tools = normalize_json_like_value(raw_tools)
    if not isinstance(raw_tools, list):
        return []
    return [_normalize_tool_definition_for_pool(tool) for tool in raw_tools if isinstance(tool, dict)]


def _extract_agent_prompt_for_pool(messages: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt: list[dict[str, Any]] = []
    expected_calls = mv1.normalize_ground_truth_calls(ground_truth)
    for message in messages:
        if not isinstance(message, dict):
            continue
        if mv1.normalize_ground_truth_calls(message.get("tool_calls")) == expected_calls:
            break
        prompt.append(_normalize_prompt_message_for_pool(message))
    return prompt


def _load_bfcl_prompt_for_pool(raw_turns: Any) -> list[dict[str, Any]]:
    turns = normalize_json_like_value(raw_turns)
    if isinstance(turns, list) and turns:
        first = turns[0]
        if isinstance(first, list):
            return [_normalize_prompt_message_for_pool(message) for message in first if isinstance(message, dict)]
        if isinstance(first, dict):
            return [_normalize_prompt_message_for_pool(message) for message in turns if isinstance(message, dict)]
    return []


def _normalize_json_tree(value: Any) -> Any:
    value = normalize_json_like_value(value)
    if isinstance(value, dict):
        return {str(key): _normalize_json_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_tree(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_tree(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _prune_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        pruned = {key: _prune_empty_values(item) for key, item in value.items()}
        return {key: item for key, item in pruned.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        pruned = [_prune_empty_values(item) for item in value]
        return [item for item in pruned if item not in (None, "", [], {})]
    return value


def _source_fields_for_pool(payload: dict[str, Any], *, excluded_keys: set[str] | None = None) -> dict[str, Any]:
    source_fields: dict[str, Any] = {}
    for key, value in payload.items():
        if excluded_keys and key in excluded_keys:
            continue
        normalized = _prune_empty_values(_normalize_json_tree(value))
        if normalized in (None, "", [], {}):
            continue
        source_fields[str(key)] = normalized
    return source_fields


def _normalize_prompt_for_pool(prompt: Any) -> Any:
    if not isinstance(prompt, list):
        return prompt
    return [
        _normalize_prompt_message_for_pool(message) if isinstance(message, dict) else {"role": "user", "content": "" if message is None else str(message)}
        for message in prompt
    ]


def _normalize_pool_sample(
    sample: dict[str, Any],
    *,
    dataset_format: str,
    source_fields: dict[str, Any] | None = None,
    source_record_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = dict(sample)
    normalized["prompt"] = _normalize_prompt_for_pool(normalized.get("prompt"))

    tools = _normalize_tool_definitions_for_pool(normalized.get("tools") or (normalized.get("metadata") or {}).get("tools") or [])
    metadata = _normalize_json_tree(normalized.get("metadata") or {})
    metadata.pop("tools", None)
    if metadata.get("ground_truth") not in (None, "", [], {}):
        metadata["ground_truth"] = _normalize_ground_truth_calls_for_pool(metadata.get("ground_truth"))
    if source_fields:
        metadata["source_fields"] = source_fields
    if source_record_fields:
        metadata["source_record_fields"] = source_record_fields

    normalized["label"] = normalized.get("label", "")
    normalized["metadata"] = metadata
    normalized["tools"] = tools
    return _with_pool_metadata(normalized, dataset_format=dataset_format)


def _default_source_fields(row: dict[str, Any], dataset_format: str) -> dict[str, Any]:
    return _source_fields_for_pool(row, excluded_keys=SOURCE_FIELD_EXCLUDED_KEYS.get(dataset_format))


def dump_pool(samples: Iterable[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp_path, out_path)
    print(f"✅ Saved {count} samples to {out_path}")


def _pool_output_paths(domain: str, filename: str) -> list[Path]:
    domain_root = AVALANCHE_ROOT / "data" / "pool" / domain
    existing = sorted(
        path
        for path in domain_root.rglob(filename)
        if path.is_file() and not path.name.startswith("._")
    )
    nested = [path for path in existing if path.parent != domain_root]
    if nested:
        return nested
    if existing:
        return existing
    return [domain_root / filename]


def _extract_record_id(row: dict[str, Any]) -> Any:
    for key in (
        "id",
        "trace_id",
        "session_id",
        "question_id",
        "unique_id",
        "uuid",
        "key",
        "Record ID",
    ):
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def _with_pool_metadata(sample: dict[str, Any], *, dataset_format: str) -> dict[str, Any]:
    sample = dict(sample)
    metadata = dict(sample.get("metadata") or {})
    metadata.setdefault("dataset_name", dataset_format)
    metadata.setdefault("domain", _dataset_domain_for_pool(dataset_format))
    if metadata.get("domain") == "tool":
        tools = list(sample.get("tools") or [])
        sample["tools"] = tools
        metadata["parser_type"] = "qwen3"
    else:
        sample["tools"] = list(sample.get("tools") or [])
    metadata.pop("tools", None)
    sample["label"] = sample.get("label", "")
    sample["metadata"] = metadata
    return sample


def _make_pool_sample(
    *,
    prompt: Any,
    dataset_format: str,
    metadata: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    source_fields: dict[str, Any] | None = None,
    source_record_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample = {
        "prompt": prompt,
        "label": "",
        "metadata": metadata,
        "tools": list(tools or []),
    }
    return _normalize_pool_sample(
        sample,
        dataset_format=dataset_format,
        source_fields=source_fields,
        source_record_fields=source_record_fields,
    )


def _fallback_prompt_from_messages(messages: Any) -> list[dict[str, Any]]:
    return [_normalize_prompt_message_for_pool(message) for message in messages or [] if isinstance(message, dict)]


def _generic_question_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    if str(row.get("question") or "").strip():
        choices = row.get("choices")
        if isinstance(choices, list) and choices:
            return build_choice_prompt(str(row.get("question", "")), [(chr(65 + index), str(choice)) for index, choice in enumerate(choices)])
        return [{"role": "user", "content": str(row.get("question", ""))}]
    if str(row.get("prompt") or "").strip():
        return [{"role": "user", "content": str(row.get("prompt", ""))}]
    return []


def _convert_apigen_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    conversations = row.get("conversations") or []
    history: list[dict[str, Any]] = []
    source_fields = _default_source_fields(row, "apigen_mt_5k")
    system_prompt = row.get("system")
    if system_prompt:
        history.append({"role": "system", "content": str(system_prompt)})

    samples: list[dict[str, Any]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("from", "")).lower()
        value = turn.get("value", "")
        if speaker in {"human", "user"}:
            history.append({"role": "user", "content": "" if value is None else str(value)})
            continue
        if speaker in {"gpt", "assistant"}:
            history.append({"role": "assistant", "content": "" if value is None else str(value)})
            continue
        if speaker == "observation":
            history.append({"role": "tool", "content": "" if value is None else str(value)})
            continue
        if speaker != "function_call":
            continue

        ground_truth = _normalize_ground_truth_calls_for_pool(value)
        samples.append(
            _make_pool_sample(
                prompt=list(history),
                dataset_format="apigen_mt_5k",
                metadata={
                    "domain": "tool",
                    "dataset_name": "apigen_mt_5k",
                    "reward_type": "tool_call_soft",
                    "ground_truth": ground_truth,
                    "record_id": _extract_record_id(row),
                },
                tools=tools,
                source_fields=source_fields,
            )
        )

    if samples:
        return samples

    return []


def _convert_xlam_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or []
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    source_fields = _default_source_fields(row, "xlam_function_calling_60k")
    samples: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or str(message.get("role", "")) != "assistant":
            continue
        ground_truth = _normalize_ground_truth_calls_for_pool(message.get("tool_calls"))
        if not ground_truth:
            continue
        prompt = [_normalize_prompt_message_for_pool(previous) for previous in messages[:index] if isinstance(previous, dict)]
        samples.append(
            _make_pool_sample(
                prompt=prompt,
                dataset_format="xlam_function_calling_60k",
                metadata={
                    "domain": "tool",
                    "dataset_name": "xlam_function_calling_60k",
                    "reward_type": "tool_call_soft",
                    "ground_truth": ground_truth,
                    "record_id": (row.get("extra") or {}).get("id"),
                },
                tools=tools,
                source_fields=source_fields,
            )
        )
    if samples:
        return samples
    return []


def _convert_toolbench_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    conversations = mv1._normalize_toolbench_conversations(row.get("conversations") or [])
    history: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    source_fields = _default_source_fields(row, "toolbench_v1")

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("from", "")).lower()
        value = turn.get("value", "")
        text = "" if value is None else str(value)

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

        ground_truth = _normalize_ground_truth_calls_for_pool(mv1._parse_tool_calls_from_xml(text))
        if not ground_truth:
            ground_truth = _normalize_ground_truth_calls_for_pool(mv1._parse_toolbench_action_block(text))
        sample_tools = tools or mv1._build_apibench_tools(ground_truth) if ground_truth else tools
        samples.append(
            _make_pool_sample(
                prompt=list(history),
                dataset_format="toolbench_v1",
                metadata={
                    "domain": "tool",
                    "dataset_name": "toolbench_v1",
                    "reward_type": "tool_call_soft",
                    "ground_truth": ground_truth,
                    "record_id": row.get("id"),
                    "assistant_reference": text,
                },
                tools=sample_tools,
                source_fields=source_fields,
            )
        )
        if not ground_truth:
            history.append({"role": "assistant", "content": text})

    if samples:
        return samples
    return []


def _convert_toolbench_benchmark_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = mv1._toolbench_benchmark_tools(row.get("api_list"))
    allowed_tool_names = mv1._toolbench_benchmark_allowed_names(row.get("relevant_apis"))
    query = str(row.get("query") or "").strip()
    source_fields = _default_source_fields(row, "toolbench_v1_benchmark")
    return [
        _make_pool_sample(
            prompt=[{"role": "user", "content": query}] if query else [],
            dataset_format="toolbench_v1_benchmark",
            metadata={
                "domain": "tool",
                "dataset_name": "toolbench_v1_benchmark",
                "reward_type": "tool_selection_strict",
                "allowed_tool_names": allowed_tool_names,
                "record_id": row.get("query_id"),
            },
            tools=tools,
            source_fields=source_fields,
        )
    ]


def _convert_agent_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    trace_id = row.get("trace_id") or row.get("session_id")
    source_fields = _default_source_fields(row, "agent_function_calling_open_dataset")
    for record in row.get("function_calls") or []:
        if not isinstance(record, dict):
            continue
        tools = _normalize_tool_definitions_for_pool(record.get("tools") or [])
        ground_truth = _normalize_ground_truth_calls_for_pool(record.get("tool_calls"))
        messages = record.get("messages") or []
        prompt = _extract_agent_prompt_for_pool(messages, ground_truth) if ground_truth else []
        if not prompt:
            prompt = [_normalize_prompt_message_for_pool(message) for message in messages if isinstance(message, dict)]
        source_record_fields = _source_fields_for_pool(record, excluded_keys={"messages", "tool_calls", "tools"})
        samples.append(
            _make_pool_sample(
                prompt=prompt,
                dataset_format="agent_function_calling_open_dataset",
                metadata={
                    "domain": "tool",
                    "dataset_name": "agent_function_calling_open_dataset",
                    "reward_type": "tool_call_soft",
                    "ground_truth": ground_truth,
                    "record_id": trace_id,
                },
                tools=tools,
                source_fields=source_fields,
                source_record_fields=source_record_fields,
            )
        )
    return samples


def _convert_bfcl_v3_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    ground_truth = mv1._normalize_bfcl_ground_truth(row.get("ground_truth"))
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    prompt = _load_bfcl_prompt_for_pool(row.get("turns"))
    source_fields = _default_source_fields(row, "bfcl_v3")
    return [
        _make_pool_sample(
            prompt=prompt,
            dataset_format="bfcl_v3",
            metadata={
                "domain": "tool",
                "dataset_name": "bfcl_v3",
                "reward_type": "tool_call_soft",
                "ground_truth": ground_truth,
                "record_id": row.get("id"),
                "subset": row.get("subset"),
                "test_category": row.get("test_category"),
            },
            tools=tools,
            source_fields=source_fields,
        )
    ]


def _convert_bfcl_multi_turn_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    ground_truth = mv1._normalize_bfcl_ground_truth(row.get("ground_truth"))
    tools = _normalize_tool_definitions_for_pool(row.get("tools") or [])
    prompt = [_normalize_prompt_message_for_pool(message) for message in row.get("messages") or [] if isinstance(message, dict)]
    source_fields = _default_source_fields(row, "bfcl_v3_multi_turn_base")
    return [
        _make_pool_sample(
            prompt=prompt,
            dataset_format="bfcl_v3_multi_turn_base",
            metadata={
                "domain": "tool",
                "dataset_name": "bfcl_v3_multi_turn_base",
                "reward_type": "tool_call_soft",
                "ground_truth": ground_truth,
                "record_id": row.get("id"),
            },
            tools=tools,
            source_fields=source_fields,
        )
    ]


def _convert_apibench_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_text = mv1._extract_instruction_from_apibench(str(row.get("code", "")))
    source_fields = _default_source_fields(row, "apibench")
    try:
        tools, ground_truth, parse_mode = mv1._canonicalize_apibench_api_call(str(row.get("api_call", "")))
    except mv1.SkippableRowError:
        return []
    return [
        _make_pool_sample(
            prompt=[{"role": "user", "content": prompt_text}] if prompt_text else [],
            dataset_format="apibench",
            metadata={
                "domain": "tool",
                "dataset_name": "apibench",
                "reward_type": "tool_call_soft",
                "ground_truth": ground_truth,
                "provider": row.get("provider"),
                "parse_mode": parse_mode,
                "raw_api_call": row.get("api_call"),
                "accepted_for_eval": bool(ground_truth),
                "record_id": _extract_record_id(row),
            },
            tools=tools,
            source_fields=source_fields,
        )
    ]


def _convert_ifbench_row_for_pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_text = str(row.get("prompt") or "").strip()
    source_fields = _default_source_fields(row, "ifbench_test")
    return [
        _make_pool_sample(
            prompt=[{"role": "user", "content": prompt_text}] if prompt_text else [],
            dataset_format="ifbench_test",
            metadata={
                "domain": "structured",
                "dataset_name": "ifbench_test",
                "reward_type": "instruction_following_soft",
                "record_id": row.get("key"),
                "prompt_text": prompt_text,
                "instruction_id_list": row.get("instruction_id_list") or [],
                "kwargs": row.get("kwargs") or [],
            },
            tools=[],
            source_fields=source_fields,
        )
    ]


def _generic_pool_fallback(row: dict[str, Any], dataset_format: str) -> list[dict[str, Any]]:
    domain = mv1.dataset_domain(dataset_format)
    source_fields = _default_source_fields(row, dataset_format)
    metadata: dict[str, Any] = {
        "domain": domain,
        "dataset_name": dataset_format,
        "record_id": _extract_record_id(row),
    }
    prompt: Any = []
    tools: list[dict[str, Any]] = []

    if dataset_format in {"ifeval", "ifbench_test"}:
        prompt_text = str(row.get("prompt") or "").strip()
        if dataset_format == "ifbench_test":
            prompt = [{"role": "user", "content": prompt_text}] if prompt_text else []
        else:
            prompt = [{"role": "user", "content": prompt_text}] if prompt_text else []
        metadata["reward_type"] = "instruction_following_strict" if dataset_format == "ifeval" else "instruction_following_soft"
        metadata["instruction_id_list"] = row.get("instruction_id_list") or []
        metadata["kwargs"] = row.get("kwargs") or []
        metadata["prompt_text"] = prompt_text
    elif dataset_format == "jsonschemabench":
        schema = normalize_json_like_value(row.get("json_schema")) or {}
        prompt_text = str(row.get("prompt") or row.get("instruction") or "")
        prompt = [{"role": "user", "content": prompt_text}] if prompt_text else []
        metadata["reward_type"] = "structured_json_schema"
        metadata["schema"] = schema if isinstance(schema, dict) else {}
    elif dataset_format == "gpqa":
        prompt = _generic_question_prompt({"question": row.get("Question")})
        metadata["reward_type"] = "stem_mcqa"
        metadata["answer"] = ""
    else:
        converted = mv1.convert_row(row, dataset_format) if dataset_format not in {"apibench"} else []
        if converted:
            return [
                _normalize_pool_sample(sample, dataset_format=dataset_format, source_fields=source_fields)
                for sample in converted
            ]
        prompt = _generic_question_prompt(row)
        metadata["reward_type"] = ""

    return [
        _make_pool_sample(
            prompt=prompt,
            dataset_format=dataset_format,
            metadata=metadata,
            tools=tools,
            source_fields=source_fields,
        )
    ]


def convert_row_for_pool(row: dict[str, Any], dataset_format: str) -> list[dict[str, Any]]:
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
    if dataset_format == "ifbench_test":
        return _convert_ifbench_row_for_pool(row)
    if dataset_format == "gpqa":
        converted = mv1.convert_gpqa_row(row)
        if not converted:
            return []
        source_fields = _default_source_fields(row, dataset_format)
        return [
            _normalize_pool_sample(sample, dataset_format=dataset_format, source_fields=source_fields)
            for sample in converted
        ]

    try:
        converted = mv1.convert_row(row, dataset_format)
    except mv1.SkippableRowError:
        converted = []
    if converted:
        source_fields = _default_source_fields(row, dataset_format)
        return [
            _normalize_pool_sample(sample, dataset_format=dataset_format, source_fields=source_fields)
            for sample in converted
        ]
    return _generic_pool_fallback(row, dataset_format)


def iter_pool_samples(source: Path, dataset_format: str) -> Iterator[dict[str, Any]]:
    for row in mv1.iter_rows(source):
        if not isinstance(row, dict):
            continue
        for sample in convert_row_for_pool(row, dataset_format):
            yield sample


def process_dataset(path_str: str, format_name: str) -> None:
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"⚠️ Source missing: {path}")
        return

    domain = _dataset_domain_for_pool(format_name)
    print(f"Processing {format_name} ({domain})...")
    samples = list(mv0.iter_selected_samples(iter_pool_samples(path, format_name), 0, None))
    filename = f"{format_name}_{path.stem}.jsonl"
    for out_path in _pool_output_paths(domain, filename):
        dump_pool(samples, out_path)


def process_ifrl(path_str: str) -> None:
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"⚠️ IF-RL Source missing: {path}")
        return

    print("Processing IF-RL...")
    out_path = AVALANCHE_ROOT / "data" / "pool" / "ifrl" / f"ifrl_{path.stem}.jsonl"

    sys.path.append(str(SCRIPT_DIR / "if_rl"))
    import prepare_ifrl_data as ifrl

    samples = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_text = ifrl.normalize_prompt(row.get("prompt", ""))
            metadata = {
                "record_id": row.get("id", ""),
                "prompt_text": prompt_text,
                "dataset": row.get("dataset", ""),
                "agent_ref": row.get("agent_ref", ""),
            }
            if "kwargs" in row:
                metadata["kwargs"] = row["kwargs"]
            if "instruction_id_list" in row:
                metadata["instruction_id_list"] = row["instruction_id_list"]
            source_fields = _source_fields_for_pool(
                row,
                excluded_keys={"agent_ref", "dataset", "id", "instruction_id_list", "kwargs", "prompt"},
            )
            if source_fields:
                metadata["source_fields"] = source_fields

            samples.append(
                _make_pool_sample(
                    prompt=[{"role": "user", "content": prompt_text}] if prompt_text else [],
                    dataset_format="ifrl",
                    metadata={
                        "domain": "ifrl",
                        "dataset_name": row.get("dataset") or "ifrl",
                        **metadata,
                    },
                    tools=[],
                )
            )
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
