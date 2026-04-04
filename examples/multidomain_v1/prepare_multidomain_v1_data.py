#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
V0_SCRIPT = ROOT / "multidomain_v0" / "prepare_mixed_domain_data.py"


def _load_v0_module():
    spec = importlib.util.spec_from_file_location("prepare_mixed_domain_data_v0", V0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_v0 = _load_v0_module()

SourceSpec = _v0.SourceSpec
make_sample = _v0.make_sample
normalize_message = _v0.normalize_message
normalize_json_like_value = _v0.normalize_json_like_value
normalize_tool_definition = _v0.normalize_tool_definition
normalize_tool_definitions = _v0.normalize_tool_definitions
normalize_ground_truth_calls = _v0.normalize_ground_truth_calls
iter_selected_samples = _v0.iter_selected_samples
build_choice_prompt = _v0.build_choice_prompt
prepare_single_source_samples = _v0.prepare_single_source_samples
allocate_sample_counts = _v0.allocate_sample_counts
next_domain_sample = _v0.next_domain_sample
build_weighted_schedule = _v0.build_weighted_schedule


class SkippableRowError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def dataset_domain(dataset_format: str) -> str:
    if dataset_format in {
        "nemotron_knowledge_mcqa",
        "ai2_arc",
        "scienceqa",
        "medmcqa",
        "openbookqa",
        "sciq",
        "mmlu_pro",
        "gpqa",
    }:
        return "stem"
    if dataset_format in {
        "apigen_mt_5k",
        "xlam_function_calling_60k",
        "toolbench_v1",
        "toolbench_v1_benchmark",
        "apibench",
        "agent_function_calling_open_dataset",
        "bfcl_v3",
        "bfcl_v3_multi_turn_base",
    }:
        return "tool"
    if dataset_format in {"nemotron_structured_outputs", "ifeval", "ifbench_test", "jsonschemabench"}:
        return "structured"
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def _rewrite_reward_type(samples: list[dict[str, Any]], reward_type: str) -> list[dict[str, Any]]:
    for sample in samples:
        metadata = sample.get("metadata") or {}
        metadata["reward_type"] = reward_type
        sample["metadata"] = metadata
    return samples


def _rewrite_parser_type(samples: list[dict[str, Any]], parser_type: str) -> list[dict[str, Any]]:
    for sample in samples:
        metadata = sample.get("metadata") or {}
        if metadata.get("domain") == "tool":
            metadata["parser_type"] = parser_type
            sample["metadata"] = metadata
    return samples


def _parse_tool_calls_from_xml(text: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text or "", flags=re.DOTALL):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            parsed.append(payload)
    return parsed


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                if isinstance(row, dict):
                    yield row
        return
    yield from _v0.iter_rows(path)


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    escape = False

    for char in text:
        if escape:
            current.append(char)
            escape = False
            continue
        if quote:
            current.append(char)
            if char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
            current.append(char)
            continue
        if char in ")]}":
            depth = max(depth - 1, 0)
            current.append(char)
            continue
        if char == delimiter and depth == 0:
            segment = "".join(current).strip()
            if segment:
                parts.append(segment)
            current = []
            continue
        current.append(char)

    segment = "".join(current).strip()
    if segment:
        parts.append(segment)
    return parts


def _find_top_level_equals(text: str) -> int:
    depth = 0
    quote: str | None = None
    escape = False

    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if quote:
            if char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth = max(depth - 1, 0)
            continue
        if char == "=" and depth == 0:
            return index
    return -1


def _extract_attr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _extract_attr_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_or_source(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def _schema_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"


def _extract_instruction_from_apibench(code_block: str) -> str:
    match = re.search(r"###Instruction:\s*(.*?)\s*###Output:", code_block or "", flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return (code_block or "").strip()


def _arguments_to_properties(arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: {"type": _schema_type(value)} for key, value in arguments.items()}


def _build_apibench_tools(ground_truth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in ground_truth:
        name = str(call.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "Call the benchmark API exactly as requested.",
                    "parameters": {
                        "type": "object",
                        "properties": _arguments_to_properties(arguments),
                        "required": list(arguments),
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def _normalize_toolbench_conversations(raw_conversations: Any) -> list[dict[str, Any]]:
    if isinstance(raw_conversations, list):
        return [turn for turn in raw_conversations if isinstance(turn, dict)]

    if isinstance(raw_conversations, dict):
        speakers = raw_conversations.get("from")
        values = raw_conversations.get("value")
        if isinstance(speakers, list) and isinstance(values, list):
            return [
                {"from": speaker, "value": value}
                for speaker, value in zip(speakers, values)
            ]

    return []


def _parse_toolbench_action_block(text: str) -> list[dict[str, Any]]:
    action_match = re.search(r"Action:\s*([^\n]+)", text or "")
    action_input_match = re.search(r"Action Input:\s*(.+)", text or "", flags=re.DOTALL)
    if not action_match:
        return []

    action_name = action_match.group(1).strip()
    if not action_name or action_name.lower() == "finish":
        return []

    raw_arguments = action_input_match.group(1).strip() if action_input_match else "{}"
    arguments = normalize_json_like_value(raw_arguments)
    if not isinstance(arguments, dict):
        arguments = {}
    return [{"name": action_name, "arguments": arguments}]


def _stable_shuffle_choices(record_id: str, choices: list[tuple[str, str]]) -> list[tuple[str, str]]:
    keyed: list[tuple[str, str, str]] = []
    for index, (_, text) in enumerate(choices):
        digest = hashlib.md5(f"{record_id}:{index}:{text}".encode("utf-8")).hexdigest()
        keyed.append((digest, str(index), text))
    keyed.sort()
    labels = list(_v0.CHOICE_LABELS)
    return [(labels[index], item[2]) for index, item in enumerate(keyed)]


def _api_name_from_entry(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (list, tuple)):
        for value in reversed(item):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if not isinstance(item, dict):
        return ""
    for key in ("name", "api_name", "tool_name", "api", "tool"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _toolbench_benchmark_tools(raw_api_list: Any) -> list[dict[str, Any]]:
    payload = normalize_json_like_value(raw_api_list)
    if not isinstance(payload, list):
        return []

    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload:
        name = _api_name_from_entry(item)
        if not name or name in seen:
            continue
        seen.add(name)
        description = ""
        parameters: dict[str, Any] = {"type": "object", "properties": {}}
        if isinstance(item, dict):
            description = str(item.get("description") or item.get("api_description") or "")
            raw_parameters = item.get("parameters") or item.get("args") or item.get("input_schema")
            if isinstance(raw_parameters, dict):
                parameters = raw_parameters
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return tools


def _toolbench_benchmark_allowed_names(raw_relevant_apis: Any) -> list[str]:
    payload = normalize_json_like_value(raw_relevant_apis)
    if not isinstance(payload, list):
        return []
    names: list[str] = []
    for item in payload:
        name = _api_name_from_entry(item)
        if name:
            names.append(name)
    return names


def _canonicalize_ast_call(node: ast.Call) -> dict[str, Any]:
    name = _extract_attr_name(node.func)
    if not name:
        raise SkippableRowError("unsupported_ast_call")

    arguments: dict[str, Any] = {}
    for index, arg in enumerate(node.args):
        key = f"arg{index}"
        value = _literal_or_source(arg)
        arguments[key] = value

    for keyword in node.keywords:
        key = keyword.arg or f"kw_{len(arguments)}"
        value = _literal_or_source(keyword.value)
        arguments[key] = value
    return {"name": name, "arguments": arguments}


def _parse_ast_api_calls(expression: str) -> tuple[list[dict[str, Any]], str]:
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError as exc:
        raise SkippableRowError("invalid_python_syntax") from exc

    if isinstance(node, ast.Call):
        return [_canonicalize_ast_call(node)], "ast"
    if isinstance(node, (ast.Tuple, ast.List)) and all(isinstance(item, ast.Call) for item in node.elts):
        return [_canonicalize_ast_call(item) for item in node.elts], "ast_multi_call"
    raise SkippableRowError("unsupported_ast_node")


def _parse_cli_style_api_call(expression: str) -> tuple[list[dict[str, Any]], str]:
    try:
        tokens = shlex.split(expression)
    except ValueError as exc:
        raise SkippableRowError("invalid_cli_syntax") from exc

    if not tokens or not any(token.startswith("--") for token in tokens[1:]):
        raise SkippableRowError("not_cli_style")

    name = tokens[0].strip()
    if not name:
        raise SkippableRowError("missing_cli_command")

    arguments: dict[str, Any] = {}
    positional_index = 0
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            raw_key = token[2:]
            if not raw_key:
                raise SkippableRowError("invalid_cli_flag")
            if "=" in raw_key:
                key, value_text = raw_key.split("=", 1)
                value: Any = normalize_json_like_value(value_text)
            else:
                key = raw_key
                next_token = tokens[index + 1] if index + 1 < len(tokens) else None
                if next_token is not None and not next_token.startswith("--"):
                    value = normalize_json_like_value(next_token)
                    index += 1
                else:
                    value = True
            arguments[key.replace("-", "_")] = value
        else:
            arguments[f"arg{positional_index}"] = normalize_json_like_value(token)
            positional_index += 1
        index += 1

    return [{"name": name, "arguments": arguments}], "cli_style"


def _parse_loose_value(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _parse_loose_single_call(expression: str) -> dict[str, Any]:
    text = expression.strip()
    open_index = text.find("(")
    close_index = text.rfind(")")
    if open_index <= 0 or close_index <= open_index:
        raise SkippableRowError("not_loose_call")

    name = text[:open_index].strip()
    if not name:
        raise SkippableRowError("missing_call_name")

    arguments: dict[str, Any] = {}
    positional_index = 0
    for item in _split_top_level(text[open_index + 1 : close_index]):
        equal_index = _find_top_level_equals(item)
        if equal_index > 0:
            key = item[:equal_index].strip().replace("-", "_")
            value = _parse_loose_value(item[equal_index + 1 :])
            arguments[key] = value
            continue
        arguments[f"arg{positional_index}"] = _parse_loose_value(item)
        positional_index += 1

    return {"name": name, "arguments": arguments}


def _parse_loose_api_calls(expression: str) -> tuple[list[dict[str, Any]], str]:
    text = expression.strip()
    if not text:
        raise SkippableRowError("empty_expression")

    segments = _split_top_level(text)
    if len(segments) > 1 and all("(" in segment and segment.endswith(")") for segment in segments):
        return [_parse_loose_single_call(segment) for segment in segments], "loose_multi_call"
    if "(" in text and text.endswith(")"):
        return [_parse_loose_single_call(text)], "loose_call"
    raise SkippableRowError("not_loose_call")


def _canonicalize_apibench_api_call(expression: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    cleaned = (expression or "").strip()
    if not cleaned:
        raise SkippableRowError("empty_expression")

    parsers = (_parse_ast_api_calls, _parse_cli_style_api_call, _parse_loose_api_calls)
    fallback_parsers = (_parse_cli_style_api_call, _parse_loose_api_calls, _parse_ast_api_calls)
    reasons: list[str] = []

    def _attempt(
        text: str,
        forced_mode: str | None = None,
        parser_sequence: Sequence[Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str] | None:
        local_reasons: list[str] = []
        for parser in parser_sequence or parsers:
            try:
                ground_truth, mode = parser(text)
                return _build_apibench_tools(ground_truth), ground_truth, forced_mode or mode
            except SkippableRowError as exc:
                local_reasons.append(exc.reason)
        reasons.extend(local_reasons)
        return None

    direct = _attempt(cleaned)
    if direct is not None:
        return direct

    try:
        literal = ast.literal_eval(cleaned)
    except Exception:
        literal = None
    if isinstance(literal, str):
        wrapped = _attempt(literal.strip(), forced_mode="wrapped_string", parser_sequence=fallback_parsers)
        if wrapped is not None:
            return wrapped

    if cleaned.endswith("."):
        trailing = _attempt(cleaned.rstrip(".").strip(), forced_mode="trailing_period", parser_sequence=fallback_parsers)
        if trailing is not None:
            return trailing

    raise SkippableRowError("unparseable_expression" if reasons else "empty_expression")


def _extract_agent_prompt(messages: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> list[dict[str, str]]:
    prompt: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if normalize_ground_truth_calls(message.get("tool_calls")) == ground_truth:
            break
        prompt.append(normalize_message(message))
    return prompt


def _load_bfcl_prompt(raw_turns: Any) -> list[dict[str, str]]:
    turns = normalize_json_like_value(raw_turns)
    if isinstance(turns, list) and turns:
        first = turns[0]
        if isinstance(first, list):
            return [normalize_message(message) for message in first if isinstance(message, dict)]
        if isinstance(first, dict):
            return [normalize_message(message) for message in turns if isinstance(message, dict)]
    return []


def _normalize_bfcl_call_item(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        try:
            return _parse_loose_single_call(item)
        except SkippableRowError:
            return None
    if isinstance(item, dict):
        if "function" in item or "name" in item:
            normalized = normalize_ground_truth_calls(item)
            return normalized[0] if normalized else None
        if len(item) == 1:
            name, arguments = next(iter(item.items()))
            arguments = normalize_json_like_value(arguments)
            if not isinstance(arguments, dict):
                arguments = {}
            return {"name": str(name), "arguments": arguments}
    return None


def _normalize_bfcl_ground_truth(raw_ground_truth: Any) -> list[dict[str, Any]]:
    payload = normalize_json_like_value(raw_ground_truth)
    if payload in ({}, [], None, ""):
        return []
    if isinstance(payload, list):
        calls: list[dict[str, Any]] = []
        for item in payload:
            normalized = _normalize_bfcl_call_item(item)
            if normalized:
                calls.append(normalized)
        return calls
    if isinstance(payload, dict):
        normalized = _normalize_bfcl_call_item(payload)
        return [normalized] if normalized else []
    return []


def convert_apigen_mt_5k_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    return _rewrite_reward_type(_v0.convert_apigen_mt_5k_row(row), "tool_call_soft")


def convert_xlam_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    return _rewrite_reward_type(_v0.convert_xlam_row(row), "tool_call_soft")


def convert_toolbench_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = normalize_tool_definitions(row.get("tools") or [])
    conversations = _normalize_toolbench_conversations(row.get("conversations") or [])
    history: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []

    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("from", "")).lower()
        value = turn.get("value", "")

        if speaker == "system":
            history.append({"role": "system", "content": "" if value is None else str(value)})
            continue
        if speaker in {"human", "user"}:
            history.append({"role": "user", "content": "" if value is None else str(value)})
            continue
        if speaker in {"function", "tool", "observation"}:
            history.append({"role": "tool", "content": "" if value is None else str(value)})
            continue
        if speaker not in {"assistant", "gpt"}:
            continue

        assistant_text = "" if value is None else str(value)
        ground_truth = normalize_ground_truth_calls(_parse_tool_calls_from_xml(assistant_text))
        if not ground_truth:
            ground_truth = normalize_ground_truth_calls(_parse_toolbench_action_block(assistant_text))
        if ground_truth:
            sample_tools = tools or _build_apibench_tools(ground_truth)
            samples.append(
                make_sample(
                    prompt=list(history),
                    metadata={
                        "domain": "tool",
                        "dataset_name": "toolbench_v1",
                        "reward_type": "tool_call_soft",
                        "ground_truth": ground_truth,
                        "tools": sample_tools,
                        "record_id": row.get("id"),
                    },
                    tools=sample_tools,
                )
            )
            continue

        history.append({"role": "assistant", "content": "" if value is None else str(value)})

    return samples


def convert_toolbench_benchmark_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = _toolbench_benchmark_tools(row.get("api_list"))
    allowed_tool_names = _toolbench_benchmark_allowed_names(row.get("relevant_apis"))
    query = str(row.get("query") or "").strip()
    if not tools or not allowed_tool_names or not query:
        return []
    return [
        make_sample(
            prompt=[{"role": "user", "content": query}],
            metadata={
                "domain": "tool",
                "dataset_name": "toolbench_v1_benchmark",
                "reward_type": "tool_selection_strict",
                "allowed_tool_names": allowed_tool_names,
                "tools": tools,
                "record_id": row.get("query_id"),
            },
            tools=tools,
        )
    ]


def convert_agent_function_calling_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    trace_id = row.get("trace_id") or row.get("session_id")
    for record in row.get("function_calls") or []:
        if not isinstance(record, dict):
            continue
        tools = normalize_tool_definitions(record.get("tools") or [])
        ground_truth = normalize_ground_truth_calls(record.get("tool_calls"))
        if not tools or not ground_truth:
            continue
        messages = record.get("messages") or []
        prompt = _extract_agent_prompt(messages, ground_truth)
        if not prompt:
            prompt = [normalize_message(message) for message in messages if isinstance(message, dict)]
        samples.append(
            make_sample(
                prompt=prompt,
                metadata={
                    "domain": "tool",
                    "dataset_name": "agent_function_calling_open_dataset",
                    "reward_type": "tool_call_soft",
                    "ground_truth": ground_truth,
                    "tools": tools,
                    "parser_type": "qwen25",
                    "record_id": trace_id,
                },
                tools=tools,
            )
        )
    return samples


def convert_bfcl_v3_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    ground_truth = _normalize_bfcl_ground_truth(row.get("ground_truth"))
    if not ground_truth:
        return []
    tools = normalize_tool_definitions(row.get("tools") or [])
    prompt = _load_bfcl_prompt(row.get("turns"))
    if not prompt:
        raise SkippableRowError("missing_bfcl_prompt")
    return [
        make_sample(
            prompt=prompt,
            metadata={
                "domain": "tool",
                "dataset_name": "bfcl_v3",
                "reward_type": "tool_call_soft",
                "ground_truth": ground_truth,
                "tools": tools,
                "parser_type": "qwen25",
                "record_id": row.get("id"),
                "subset": row.get("subset"),
                "test_category": row.get("test_category"),
            },
            tools=tools,
        )
    ]


def convert_bfcl_multi_turn_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    ground_truth = _normalize_bfcl_ground_truth(row.get("ground_truth"))
    if not ground_truth:
        return []
    tools = normalize_tool_definitions(row.get("tools") or [])
    prompt = [normalize_message(message) for message in row.get("messages") or [] if isinstance(message, dict)]
    if not prompt:
        raise SkippableRowError("missing_bfcl_prompt")
    return [
        make_sample(
            prompt=prompt,
            metadata={
                "domain": "tool",
                "dataset_name": "bfcl_v3_multi_turn_base",
                "reward_type": "tool_call_soft",
                "ground_truth": ground_truth,
                "tools": tools,
                "parser_type": "qwen25",
            },
            tools=tools,
        )
    ]


def convert_apibench_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools, ground_truth, parse_mode = _canonicalize_apibench_api_call(str(row.get("api_call", "")))
    prompt_text = _extract_instruction_from_apibench(str(row.get("code", "")))
    return [
        make_sample(
            prompt=[{"role": "user", "content": prompt_text}],
            metadata={
                "domain": "tool",
                "dataset_name": "apibench",
                "reward_type": "tool_call_soft",
                "ground_truth": ground_truth,
                "tools": tools,
                "provider": row.get("provider"),
                "parse_mode": parse_mode,
                "raw_api_call": row.get("api_call"),
                "accepted_for_eval": True,
            },
            tools=tools,
        )
    ]


def convert_ifbench_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_text = str(row.get("prompt", ""))
    return [
        make_sample(
            prompt=[{"role": "user", "content": prompt_text}],
            metadata={
                "domain": "structured",
                "dataset_name": "ifbench_test",
                "reward_type": "instruction_following_soft",
                "record_id": row.get("key"),
                "prompt_text": prompt_text,
                "instruction_id_list": row.get("instruction_id_list") or [],
                "kwargs": row.get("kwargs") or [],
            },
        )
    ]


def convert_ifeval_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    samples = _v0.convert_ifeval_row(row)
    return _rewrite_reward_type(samples, "instruction_following_strict")


def convert_jsonschemabench_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    schema = normalize_json_like_value(row.get("json_schema")) or {}
    prompt_text = str(row.get("prompt") or row.get("instruction") or "")
    if prompt_text:
        content = f"{prompt_text}\n\nSchema:\n{json.dumps(schema, ensure_ascii=False)}"
    else:
        content = json.dumps(schema, ensure_ascii=False)
    prompt = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return [
        make_sample(
            prompt=prompt,
            metadata={
                "domain": "structured",
                "dataset_name": "jsonschemabench",
                "reward_type": "structured_json_schema",
                "schema": schema if isinstance(schema, dict) else {},
                "record_id": row.get("unique_id"),
            },
        )
    ]


def convert_gpqa_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    question = str(row.get("Question") or "").strip()
    correct = str(row.get("Correct Answer") or "").strip()
    incorrect = [
        str(row.get("Incorrect Answer 1") or "").strip(),
        str(row.get("Incorrect Answer 2") or "").strip(),
        str(row.get("Incorrect Answer 3") or "").strip(),
    ]
    if not question or not correct or any(not choice for choice in incorrect):
        return []

    record_id = str(row.get("Record ID") or question[:32])
    ordered_choices = _stable_shuffle_choices(
        record_id,
        [("correct", correct), ("incorrect_1", incorrect[0]), ("incorrect_2", incorrect[1]), ("incorrect_3", incorrect[2])],
    )
    answer = ""
    for label, text in ordered_choices:
        if text == correct:
            answer = label
            break
    if not answer:
        return []

    return [
        make_sample(
            prompt=build_choice_prompt(question, ordered_choices),
            metadata={
                "domain": "stem",
                "dataset_name": "gpqa",
                "reward_type": "stem_mcqa",
                "answer": answer,
                "record_id": record_id,
                "subdomain": row.get("Subdomain"),
                "high_level_domain": row.get("High-level domain"),
            },
        )
    ]


def convert_row(row: dict[str, Any], dataset_format: str) -> list[dict[str, Any]]:
    if dataset_format == "apigen_mt_5k":
        return convert_apigen_mt_5k_row(row)
    if dataset_format == "xlam_function_calling_60k":
        return convert_xlam_row(row)
    if dataset_format == "toolbench_v1":
        return convert_toolbench_row(row)
    if dataset_format == "toolbench_v1_benchmark":
        return convert_toolbench_benchmark_row(row)
    if dataset_format == "agent_function_calling_open_dataset":
        return convert_agent_function_calling_row(row)
    if dataset_format == "bfcl_v3":
        return convert_bfcl_v3_row(row)
    if dataset_format == "bfcl_v3_multi_turn_base":
        return convert_bfcl_multi_turn_row(row)
    if dataset_format == "apibench":
        return convert_apibench_row(row)
    if dataset_format == "ifbench_test":
        return convert_ifbench_row(row)
    if dataset_format == "ifeval":
        return convert_ifeval_row(row)
    if dataset_format == "jsonschemabench":
        return convert_jsonschemabench_row(row)
    if dataset_format == "gpqa":
        return convert_gpqa_row(row)
    return _v0.convert_row(row, dataset_format)


def iter_conversion_outcomes(source: Path, dataset_format: str) -> Iterator[tuple[list[dict[str, Any]], str | None]]:
    for row in iter_rows(source):
        try:
            converted = convert_row(row, dataset_format)
        except SkippableRowError as exc:
            yield [], exc.reason
            continue
        if not converted:
            yield [], "no_samples"
            continue
        yield converted, None


def iter_converted_samples(source: Path, dataset_format: str) -> Iterator[dict[str, Any]]:
    for converted_rows, reason in iter_conversion_outcomes(source, dataset_format):
        if reason is not None:
            continue
        for converted in converted_rows:
            yield converted


def count_converted_samples(spec: SourceSpec) -> int:
    return sum(1 for _ in iter_converted_samples(spec.source, spec.dataset_format))


def build_source_report(source: Path, dataset_format: str) -> dict[str, Any]:
    total_rows = 0
    accepted_rows = 0
    skipped_rows = 0
    skip_reasons: Counter[str] = Counter()
    accepted_parse_modes: Counter[str] = Counter()

    for converted_rows, reason in iter_conversion_outcomes(source, dataset_format):
        total_rows += 1
        if reason is not None:
            skipped_rows += 1
            skip_reasons[reason] += 1
            continue
        accepted_rows += 1
        for sample in converted_rows:
            metadata = sample.get("metadata") or {}
            parse_mode = metadata.get("parse_mode")
            if parse_mode:
                accepted_parse_modes[str(parse_mode)] += 1

    coverage = 0.0 if total_rows == 0 else accepted_rows / total_rows
    return {
        "dataset_format": dataset_format,
        "total_rows": total_rows,
        "accepted_rows": accepted_rows,
        "skipped_rows": skipped_rows,
        "coverage": coverage,
        "skip_reasons": dict(skip_reasons),
        "accepted_parse_modes": dict(accepted_parse_modes),
    }


def prepare_mixed_samples(
    specs: Sequence[SourceSpec],
    skip_samples: int,
    max_samples: int | None,
) -> Iterator[dict[str, Any]]:
    capacities = [count_converted_samples(spec) for spec in specs]
    ratios = [spec.ratio for spec in specs]
    skip_counts = allocate_sample_counts(capacities, ratios, skip_samples)
    remaining_capacities = [max(0, capacity - skip) for capacity, skip in zip(capacities, skip_counts)]
    target_counts = allocate_sample_counts(remaining_capacities, ratios, max_samples)

    domains: list[str] = []
    domain_to_indices: dict[str, list[int]] = {}
    domain_weights: dict[str, float] = {}
    for index, spec in enumerate(specs):
        if spec.domain not in domain_to_indices:
            domains.append(spec.domain)
            domain_to_indices[spec.domain] = []
            domain_weights[spec.domain] = 0.0
        domain_to_indices[spec.domain].append(index)
        domain_weights[spec.domain] += spec.ratio

    total_target = sum(target_counts)
    remaining_counts = target_counts[:]
    iterators = [
        iter(iter_selected_samples(iter_converted_samples(spec.source, spec.dataset_format), skip_counts[index], None))
        for index, spec in enumerate(specs)
    ]
    active_sources = [True for _ in specs]
    source_schedules = {
        domain: build_weighted_schedule([specs[index].ratio for index in domain_to_indices[domain]])
        for domain in domains
    }
    domain_remaining = {
        domain: sum(remaining_counts[index] for index in domain_to_indices[domain])
        for domain in domains
    }
    yielded = 0

    for domain in sorted(domains, key=lambda name: domain_weights[name], reverse=True):
        if domain_remaining[domain] <= 0:
            continue
        sample = next_domain_sample(domain, domain_to_indices[domain], source_schedules, iterators, active_sources, remaining_counts)
        if sample is None:
            continue
        yielded += 1
        domain_remaining[domain] -= 1
        yield sample
        if yielded >= total_target:
            return

    domain_schedule = build_weighted_schedule([domain_weights[domain] for domain in domains])
    while True:
        emitted = False
        for domain_index in domain_schedule:
            domain = domains[domain_index]
            if domain_remaining[domain] <= 0:
                continue
            sample = next_domain_sample(domain, domain_to_indices[domain], source_schedules, iterators, active_sources, remaining_counts)
            if sample is None:
                continue
            emitted = True
            yielded += 1
            domain_remaining[domain] -= 1
            yield sample
            if yielded >= total_target:
                return
        if not emitted:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert multidomain v1 datasets into slime jsonl format.")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--dataset-format", action="append", required=True)
    parser.add_argument("--source-ratio", action="append", type=float, required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--reward-type-override", default=None)
    parser.add_argument("--parser-type", default="qwen25")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (len(args.source) == len(args.dataset_format) == len(args.source_ratio)):
        raise ValueError("source, dataset-format, and source-ratio counts must match")
    if args.report_path and len(args.source) != 1:
        raise ValueError("report-path currently requires a single source")

    specs = [
        SourceSpec(Path(source), dataset_format, float(ratio), dataset_domain(dataset_format))
        for source, dataset_format, ratio in zip(args.source, args.dataset_format, args.source_ratio)
    ]

    capacities = [count_converted_samples(spec) for spec in specs]
    empty_enabled_specs = [
        f"{spec.dataset_format}:{spec.source}"
        for spec, capacity in zip(specs, capacities)
        if spec.ratio > 0 and capacity == 0
    ]
    if empty_enabled_specs:
        raise SystemExit(
            "Enabled sources produced zero converted samples: " + ", ".join(empty_enabled_specs)
        )

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(specs) == 1:
        samples = iter_selected_samples(iter_converted_samples(specs[0].source, specs[0].dataset_format), args.skip_samples, args.max_samples)
    else:
        samples = prepare_mixed_samples(specs, args.skip_samples, args.max_samples)

    with dest.open("w", encoding="utf-8") as fout:
        for sample in samples:
            sample = _rewrite_parser_type([sample], str(args.parser_type))[0]
            if args.reward_type_override:
                sample = _rewrite_reward_type([sample], str(args.reward_type_override))[0]
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if args.report_path:
        report = build_source_report(Path(args.source[0]), args.dataset_format[0])
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
