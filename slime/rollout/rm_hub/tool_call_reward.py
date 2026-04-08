from __future__ import annotations

import json
import re
from typing import Any, Callable

from slime.rollout.rm_hub.reward_text import canonicalize, normalize_python_call_text

try:
    import jsonschema as _jsonschema
except ImportError:
    _jsonschema = None

TRAIN_TOOL_REWARD_TYPES = {
    "function_call_single",
    "api_call_text",
}

TOOL_CALL_REWARD_TYPES = {
    "tool_call_soft",
    "tool_call_strict",
    "tool_call",
    "tool_selection_strict",
    *TRAIN_TOOL_REWARD_TYPES,
}


def normalize_ground_truth_calls(raw_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        normalized.append(
            {
                "name": str(call.get("name", "")),
                "arguments": canonicalize(call.get("arguments", call.get("parameters", {}))),
            }
        )
    return normalized


def parse_tool_calls_from_xml(response: str) -> list[dict[str, Any]]:
    parsed_calls: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            parsed_calls.append(payload)
    return parsed_calls


def parse_tool_calls(response: str, tools: list[dict[str, Any]], parser_type: str) -> list[dict[str, Any]]:
    xml_calls = parse_tool_calls_from_xml(response)
    if xml_calls:
        return xml_calls
    try:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.managers.io_struct import Function, Tool
    except Exception:
        return []
    parser = FunctionCallParser(
        tools=[
            Tool(
                function=Function(
                    name=tool["function"]["name"],
                    description=tool["function"].get("description", ""),
                    parameters=tool["function"].get("parameters") or {"type": "object", "properties": {}},
                ),
                type=tool.get("type", "function"),
            )
            for tool in tools
        ],
        tool_call_parser=parser_type,
    )
    _, calls = parser.parse_non_stream(response)
    return [call.model_dump() for call in calls]


def parse_predicted_calls(response: str, tools: list[dict[str, Any]], parser_type: str) -> list[dict[str, Any]]:
    parsed_calls = parse_tool_calls(response, tools=tools, parser_type=parser_type)
    return [
        {
            "name": call.get("name", ""),
            "arguments": call.get("parameters", call.get("arguments", {})),
        }
        for call in parsed_calls
    ]


def _tool_schema_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    schema_map: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        parameters = function.get("parameters")
        schema_map[name] = parameters if isinstance(parameters, dict) else {}
    return schema_map


def _bfcl_value_matches(predicted_value: Any, gt_value: Any) -> bool:
    canon_pred = canonicalize(predicted_value)
    canon_gt = canonicalize(gt_value)
    if canon_pred == canon_gt:
        return True
    if isinstance(canon_gt, list):
        return any(canonicalize(alt) == canon_pred for alt in canon_gt)
    return False


def _unwrap_bfcl_args(arguments: dict[str, Any]) -> dict[str, Any]:
    unwrapped: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, list) and len(value) == 1:
            unwrapped[key] = value[0]
        else:
            unwrapped[key] = value
    return unwrapped


def _calls_match_ground_truth_bfcl(predicted_calls: list[dict[str, Any]], expected_calls: list[dict[str, Any]]) -> bool:
    pred = normalize_ground_truth_calls(predicted_calls)
    exp = normalize_ground_truth_calls(expected_calls)
    if len(pred) != len(exp):
        return False
    for pred_call, exp_call in zip(pred, exp):
        if pred_call["name"] != exp_call["name"]:
            return False
        pred_args = pred_call.get("arguments", {})
        exp_args = exp_call.get("arguments", {})
        if not isinstance(pred_args, dict) or not isinstance(exp_args, dict):
            if canonicalize(pred_args) != canonicalize(exp_args):
                return False
            continue
        if canonicalize(pred_args) != canonicalize(_unwrap_bfcl_args(exp_args)):
            return False
    return True


def _exact_match_fraction(predicted: dict[str, Any], expected: list[str], expected_values: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    matched = 0
    for key in expected:
        value = expected_values[key]
        if key in predicted and _bfcl_value_matches(predicted[key], value):
            matched += 1
    return matched / len(expected)


def _check_python_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _schema_violation_penalty(predicted: dict[str, Any], schema: dict[str, Any], expected_args: dict[str, Any]) -> float:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    if not predicted:
        return 0.0
    penalties = 0.0
    known_optional = set(properties) - set(expected_args)
    unknown_keys = [key for key in predicted if key not in properties]
    if unknown_keys:
        penalties += 0.1 * (len(unknown_keys) / max(len(predicted), 1))
    stray_optional = [key for key in predicted if key in known_optional]
    if stray_optional:
        penalties += 0.05 * (len(stray_optional) / max(len(predicted), 1))
    invalid_keys = 0
    for key, value in predicted.items():
        prop_schema = properties.get(key)
        if not isinstance(prop_schema, dict):
            continue
        expected_type = prop_schema.get("type")
        if isinstance(expected_type, str) and not _check_python_type(canonicalize(value), expected_type):
            invalid_keys += 1
            continue
        enum_values = prop_schema.get("enum")
        if isinstance(enum_values, list) and canonicalize(value) not in [canonicalize(item) for item in enum_values]:
            invalid_keys += 1
    if invalid_keys:
        penalties += 0.1 * (invalid_keys / max(len(predicted), 1))
    return penalties


def tool_selection_strict_score(predicted_calls: list[dict[str, Any]], allowed_tool_names: list[str]) -> float:
    allowed = {str(name).strip() for name in allowed_tool_names if str(name).strip()}
    if not predicted_calls or not allowed:
        return 0.0
    predicted_name = str(predicted_calls[0].get("name", "")).strip()
    return 1.0 if predicted_name in allowed else 0.0


def tool_call_soft_score(
    predicted_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> float:
    predicted = normalize_ground_truth_calls(predicted_calls)
    expected = normalize_ground_truth_calls(expected_calls)
    if not predicted or not expected:
        return 0.0
    schema_map = _tool_schema_map(tools)
    scores: list[float] = []
    pair_count = min(len(predicted), len(expected))
    for index in range(pair_count):
        score = 0.1
        predicted_name = predicted[index]["name"]
        expected_name = expected[index]["name"]
        predicted_args = predicted[index].get("arguments")
        expected_args = expected[index].get("arguments")
        if predicted_name == expected_name:
            score += 0.25
            schema = schema_map.get(expected_name) or {}
            required = schema.get("required")
            required_keys = [str(key) for key in required] if isinstance(required, list) else []
            if isinstance(predicted_args, dict) and isinstance(expected_args, dict):
                required_keys = [key for key in required_keys if key in expected_args]
                optional_keys = [key for key in expected_args if key not in required_keys]
                required_coverage = 1.0
                if required_keys:
                    required_present = sum(1 for key in required_keys if key in predicted_args)
                    required_coverage = required_present / len(required_keys)
                score += 0.25 * required_coverage
                score += 0.20 * _exact_match_fraction(predicted_args, required_keys, expected_args)
                score += 0.10 * _exact_match_fraction(predicted_args, optional_keys, expected_args)
                score -= _schema_violation_penalty(predicted_args, schema, expected_args)
        scores.append(score)
    if not scores:
        return 0.0
    base = sum(scores) / max(len(predicted), len(expected))
    if len(predicted) != len(expected):
        base -= 0.1 * (abs(len(predicted) - len(expected)) / max(len(predicted), len(expected)))
    return round(min(1.0, max(0.0, base)), 6)


def strict_tool_score(
    predicted_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> float:
    if not _calls_match_ground_truth_bfcl(predicted_calls, expected_calls):
        return 0.0
    if _jsonschema is None:
        return 1.0
    schema_map = _tool_schema_map(tools)
    for call in predicted_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        param_schema = schema_map.get(name)
        if param_schema and isinstance(args, dict):
            try:
                _jsonschema.validate(args, param_schema)
            except (_jsonschema.ValidationError, _jsonschema.SchemaError):
                return 0.0
    return 1.0


def compute_tool_call_reward(
    metadata: dict[str, Any],
    response: str,
    *,
    parse_predicted_calls_fn: Callable[[str, list[dict[str, Any]], str], list[dict[str, Any]]] = parse_predicted_calls,
) -> float:
    reward_type = str(metadata.get("reward_type", "")).strip()
    if reward_type not in TOOL_CALL_REWARD_TYPES:
        return 0.0
    if reward_type == "api_call_text":
        return 1.0 if normalize_python_call_text(response) == normalize_python_call_text(str(metadata.get("raw_api_call") or "")) else 0.0
    tools = metadata.get("tools") or []
    expected_calls = metadata.get("ground_truth") or []
    allowed_tool_names = metadata.get("allowed_tool_names") or []
    if not tools or (not expected_calls and reward_type != "tool_selection_strict"):
        return 0.0
    try:
        predicted_calls = parse_predicted_calls_fn(
            response,
            tools=tools,
            parser_type=str(metadata.get("parser_type", "qwen25")),
        )
    except Exception:
        return 0.0
    if reward_type == "tool_selection_strict":
        return tool_selection_strict_score(predicted_calls, allowed_tool_names)
    if reward_type in {"tool_call_soft", *TRAIN_TOOL_REWARD_TYPES}:
        return tool_call_soft_score(predicted_calls, expected_calls, tools)
    return strict_tool_score(predicted_calls, expected_calls, tools)
