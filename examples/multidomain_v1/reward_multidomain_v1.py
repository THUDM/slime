from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V0_SCRIPT = ROOT / "multidomain_v0" / "reward_mixed_domain.py"


def _load_v0_module():
    spec = importlib.util.spec_from_file_location("reward_mixed_domain_v0", V0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_v0 = _load_v0_module()


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


def _exact_match_fraction(predicted: dict[str, Any], expected: list[str], expected_values: dict[str, Any]) -> float:
    if not expected:
        return 1.0
    matched = 0
    for key in expected:
        value = expected_values[key]
        if key in predicted and _v0._canonicalize(predicted[key]) == _v0._canonicalize(value):
            matched += 1
    return matched / len(expected)


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
        if isinstance(expected_type, str) and not _v0._check_python_type(_v0._canonicalize(value), expected_type):
            invalid_keys += 1
            continue
        enum_values = prop_schema.get("enum")
        if isinstance(enum_values, list) and _v0._canonicalize(value) not in [_v0._canonicalize(item) for item in enum_values]:
            invalid_keys += 1
    if invalid_keys:
        penalties += 0.1 * (invalid_keys / max(len(predicted), 1))

    return penalties


def _parse_predicted_calls(response: str, tools: list[dict[str, Any]], parser_type: str) -> list[dict[str, Any]]:
    parsed_calls = _v0._parse_tool_calls(response, tools=tools, parser_type=parser_type)
    return [
        {
            "name": call.get("name", ""),
            "arguments": call.get("parameters", call.get("arguments", {})),
        }
        for call in parsed_calls
    ]


def _tool_selection_strict_score(predicted_calls: list[dict[str, Any]], allowed_tool_names: list[str]) -> float:
    allowed = {str(name).strip() for name in allowed_tool_names if str(name).strip()}
    if not predicted_calls or not allowed:
        return 0.0
    predicted_name = str(predicted_calls[0].get("name", "")).strip()
    return 1.0 if predicted_name in allowed else 0.0


def _tool_call_soft_score(
    predicted_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> float:
    predicted = _v0.normalize_ground_truth_calls(predicted_calls)
    expected = _v0.normalize_ground_truth_calls(expected_calls)
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


def _instruction_rule_scores(metadata: dict[str, Any], sample) -> list[float]:
    instruction_ids = _v0._normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        return []
    prompt_text = metadata.get("prompt_text") or sample.prompt or ""
    prompt_text = prompt_text if isinstance(prompt_text, str) else str(prompt_text)
    kwargs_list = _v0._normalize_kwargs(metadata.get("kwargs"), len(instruction_ids))
    passed: list[float] = []
    for instruction_id, rule_kwargs in zip(instruction_ids, kwargs_list):
        try:
            passed.append(1.0 if _v0._check_instruction(instruction_id, rule_kwargs, prompt_text, sample.response or "") else 0.0)
        except Exception:
            passed.append(0.0)
    return passed


async def reward_func(args, sample, **kwargs):
    if sample.status == sample.Status.TRUNCATED:
        return 0.0

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    reward_type = str(metadata.get("reward_type", ""))
    response = _v0._clean_response(sample.response or "")
    if not response:
        return 0.0

    if reward_type == "stem_mcqa":
        return 1.0 if _v0._extract_mcqa_answer(response) == str(metadata.get("answer", "")).strip().upper() else 0.0

    if reward_type in {"tool_call_soft", "tool_call_strict", "tool_call", "tool_selection_strict"}:
        tools = metadata.get("tools") or []
        expected_calls = metadata.get("ground_truth") or []
        allowed_tool_names = metadata.get("allowed_tool_names") or []
        if not tools or (not expected_calls and reward_type != "tool_selection_strict"):
            return 0.0
        try:
            predicted_calls = _parse_predicted_calls(
                response,
                tools=tools,
                parser_type=str(metadata.get("parser_type", "qwen25")),
            )
        except Exception:
            return 0.0
        if reward_type == "tool_selection_strict":
            return _tool_selection_strict_score(predicted_calls, allowed_tool_names)
        if reward_type == "tool_call_soft":
            return _tool_call_soft_score(predicted_calls, expected_calls, tools)
        return 1.0 if _v0.calls_match_ground_truth(predicted_calls, expected_calls) else 0.0

    if reward_type == "structured_json_schema":
        try:
            payload = _v0.json.loads(_v0._extract_json_payload(response))
        except Exception:
            return 0.0
        schema = metadata.get("schema") if isinstance(metadata.get("schema"), dict) else {}
        return 1.0 if _v0._validate_json_schema(payload, schema) else 0.0

    if reward_type in {"instruction_following_soft", "instruction_following_strict", "ifeval"}:
        scores = _instruction_rule_scores(metadata, sample)
        if not scores:
            return 0.0
        if reward_type == "instruction_following_soft":
            return sum(scores) / len(scores)
        return 1.0 if all(score == 1.0 for score in scores) else 0.0

    return 0.0
