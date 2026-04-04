from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V0_SCRIPT = ROOT / "multidomain_v0" / "reward_mixed_domain.py"
IFBENCH_DIR = ROOT / "if_rl" / "offline_ifbench"


def _load_v0_module():
    spec = importlib.util.spec_from_file_location("reward_mixed_domain_v0", V0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_v0 = _load_v0_module()

_IFBENCH_REGISTRY: dict | None = None


def _load_ifbench_registry() -> dict:
    global _IFBENCH_REGISTRY
    if _IFBENCH_REGISTRY is not None:
        return _IFBENCH_REGISTRY
    if not IFBENCH_DIR.is_dir():
        _IFBENCH_REGISTRY = {}
        return _IFBENCH_REGISTRY
    try:
        for mod_name in ("instructions_util", "instructions", "instructions_registry", "evaluation_lib"):
            mod_path = IFBENCH_DIR / f"{mod_name}.py"
            spec = importlib.util.spec_from_file_location(mod_name, mod_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        _IFBENCH_REGISTRY = sys.modules["instructions_registry"].INSTRUCTION_DICT
    except Exception:
        _IFBENCH_REGISTRY = {}
    return _IFBENCH_REGISTRY


def _check_ifbench_instruction(instruction_id: str, rule_kwargs: dict, prompt: str, response: str) -> bool:
    registry = _load_ifbench_registry()
    if not registry or instruction_id not in registry:
        return False
    instruction_cls = registry[instruction_id]
    instruction = instruction_cls(instruction_id)
    filtered_kwargs = {k: v for k, v in rule_kwargs.items() if v is not None}
    instruction.build_description(**filtered_kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    return bool(response and response.strip() and instruction.check_following(response))


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
    """Check if predicted value matches a BFCL v3 ground truth value.

    BFCL v3 wraps each argument value in a list of acceptable alternatives,
    e.g. {"user_id": [7890]} means 7890 is the acceptable value.
    """
    canon_pred = _v0._canonicalize(predicted_value)
    canon_gt = _v0._canonicalize(gt_value)
    if canon_pred == canon_gt:
        return True
    # BFCL v3: gt_value is a list of acceptable alternatives
    if isinstance(canon_gt, list):
        return any(_v0._canonicalize(alt) == canon_pred for alt in canon_gt)
    return False


def _unwrap_bfcl_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Unwrap BFCL v3 list-wrapped argument values for normalization."""
    unwrapped = {}
    for key, value in arguments.items():
        if isinstance(value, list) and len(value) == 1:
            unwrapped[key] = value[0]
        else:
            unwrapped[key] = value
    return unwrapped


def _calls_match_ground_truth_bfcl(predicted_calls: list[dict[str, Any]], expected_calls: list[dict[str, Any]]) -> bool:
    """BFCL-aware version of calls_match_ground_truth."""
    pred = _v0.normalize_ground_truth_calls(predicted_calls)
    exp = _v0.normalize_ground_truth_calls(expected_calls)
    if len(pred) != len(exp):
        return False
    for p, e in zip(pred, exp):
        if p["name"] != e["name"]:
            return False
        p_args = p.get("arguments", {})
        e_args = e.get("arguments", {})
        if not isinstance(p_args, dict) or not isinstance(e_args, dict):
            if _v0._canonicalize(p_args) != _v0._canonicalize(e_args):
                return False
            continue
        e_unwrapped = _unwrap_bfcl_args(e_args)
        if _v0._canonicalize(p_args) != _v0._canonicalize(e_unwrapped):
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
    ifbench_registry = _load_ifbench_registry()
    passed: list[float] = []
    for instruction_id, rule_kwargs in zip(instruction_ids, kwargs_list):
        try:
            if ifbench_registry and instruction_id in ifbench_registry:
                result = _check_ifbench_instruction(instruction_id, rule_kwargs, prompt_text, sample.response or "")
            else:
                result = _v0._check_instruction(instruction_id, rule_kwargs, prompt_text, sample.response or "")
            passed.append(1.0 if result else 0.0)
        except Exception:
            passed.append(0.0)
    return passed


def _ifbench_rule_scores(metadata: dict[str, Any], sample, *, strict: bool) -> list[float] | None:
    instruction_ids = _v0._normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        return []

    _load_ifbench_registry()  # ensures evaluation_lib is in sys.modules
    try:
        evaluation_lib = sys.modules["evaluation_lib"]
    except KeyError:
        return None

    prompt_text = metadata.get("prompt_text") or sample.prompt or ""
    prompt_text = prompt_text if isinstance(prompt_text, str) else str(prompt_text)

    kwargs_list = _v0._normalize_kwargs(metadata.get("kwargs"), len(instruction_ids))
    sanitized_kwargs: list[dict[str, Any]] = []
    for entry in kwargs_list:
        if isinstance(entry, dict):
            sanitized_kwargs.append({k: v for k, v in entry.items() if v is not None})
        else:
            sanitized_kwargs.append({})

    raw_key = metadata.get("record_id", metadata.get("key", 0))
    try:
        key = int(raw_key)
    except (TypeError, ValueError):
        key = hash(str(raw_key)) % (10**9)

    try:
        input_example = evaluation_lib.InputExample(
            key=key,
            instruction_id_list=instruction_ids,
            prompt=prompt_text,
            kwargs=sanitized_kwargs,
        )
        prompt_to_response = {prompt_text: sample.response or ""}
        verifier = (
            evaluation_lib.test_instruction_following_strict
            if strict
            else evaluation_lib.test_instruction_following_loose
        )
        result = verifier(input_example, prompt_to_response)
    except Exception:
        return None

    follow_list = getattr(result, "follow_instruction_list", None)
    if not isinstance(follow_list, list):
        return None
    return [1.0 if bool(item) else 0.0 for item in follow_list]


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

    if reward_type == "bfcl_official":
        raise RuntimeError(
            "BFCL official evaluation is no longer supported through the generic multidomain reward router. "
            "Use the official BFCL evaluator over pool native rows instead."
        )

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
        return 1.0 if _calls_match_ground_truth_bfcl(predicted_calls, expected_calls) else 0.0

    if reward_type == "structured_json_schema":
        try:
            payload = _v0.json.loads(_v0._extract_json_payload(response))
        except Exception:
            return 0.0
        schema = metadata.get("schema") if isinstance(metadata.get("schema"), dict) else {}
        return 1.0 if _v0._validate_json_schema(payload, schema) else 0.0

    if reward_type in {"instruction_following_soft", "instruction_following_strict", "ifeval"}:
        dataset_name = str(metadata.get("dataset_name", "")).strip().lower()
        if dataset_name == "ifbench_test":
            scores = _ifbench_rule_scores(metadata, sample, strict=reward_type != "instruction_following_soft")
            if scores is None:
                raise RuntimeError(
                    "IFBench official evaluation_lib is required for ifbench_test. "
                    "Heuristic fallback has been removed."
                )
        else:
            scores = _instruction_rule_scores(metadata, sample)
        if not scores:
            return 0.0
        if reward_type == "instruction_following_soft":
            return sum(scores) / len(scores)
        return 1.0 if all(score == 1.0 for score in scores) else 0.0

    return 0.0
