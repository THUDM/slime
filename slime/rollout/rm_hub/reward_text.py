from __future__ import annotations

import ast
import json
import re
from typing import Any

try:
    import jsonschema as _jsonschema
except ImportError:
    _jsonschema = None


def clean_response(text: str) -> str:
    text = text or ""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx >= 0:
        text = text[idx + len(marker) :]
    return text.strip().replace("<|endoftext|>", "").replace("<pad>", "").replace("<|im_end|>", "").strip()


def extract_json_payload(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def maybe_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text
    return value


def canonicalize(value: Any) -> Any:
    value = maybe_parse_json(value)
    if isinstance(value, dict):
        return {str(key): canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def extract_mcqa_answer(response: str) -> str:
    match = re.search(r"(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]*([A-Z])\b", response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"Answer\s*:\s*([A-Z])\b", response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"<answer>\s*([A-Z])\s*</answer>", response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{([A-Z])\}", response)
    if match:
        return match.group(1).upper()
    lines = response.strip().splitlines()
    if not lines:
        return ""
    last = lines[-1].strip()
    if re.fullmatch(r"[A-Za-z]", last):
        return last.upper()
    match = re.fullmatch(r"\*{1,2}([A-Za-z])\*{1,2}", last)
    if match:
        return match.group(1).upper()
    match = re.fullmatch(r"[(\[]([A-Za-z])[)\]]", last)
    if match:
        return match.group(1).upper()
    match = re.fullmatch(r"([A-Za-z])[.)]\s*", last)
    if match:
        return match.group(1).upper()
    match = re.search(r"[:\s]([A-Z])\s*\.?\s*$", last)
    if match:
        return match.group(1).upper()
    return ""


def compute_stem_mcqa_reward(response: str, answer: Any) -> float:
    return 1.0 if extract_mcqa_answer(response) == str(answer or "").strip().upper() else 0.0


def normalize_freeform_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_python_call_text(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    for marker in ("<<<api_call>>>:", "api_call:", "API Call:"):
        if marker in value:
            value = value.split(marker, 1)[1].strip()
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    value = lines[0] if lines else value
    try:
        parsed = ast.parse(value, mode="eval")
        return ast.unparse(parsed.body)
    except Exception:
        return normalize_freeform_text(value)


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


def validate_json_schema_basic(data: Any, schema: Any) -> bool:
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        return True
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _check_python_type(data, expected_type):
        return False
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and data not in enum_values:
        return False
    if expected_type == "object":
        if not isinstance(data, dict):
            return False
        required_raw = schema.get("required")
        required = [str(key) for key in required_raw] if isinstance(required_raw, list) else []
        if any(key not in data for key in required):
            return False
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in properties:
                    return False
        for key, value in data.items():
            if key in properties and not validate_json_schema_basic(value, properties[key]):
                return False
        return True
    if expected_type == "array":
        if not isinstance(data, list):
            return False
        item_schema = schema.get("items")
        if isinstance(item_schema, (dict, bool)):
            return all(validate_json_schema_basic(item, item_schema) for item in data)
    return True


def validate_json_schema_full(payload: Any, schema: dict[str, Any]) -> bool:
    if _jsonschema is None:
        return validate_json_schema_basic(payload, schema)
    try:
        _jsonschema.validate(payload, schema)
        return True
    except (_jsonschema.ValidationError, _jsonschema.SchemaError):
        return False


def compute_structured_json_reward(response: str, schema: dict[str, Any]) -> float:
    try:
        payload = json.loads(extract_json_payload(response))
    except Exception:
        return 0.0
    return 1.0 if validate_json_schema_full(payload, schema) else 0.0
