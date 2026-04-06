from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from dataset_registry import (
    DEFAULT_TRAIN_DATASETS_BY_DOMAIN,
    DEFAULT_TRAIN_DATASETS_BY_GROUP,
    EVAL_DATASET_SPECS,
    TRAIN_DATASET_DOMAIN_MAP,
    TRAIN_DATASET_GROUP_MAP,
    TRAIN_DATASET_SOURCE_MAP,
)
from dataset_selection import (
    discover_canonical_train_sources as discover_canonical_train_sources_shared,
    resolve_named_datasets as resolve_named_datasets_shared,
    train_domains_for_datasets as train_domains_for_datasets_shared,
    train_groups_for_datasets as train_groups_for_datasets_shared,
)


logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
IFBENCH_DIR = SCRIPT_DIR / "if_rl" / "offline_ifbench"

GENERIC_EVAL_DATASETS: list[tuple[str, str, int]] = [
    (spec.name, spec.relpath, spec.n_samples_per_eval_prompt)
    for spec in EVAL_DATASET_SPECS.values()
    if not spec.official
]

OFFICIAL_EVAL_DATASETS: list[tuple[str, str]] = [
    (spec.name, spec.relpath)
    for spec in EVAL_DATASET_SPECS.values()
    if spec.official
]

TRAIN_TOOL_REWARD_TYPES = {
    "function_call_single",
    "api_call_text",
}


try:
    import jsonschema as _jsonschema
except ImportError:
    _jsonschema = None


def default_train_datasets_for_domain(domain: str) -> tuple[str, ...]:
    try:
        return DEFAULT_TRAIN_DATASETS_BY_DOMAIN[domain]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_TRAIN_DATASETS_BY_DOMAIN))
        raise ValueError(f"Unsupported train domain '{domain}'. Supported domains: {supported}") from exc


def default_train_datasets_for_group(group: str) -> tuple[str, ...]:
    try:
        return DEFAULT_TRAIN_DATASETS_BY_GROUP[group]
    except KeyError as exc:
        supported = ", ".join(sorted(DEFAULT_TRAIN_DATASETS_BY_GROUP))
        raise ValueError(f"Unsupported train group '{group}'. Supported groups: {supported}") from exc


def train_domains_for_datasets(dataset_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(train_domains_for_datasets_shared(list(dataset_names)))


def train_groups_for_datasets(dataset_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(train_groups_for_datasets_shared(list(dataset_names)))


def domain_signature(domains: list[str] | tuple[str, ...]) -> str:
    ordered = tuple(dict.fromkeys(domains))
    if not ordered:
        return "unknown"
    if len(ordered) == 1:
        return ordered[0]
    return "mixed-" + "+".join(ordered)


def group_signature(groups: list[str] | tuple[str, ...]) -> str:
    ordered = tuple(dict.fromkeys(groups))
    if not ordered:
        return "unknown"
    formatted = tuple(group.replace("_", "-") for group in ordered)
    if len(formatted) == 1:
        return formatted[0]
    return "mixed-" + "+".join(formatted)


def domain_signature_for_train_datasets(dataset_names: list[str] | tuple[str, ...]) -> str:
    return domain_signature(train_domains_for_datasets(dataset_names))


def group_signature_for_train_datasets(dataset_names: list[str] | tuple[str, ...]) -> str:
    return group_signature(train_groups_for_datasets(dataset_names))


def resolve_named_datasets(pool_root: Path, dataset_names: list[str] | tuple[str, ...]) -> list[Path]:
    return resolve_named_datasets_shared(pool_root, list(dataset_names))


def discover_canonical_train_sources(pool_root: Path) -> list[Path]:
    return discover_canonical_train_sources_shared(pool_root)


def _clean_response(text: str) -> str:
    text = text or ""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx >= 0:
        text = text[idx + len(marker) :]
    return text.strip().replace("<|endoftext|>", "").replace("<pad>", "").replace("<|im_end|>", "").strip()


def _extract_json_payload(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _maybe_parse_json(value: Any) -> Any:
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


def _canonicalize(value: Any) -> Any:
    value = _maybe_parse_json(value)
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


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
                "arguments": _canonicalize(call.get("arguments", call.get("parameters", {}))),
            }
        )
    return normalized


def _parse_tool_calls_from_xml(response: str) -> list[dict[str, Any]]:
    parsed_calls: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            parsed_calls.append(payload)
    return parsed_calls


def _parse_tool_calls(response: str, tools: list[dict[str, Any]], parser_type: str) -> list[dict[str, Any]]:
    xml_calls = _parse_tool_calls_from_xml(response)
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


def _parse_predicted_calls(response: str, tools: list[dict[str, Any]], parser_type: str) -> list[dict[str, Any]]:
    parsed_calls = _parse_tool_calls(response, tools=tools, parser_type=parser_type)
    return [
        {
            "name": call.get("name", ""),
            "arguments": call.get("parameters", call.get("arguments", {})),
        }
        for call in parsed_calls
    ]


def _extract_mcqa_answer(response: str) -> str:
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


def _normalize_freeform_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_python_call_text(text: str) -> str:
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
        return _normalize_freeform_text(value)


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


def _validate_json_schema_basic(data: Any, schema: Any) -> bool:
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
            if key in properties and not _validate_json_schema_basic(value, properties[key]):
                return False
        return True
    if expected_type == "array":
        if not isinstance(data, list):
            return False
        item_schema = schema.get("items")
        if isinstance(item_schema, (dict, bool)):
            return all(_validate_json_schema_basic(item, item_schema) for item in data)
    return True


def _validate_json_schema_full(payload: Any, schema: dict[str, Any]) -> bool:
    if _jsonschema is None:
        return _validate_json_schema_basic(payload, schema)
    try:
        _jsonschema.validate(payload, schema)
        return True
    except (_jsonschema.ValidationError, _jsonschema.SchemaError):
        return False


def _normalize_instruction_ids(raw_ids: Any) -> list[str]:
    output: list[str] = []
    for value in raw_ids or []:
        text = str(value).strip() if value is not None else ""
        if text:
            output.append(text)
    return output


def _normalize_kwargs(raw_kwargs: Any, n: int) -> list[dict[str, Any]]:
    if isinstance(raw_kwargs, list):
        items = [dict(x) if isinstance(x, dict) else {} for x in raw_kwargs]
    elif isinstance(raw_kwargs, dict):
        items = [dict(raw_kwargs) for _ in range(n)]
    else:
        items = [{} for _ in range(n)]
    if len(items) < n:
        items.extend({} for _ in range(n - len(items)))
    elif len(items) > n:
        items = items[:n]
    return [{k: v for k, v in item.items() if v is not None} for item in items]


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text, flags=re.UNICODE)


def _normalized_words(text: str) -> list[str]:
    return [w.lower() for w in _word_tokens(text)]


def _count_keyword(text: str, keyword: str) -> int:
    return sum(1 for w in _normalized_words(text) if w == str(keyword).lower())


def _relation_ok(actual: int, target: int, relation: str | None) -> bool:
    relation = (relation or "exactly").strip().lower()
    if relation in {"exactly", "equal", "equal to"}:
        return actual == target
    if relation in {"at least", "greater than or equal", "no less than"}:
        return actual >= target
    if relation in {"at most", "less than or equal", "no more than"}:
        return actual <= target
    if relation in {"less than", "fewer than"}:
        return actual < target
    if relation in {"more than", "greater than"}:
        return actual > target
    if relation == "around":
        tolerance = max(round(target * 0.1), 1)
        return abs(actual - target) <= tolerance
    return actual == target


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_paragraphs(text: str) -> list[str]:
    for divider in ("* * *", "***"):
        if divider in text:
            return [part.strip() for part in text.split(divider) if part.strip()]
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def _first_word(text: str) -> str:
    words = _word_tokens(text)
    return words[0].lower() if words else ""


def _last_word(text: str) -> str:
    words = _word_tokens(text)
    return words[-1].lower() if words else ""


def _expected_paragraphs_from_prompt(prompt: str) -> int | None:
    match = re.search(r"There should be (\d+) paragraphs", prompt, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _verify_keywords(text: str, keyword_list: list[str]) -> bool:
    lower = text.lower()
    return all(str(keyword).lower() in lower for keyword in keyword_list)


def _validate_forbidden_words(text: str, forbidden_words: list[str]) -> bool:
    lower = text.lower()
    return all(str(word).lower() not in lower for word in forbidden_words)


def _verify_letter_frequency(text: str, letter: str, frequency: int, relation: str) -> bool:
    actual = text.lower().count(str(letter).lower())
    return _relation_ok(actual, int(frequency), relation)


def _verify_total_letter_count(text: str, n: int, relation: str) -> bool:
    actual = sum(1 for ch in text if ch.isalpha())
    return _relation_ok(actual, int(n), relation)


def _verify_lowercase_word_count(text: str, n: int) -> bool:
    return len(re.findall(r"\b[a-z]+\b", text)) <= int(n)


def _verify_paragraph_count(text: str, n: int) -> bool:
    return len(_split_paragraphs(text)) == int(n)


def _validate_word_constraint(text: str, n: int, relation: str) -> bool:
    return _relation_ok(len(_word_tokens(text)), int(n), relation)


def _verify_sentence_constraint(text: str, n: int, relation: str) -> bool:
    return _relation_ok(len(_split_sentences(text)), int(n), relation)


def _validate_paragraphs(text: str, num_paragraphs: int, first_word: str, nth_paragraph: int) -> bool:
    paragraphs = _split_paragraphs(text)
    if len(paragraphs) != int(num_paragraphs):
        return False
    idx = int(nth_paragraph) - 1
    if idx < 0 or idx >= len(paragraphs):
        return False
    return _first_word(paragraphs[idx]) == str(first_word).lower()


def _verify_postscript(text: str, postscript_marker: str) -> bool:
    marker = str(postscript_marker)
    idx = text.find(marker)
    if idx < 0:
        return False
    return len(text[idx:].strip()) > len(marker)


def _validate_placeholders(text: str, n: int) -> bool:
    return len(re.findall(r"\[[^\[\]]+\]", text)) >= int(n)


def _verify_bullet_points(text: str, n: int) -> bool:
    bullets = [line for line in text.splitlines() if line.strip().startswith(("*", "-"))]
    return len(bullets) == int(n)


def _validate_title(text: str) -> bool:
    return re.search(r"<<[^<>]+>>", text) is not None


def _validate_choice(text: str, prompt: str) -> bool:
    match = re.search(r"one of the following options:\s*(\(.+?\))", prompt, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return False
    try:
        options = ast.literal_eval(match.group(1))
    except Exception:
        return False
    stripped = text.strip()
    return stripped in {str(option).strip() for option in options}


def _validate_highlighted_sections(text: str, n: int) -> bool:
    return len(re.findall(r"\*[^*\n]+\*", text)) >= int(n)


def _validate_sections(text: str, n: int, section_splitter: str) -> bool:
    splitter = str(section_splitter)
    matches = re.findall(rf"(?m)^\s*{re.escape(splitter)}\s+\d+\b", text)
    return len(matches) == int(n)


def _validate_json_format(text: str) -> bool:
    try:
        json.loads(_extract_json_payload(text))
        return True
    except Exception:
        return False


def _validate_two_responses(text: str) -> bool:
    if text.count("******") != 1:
        return False
    first, second = [part.strip() for part in text.split("******", 1)]
    return bool(first) and bool(second) and first != second


def _validate_uppercase(text: str) -> bool:
    return text == text.upper()


def _validate_lowercase(text: str) -> bool:
    return text == text.lower()


def _validate_frequency_capital_words(text: str, n: int, relation: str) -> bool:
    actual = len(re.findall(r"\b[A-Z]+\b", text))
    return _relation_ok(actual, int(n), relation)


def _validate_end(text: str, end_phrase: str) -> bool:
    return text.rstrip().endswith(str(end_phrase))


def _validate_quotation(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith('"') and stripped.endswith('"')


def _validate_no_commas(text: str) -> bool:
    return "," not in text


def _validate_no_char(text: str, forbidden: str) -> bool:
    return forbidden not in text


def _validate_same_start_end_word(text: str) -> bool:
    words = _normalized_words(text)
    return len(words) >= 2 and words[0] == words[-1]


def _validate_first_word_answer(text: str, first_word: str) -> bool:
    return _first_word(text) == str(first_word).lower()


def _validate_last_word_answer(text: str, last_word: str) -> bool:
    return _last_word(text) == str(last_word).lower()


def _validate_first_word_each_sentence(text: str, first_word: str) -> bool:
    expected = str(first_word).lower()
    sentences = _split_sentences(text)
    return bool(sentences) and all(_first_word(sentence) == expected for sentence in sentences)


def _validate_last_word_each_sentence(text: str, last_word: str) -> bool:
    expected = str(last_word).lower()
    sentences = _split_sentences(text)
    return bool(sentences) and all(_last_word(sentence) == expected for sentence in sentences)


def _validate_unique_words(text: str) -> bool:
    words = _normalized_words(text)
    return len(words) == len(set(words))


def _validate_square_brackets(text: str) -> bool:
    chunks = re.findall(r"\[[^\[\]\s]+\]", text)
    compact = re.sub(r"\s+", "", text)
    return bool(chunks) and "".join(chunks) == compact


def _validate_bigram_wrapping(text: str) -> bool:
    matches = re.findall(r"<<([^<>]+)>>", text)
    if not matches:
        return False
    compact = re.sub(r"\s+", "", text)
    rebuilt = "".join(f"<<{match}>>" for match in matches)
    if rebuilt != compact:
        return False
    return all(len(_word_tokens(match)) == 2 for match in matches)


def _validate_sentence_hyphens(text: str) -> bool:
    return "-" in text and " - " not in text and "- " not in text and " -" not in text


def _validate_palindrome(text: str) -> bool:
    for word in _normalized_words(text):
        if len(word) >= 3 and word == word[::-1]:
            return True
    return False


def _validate_no_adjacent_consecutive(text: str) -> bool:
    words = _normalized_words(text)
    initials = [word[0] for word in words if word and word[0].isalpha()]
    for left, right in zip(initials, initials[1:]):
        if abs(ord(left) - ord(right)) == 1:
            return False
    return True


def _validate_keyword_specific_position(text: str, keyword: str, n: int, m: int) -> bool:
    sentences = _split_sentences(text)
    idx = int(n) - 1
    if idx < 0 or idx >= len(sentences):
        return False
    words = _normalized_words(sentences[idx])
    pos = int(m) - 1
    if pos < 0 or pos >= len(words):
        return False
    return words[pos] == str(keyword).lower()


def _validate_repeat_phrase(text: str, phrase: str, small_n: int) -> bool:
    base = _normalized_words(phrase)
    if not base:
        return False
    center = len(base) // 2
    words = _normalized_words(text)
    matches = 0
    for i in range(0, len(words) - len(base) + 1):
        window = words[i : i + len(base)]
        if all(window[j] == base[j] for j in range(len(base)) if j != center) and window[center] != base[center]:
            matches += 1
    return matches == int(small_n)


def _validate_counting_composition(text: str, prompt: str, n_sent: int, n_words: int) -> bool:
    expected_paragraphs = _expected_paragraphs_from_prompt(prompt)
    paragraphs = _split_paragraphs(text)
    if expected_paragraphs is not None and len(paragraphs) != expected_paragraphs:
        return False
    for paragraph in paragraphs:
        sentences = _split_sentences(paragraph)
        if len(sentences) != int(n_sent):
            return False
        for sentence in sentences:
            if len(_word_tokens(sentence)) != int(n_words):
                return False
    return bool(paragraphs)


def _validate_response_language(text: str, language: str) -> bool:
    try:
        from langdetect import detect
    except Exception:
        return False
    try:
        return detect(text) == language
    except Exception:
        return False


def _check_instruction(instruction_id: str, rule_kwargs: dict[str, Any], prompt: str, response: str) -> bool:
    if instruction_id == "keywords:existence":
        return _verify_keywords(response, rule_kwargs.get("keywords") or [])
    if instruction_id == "keywords:forbidden_words":
        return _validate_forbidden_words(response, rule_kwargs.get("forbidden_words") or [])
    if instruction_id == "punctuation:no_comma":
        return _validate_no_commas(response)
    if instruction_id == "detectable_format:title":
        return _validate_title(response)
    if instruction_id == "detectable_format:bigram_wrapping":
        return _validate_bigram_wrapping(response)
    if instruction_id == "keywords:letter_frequency":
        return _verify_letter_frequency(response, rule_kwargs.get("letter", ""), int(rule_kwargs.get("let_frequency", 0)), rule_kwargs.get("let_relation", "exactly"))
    if instruction_id == "punctuation:punctuation_exclamation":
        return _validate_no_char(response, "!")
    if instruction_id == "last_word:last_word_answer":
        return _validate_last_word_answer(response, rule_kwargs.get("last_word", ""))
    if instruction_id == "count:lowercase_counting":
        return _verify_lowercase_word_count(response, int(rule_kwargs.get("N", 0)))
    if instruction_id in {"keywords:word_count_different_numbers", "keywords:frequency"}:
        return _relation_ok(_count_keyword(response, rule_kwargs.get("keyword", "")), int(rule_kwargs.get("frequency", 0)), rule_kwargs.get("relation", "exactly"))
    if instruction_id == "startend:end_checker":
        return _validate_end(response, rule_kwargs.get("end_phrase", ""))
    if instruction_id == "punctuation:punctuation_dot":
        return _validate_no_char(response, ".")
    if instruction_id == "keywords:word_once":
        return _count_keyword(response, rule_kwargs.get("keyword", "")) == 1
    if instruction_id == "detectable_format:sentence_hyphens":
        return _validate_sentence_hyphens(response)
    if instruction_id == "keywords:palindrome":
        return _validate_palindrome(response)
    if instruction_id == "detectable_format:number_bullet_lists":
        return _verify_bullet_points(response, int(rule_kwargs.get("num_bullets", 0)))
    if instruction_id == "letters:letter_counting2":
        return _verify_letter_frequency(response, rule_kwargs.get("letter", ""), int(rule_kwargs.get("let_frequency", 0)), rule_kwargs.get("let_relation", "exactly"))
    if instruction_id == "keywords:keyword_specific_position":
        return _validate_keyword_specific_position(response, rule_kwargs.get("keyword", ""), int(rule_kwargs.get("n", 0)), int(rule_kwargs.get("m", 0)))
    if instruction_id == "length_constraints:number_words":
        return _validate_word_constraint(response, int(rule_kwargs.get("num_words", 0)), rule_kwargs.get("relation", "exactly"))
    if instruction_id == "keywords:start_end":
        return _validate_same_start_end_word(response)
    if instruction_id == "detectable_format:square_brackets":
        return _validate_square_brackets(response)
    if instruction_id == "detectable_format:number_highlighted_sections":
        return _validate_highlighted_sections(response, int(rule_kwargs.get("num_highlights", 0)))
    if instruction_id == "detectable_content:number_placeholders":
        return _validate_placeholders(response, int(rule_kwargs.get("num_placeholders", 0)))
    if instruction_id == "copy:repeat_phrase":
        return _validate_repeat_phrase(response, rule_kwargs.get("phrase", ""), int(rule_kwargs.get("small_n", 0)))
    if instruction_id == "startend:quotation":
        return _validate_quotation(response)
    if instruction_id == "first_word:first_word_answer":
        return _validate_first_word_answer(response, rule_kwargs.get("first_word", ""))
    if instruction_id == "detectable_content:postscript":
        return _verify_postscript(response, rule_kwargs.get("postscript_marker", ""))
    if instruction_id == "language:response_language":
        return _validate_response_language(response, rule_kwargs.get("language", ""))
    if instruction_id == "keywords:no_adjacent_consecutive":
        return _validate_no_adjacent_consecutive(response)
    if instruction_id == "count:count_increment_word":
        return _count_keyword(response, rule_kwargs.get("keyword1", "")) == 1 and _count_keyword(response, rule_kwargs.get("keyword2", "")) == 2
    if instruction_id == "letters:letter_counting":
        return _verify_total_letter_count(response, int(rule_kwargs.get("N", 0)), rule_kwargs.get("relation", "exactly"))
    if instruction_id == "length_constraints:number_paragraphs":
        return _verify_paragraph_count(response, int(rule_kwargs.get("num_paragraphs", 0)))
    if instruction_id == "length_constraints:number_sentences":
        return _verify_sentence_constraint(response, int(rule_kwargs.get("num_sentences", 0)), rule_kwargs.get("relation", "exactly"))
    if instruction_id == "change_case:english_capital":
        return _validate_uppercase(response)
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        return _validate_paragraphs(response, int(rule_kwargs.get("num_paragraphs", 0)), rule_kwargs.get("first_word", ""), int(rule_kwargs.get("nth_paragraph", 0)))
    if instruction_id == "change_case:english_lowercase":
        return _validate_lowercase(response)
    if instruction_id == "first_word:first_word_sent":
        return _validate_first_word_each_sentence(response, rule_kwargs.get("first_word", ""))
    if instruction_id == "last_word:last_word_sent":
        return _validate_last_word_each_sentence(response, rule_kwargs.get("last_word", ""))
    if instruction_id == "detectable_format:multiple_sections":
        return _validate_sections(response, int(rule_kwargs.get("num_sections", 0)), rule_kwargs.get("section_spliter", rule_kwargs.get("section_splitter", "")))
    if instruction_id == "count:count_unique":
        return _validate_unique_words(response)
    if instruction_id in {"paragraphs:paragraphs", "paragraphs:paragraphs2"}:
        expected = _expected_paragraphs_from_prompt(prompt)
        return _verify_paragraph_count(response, expected) if expected is not None else False
    if instruction_id == "change_case:capital_word_frequency":
        return _validate_frequency_capital_words(response, int(rule_kwargs.get("capital_frequency", 0)), rule_kwargs.get("capital_relation", "exactly"))
    if instruction_id == "count:counting_composition":
        return _validate_counting_composition(response, prompt, int(rule_kwargs.get("n_sent", 0)), int(rule_kwargs.get("n_words", 0)))
    if instruction_id == "combination:two_responses":
        return _validate_two_responses(response)
    if instruction_id == "detectable_format:json_format":
        return _validate_json_format(response)
    if instruction_id == "detectable_format:constrained_response":
        return _validate_choice(response, prompt)
    return False


_IFBENCH_REGISTRY: dict[str, Any] | None = None
_IFEVAL_BACKEND: dict[str, Any] | None = None


def _load_ifbench_registry() -> dict[str, Any]:
    global _IFBENCH_REGISTRY
    if _IFBENCH_REGISTRY is not None:
        return _IFBENCH_REGISTRY
    existing_registry = sys.modules.get("instructions_registry")
    if existing_registry is not None:
        registry = getattr(existing_registry, "INSTRUCTION_DICT", None)
        if isinstance(registry, dict):
            _IFBENCH_REGISTRY = registry
            return _IFBENCH_REGISTRY
    if not IFBENCH_DIR.is_dir():
        _IFBENCH_REGISTRY = {}
        return _IFBENCH_REGISTRY
    try:
        for mod_name in ("instructions_util", "instructions", "instructions_registry", "evaluation_lib"):
            mod_path = IFBENCH_DIR / f"{mod_name}.py"
            spec = importlib.util.spec_from_file_location(mod_name, mod_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load {mod_name} from {mod_path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
        _IFBENCH_REGISTRY = sys.modules["instructions_registry"].INSTRUCTION_DICT
    except Exception:
        _IFBENCH_REGISTRY = {}
    return _IFBENCH_REGISTRY


def _load_ifeval_backend() -> dict[str, Any]:
    global _IFEVAL_BACKEND
    if _IFEVAL_BACKEND is not None:
        return _IFEVAL_BACKEND
    attempts: list[str] = []

    try:
        eval_mod = importlib.import_module("instruction_following_eval.evaluation_lib")
        registry_mod = importlib.import_module("instruction_following_eval.instructions_registry")
        _IFEVAL_BACKEND = {
            "kind": "google",
            "InputExample": getattr(eval_mod, "InputExample"),
            "strict": getattr(eval_mod, "test_instruction_following_strict"),
            "loose": getattr(eval_mod, "test_instruction_following_loose"),
            "registry": getattr(registry_mod, "INSTRUCTION_DICT", None),
        }
        return _IFEVAL_BACKEND
    except Exception as exc:
        attempts.append(f"instruction_following_eval.evaluation_lib: {exc}")

    try:
        eval_mod = importlib.import_module("evaluation_lib")
        _IFEVAL_BACKEND = {
            "kind": "google",
            "InputExample": getattr(eval_mod, "InputExample"),
            "strict": getattr(eval_mod, "test_instruction_following_strict"),
            "loose": getattr(eval_mod, "test_instruction_following_loose"),
            "registry": None,
        }
        return _IFEVAL_BACKEND
    except Exception as exc:
        attempts.append(f"evaluation_lib: {exc}")

    try:
        ifeval_mod = importlib.import_module("ifeval")
        _IFEVAL_BACKEND = {
            "kind": "ifeval",
            "Evaluator": getattr(ifeval_mod, "Evaluator"),
            "InputExample": getattr(ifeval_mod, "InputExample"),
            "instruction_registry": getattr(ifeval_mod, "instruction_registry"),
        }
        return _IFEVAL_BACKEND
    except Exception as exc:
        attempts.append(f"ifeval: {exc}")

    raise RuntimeError(
        "Official IFEval backend is required. Supported imports: "
        "instruction_following_eval.evaluation_lib, evaluation_lib, or ifeval. "
        f"Tried: {'; '.join(attempts)}"
    )


def _check_ifbench_instruction(instruction_id: str, rule_kwargs: dict[str, Any], prompt: str, response: str) -> bool:
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
    canon_pred = _canonicalize(predicted_value)
    canon_gt = _canonicalize(gt_value)
    if canon_pred == canon_gt:
        return True
    if isinstance(canon_gt, list):
        return any(_canonicalize(alt) == canon_pred for alt in canon_gt)
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
            if _canonicalize(pred_args) != _canonicalize(exp_args):
                return False
            continue
        if _canonicalize(pred_args) != _canonicalize(_unwrap_bfcl_args(exp_args)):
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
        if isinstance(expected_type, str) and not _check_python_type(_canonicalize(value), expected_type):
            invalid_keys += 1
            continue
        enum_values = prop_schema.get("enum")
        if isinstance(enum_values, list) and _canonicalize(value) not in [_canonicalize(item) for item in enum_values]:
            invalid_keys += 1
    if invalid_keys:
        penalties += 0.1 * (invalid_keys / max(len(predicted), 1))
    return penalties


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


def _strict_tool_score(
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


def _instruction_rule_scores(metadata: dict[str, Any], prompt_text: str, response: str) -> list[float]:
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        return []
    kwargs_list = _normalize_kwargs(metadata.get("kwargs"), len(instruction_ids))
    passed: list[float] = []
    for instruction_id, rule_kwargs in zip(instruction_ids, kwargs_list):
        try:
            passed.append(1.0 if _check_instruction(instruction_id, rule_kwargs, prompt_text, response) else 0.0)
        except Exception:
            passed.append(0.0)
    return passed


def _ifbench_rule_scores(metadata: dict[str, Any], prompt_text: str, response: str, *, strict: bool) -> list[float] | None:
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        return []
    _load_ifbench_registry()
    try:
        evaluation_lib = sys.modules["evaluation_lib"]
    except KeyError:
        return None
    kwargs_list = _normalize_kwargs(metadata.get("kwargs"), len(instruction_ids))
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
        prompt_to_response = {prompt_text: response}
        verifier = evaluation_lib.test_instruction_following_strict if strict else evaluation_lib.test_instruction_following_loose
        result = verifier(input_example, prompt_to_response)
    except Exception:
        return None
    follow_list = getattr(result, "follow_instruction_list", None)
    if not isinstance(follow_list, list):
        return None
    return [1.0 if bool(item) else 0.0 for item in follow_list]


def _ifeval_rule_scores(metadata: dict[str, Any], prompt_text: str, response: str, *, strict: bool) -> list[float] | None:
    backend = _load_ifeval_backend()
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        return []
    kwargs_list = _normalize_kwargs(metadata.get("kwargs"), len(instruction_ids))
    raw_key = metadata.get("record_id", metadata.get("key", 0))
    try:
        key = int(raw_key)
    except (TypeError, ValueError):
        key = hash(str(raw_key)) % (10**9)

    if backend["kind"] == "google":
        try:
            input_example = backend["InputExample"](
                key=key,
                instruction_id_list=instruction_ids,
                prompt=prompt_text,
                kwargs=kwargs_list,
            )
            prompt_to_response = {prompt_text: response}
            verifier = backend["strict"] if strict else backend["loose"]
            result = verifier(input_example, prompt_to_response)
        except Exception:
            return None
        follow_list = getattr(result, "follow_instruction_list", None)
        if not isinstance(follow_list, list):
            return None
        return [1.0 if bool(item) else 0.0 for item in follow_list]

    try:
        evaluator = backend["Evaluator"](backend["instruction_registry"])
        input_example = backend["InputExample"](
            key=key,
            instruction_id_list=instruction_ids,
            prompt=prompt_text,
            kwargs=kwargs_list,
        )
        report, outputs = evaluator.evaluate([input_example], {prompt_text: response})
    except Exception:
        return None
    if not isinstance(outputs, list) or not outputs:
        return None
    follow_list = getattr(outputs[0], "follow_instruction_list", None)
    if not isinstance(follow_list, list):
        return None
    return [1.0 if bool(item) else 0.0 for item in follow_list]


def compute_generic_reward(metadata: dict[str, Any], prompt_text: str, response: str) -> float:
    reward_type = str(metadata.get("reward_type", "")).strip()
    cleaned = _clean_response(response)
    if not cleaned:
        return 0.0

    if reward_type == "bfcl_official":
        raise RuntimeError(
            "BFCL official evaluation is not supported through the generic multidomain reward router. "
            "Use the BFCL official runner."
        )

    if reward_type == "stem_mcqa":
        return 1.0 if _extract_mcqa_answer(cleaned) == str(metadata.get("answer", "")).strip().upper() else 0.0

    if reward_type in {
        "tool_call_soft",
        "tool_call_strict",
        "tool_call",
        "tool_selection_strict",
        *TRAIN_TOOL_REWARD_TYPES,
    }:
        if reward_type == "api_call_text":
            return 1.0 if _normalize_python_call_text(cleaned) == _normalize_python_call_text(str(metadata.get("raw_api_call") or "")) else 0.0
        tools = metadata.get("tools") or []
        expected_calls = metadata.get("ground_truth") or []
        allowed_tool_names = metadata.get("allowed_tool_names") or []
        if not tools or (not expected_calls and reward_type != "tool_selection_strict"):
            return 0.0
        try:
            predicted_calls = _parse_predicted_calls(
                cleaned,
                tools=tools,
                parser_type=str(metadata.get("parser_type", "qwen25")),
            )
        except Exception:
            return 0.0
        if reward_type == "tool_selection_strict":
            return _tool_selection_strict_score(predicted_calls, allowed_tool_names)
        if reward_type in {"tool_call_soft", *TRAIN_TOOL_REWARD_TYPES}:
            return _tool_call_soft_score(predicted_calls, expected_calls, tools)
        return _strict_tool_score(predicted_calls, expected_calls, tools)

    if reward_type == "structured_json_schema":
        try:
            payload = json.loads(_extract_json_payload(cleaned))
        except Exception:
            return 0.0
        schema = metadata.get("schema") if isinstance(metadata.get("schema"), dict) else {}
        return 1.0 if _validate_json_schema_full(payload, schema) else 0.0

    if reward_type in {"instruction_following_soft", "instruction_following_strict", "ifeval"}:
        dataset_name = str(metadata.get("dataset_name", "")).strip().lower()
        if dataset_name == "ifbench_test":
            scores = _ifbench_rule_scores(metadata, prompt_text, cleaned, strict=reward_type != "instruction_following_soft")
            if scores is None:
                raise RuntimeError(
                    "IFBench official evaluation_lib is required for ifbench_test. "
                    "Heuristic fallback has been removed."
                )
        elif dataset_name == "ifeval" or reward_type == "ifeval":
            scores = _ifeval_rule_scores(metadata, prompt_text, cleaned, strict=reward_type != "instruction_following_soft")
            if scores is None:
                raise RuntimeError(
                    "Official IFEval backend is required for ifeval. "
                    "Heuristic fallback has been removed."
                )
        else:
            scores = _instruction_rule_scores(metadata, prompt_text, cleaned)
        if not scores:
            return 0.0
        if reward_type == "instruction_following_soft":
            return sum(scores) / len(scores)
        return 1.0 if all(score == 1.0 for score in scores) else 0.0

    return 0.0


async def reward_func(args, sample, **kwargs) -> float:
    if sample.status == sample.Status.TRUNCATED:
        return 0.0
    metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
    if "tools" not in metadata:
        tools = getattr(sample, "tools", None)
        if isinstance(tools, list):
            metadata["tools"] = tools
    prompt = metadata.get("prompt_text") if metadata.get("prompt_text") not in (None, "") else (sample.prompt or "")
    if not isinstance(prompt, str):
        prompt = str(prompt)
    return compute_generic_reward(metadata, prompt, sample.response or "")
