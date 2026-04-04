"""MOPD eval reward router — dispatches by domain to the correct verifier.

math       → reward_deepmath_mathverify  (math-verify)
code       → reward_code_execution       (code sandbox)
tool/stem/structured → improved eval verifiers (jsonschema, better MCQ extraction)
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Make local reward modules importable.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from reward_code_execution import reward_func as _code_reward  # noqa: E402
from reward_deepmath_mathverify import reward_func as _math_reward  # noqa: E402

# Import multidomain_v1 helpers (tool call parsing, canonicalization, etc.)
from reward_multidomain_v1 import (  # noqa: E402
    _ifbench_rule_scores,
    _instruction_rule_scores,
    _parse_predicted_calls,
    _calls_match_ground_truth_bfcl,
    _tool_call_soft_score,
    _tool_selection_strict_score,
    _tool_schema_map,
)

# Import v0 helpers
_ROOT = Path(__file__).resolve().parents[1]
_V0_DIR = _ROOT / "multidomain_v0"
if str(_V0_DIR) not in sys.path:
    sys.path.insert(0, str(_V0_DIR))

from reward_mixed_domain import (  # noqa: E402
    _clean_response,
    _extract_json_payload,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# jsonschema (available in environment)
# ---------------------------------------------------------------------------
try:
    import jsonschema as _jsonschema

    def _validate_json_schema_full(payload: Any, schema: dict) -> bool:
        try:
            _jsonschema.validate(payload, schema)
            return True
        except (_jsonschema.ValidationError, _jsonschema.SchemaError):
            return False
except ImportError:
    _jsonschema = None
    logger.warning("jsonschema not available; falling back to basic JSON Schema validator")

    from reward_mixed_domain import _validate_json_schema  # noqa: E402

    def _validate_json_schema_full(payload: Any, schema: dict) -> bool:
        return _validate_json_schema(payload, schema)


# ---------------------------------------------------------------------------
# Improved STEM MCQ answer extraction
# ---------------------------------------------------------------------------

def _extract_mcqa_answer(response: str) -> str:
    """Extract MCQ answer letter from a model response.

    Handles many common output formats:
      - Answer: X / The answer is X / correct answer is X
      - <answer>X</answer>
      - \\boxed{X}
      - **X** / (X) / [X] at end of response
      - Single letter on last line
    """
    # 1. Explicit "Answer: X" pattern
    match = re.search(r"(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]*([A-Z])\b", response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"Answer\s*:\s*([A-Z])\b", response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. <answer>X</answer> tag
    match = re.search(r"<answer>\s*([A-Z])\s*</answer>", response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. \boxed{X}
    match = re.search(r"\\boxed\{([A-Z])\}", response)
    if match:
        return match.group(1).upper()

    # 4. Look at the last meaningful line for common patterns
    stripped = response.strip().splitlines()
    if stripped:
        last = stripped[-1].strip()

        # Single letter
        if re.fullmatch(r"[A-Za-z]", last):
            return last.upper()

        # **X** or *X*
        match = re.fullmatch(r"\*{1,2}([A-Za-z])\*{1,2}", last)
        if match:
            return match.group(1).upper()

        # (X) or [X]
        match = re.fullmatch(r"[(\[]([A-Za-z])[)\]]", last)
        if match:
            return match.group(1).upper()

        # X. or X)
        match = re.fullmatch(r"([A-Za-z])[.)]\s*", last)
        if match:
            return match.group(1).upper()

        # Line ending with a single letter after colon/space
        match = re.search(r"[:\s]([A-Z])\s*\.?\s*$", last)
        if match:
            return match.group(1).upper()

    return ""


# ---------------------------------------------------------------------------
# Improved BFCL scoring — adds jsonschema argument validation
# ---------------------------------------------------------------------------

def _bfcl_strict_with_schema_check(
    predicted_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> float:
    """BFCL strict scoring + jsonschema validation of argument types."""
    # First check structural match (existing logic)
    if not _calls_match_ground_truth_bfcl(predicted_calls, expected_calls):
        return 0.0

    # Additionally validate predicted args against tool parameter schemas
    if _jsonschema is not None:
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


def _eval_ifeval(metadata: dict[str, Any], sample, reward_type: str) -> float:
    """IFEval/IFBench evaluation routed by dataset semantics."""
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


# ---------------------------------------------------------------------------
# Main eval reward function for tool/stem/structured
# ---------------------------------------------------------------------------

async def _eval_multidomain_reward(args, sample, **kwargs) -> float:
    """Improved multidomain reward for eval — uses jsonschema, better MCQ extraction."""
    if sample.status == sample.Status.TRUNCATED:
        return 0.0

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    reward_type = str(metadata.get("reward_type", ""))
    response = _clean_response(sample.response or "")
    if not response:
        return 0.0

    # ---- STEM MCQ ----
    if reward_type == "stem_mcqa":
        return 1.0 if _extract_mcqa_answer(response) == str(metadata.get("answer", "")).strip().upper() else 0.0

    if reward_type == "bfcl_official":
        raise RuntimeError(
            "BFCL official evaluation is not available through the generic MOPD reward router. "
            "Use the official BFCL evaluator over pool native rows instead."
        )

    # ---- Tool calling ----
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
        # tool_call_strict / tool_call — improved with jsonschema validation
        return _bfcl_strict_with_schema_check(predicted_calls, expected_calls, tools)

    # ---- Structured: JSON Schema — use jsonschema library ----
    if reward_type == "structured_json_schema":
        try:
            payload = json.loads(_extract_json_payload(response))
        except Exception:
            return 0.0
        schema = metadata.get("schema") if isinstance(metadata.get("schema"), dict) else {}
        return 1.0 if _validate_json_schema_full(payload, schema) else 0.0

    # ---- Instruction following (IFEval / IFBench) ----
    if reward_type in {"instruction_following_soft", "instruction_following_strict", "ifeval"}:
        return _eval_ifeval(metadata, sample, reward_type)

    return 0.0


# ---------------------------------------------------------------------------
# Domain router
# ---------------------------------------------------------------------------

def _infer_domain(sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(metadata.get("domain") or "").strip().lower()


async def reward_func(args, sample, **kwargs):
    domain = _infer_domain(sample)
    if domain == "math":
        return await _math_reward(args, sample, **kwargs)
    if domain == "code":
        return await _code_reward(args, sample, **kwargs)
    # tool, stem, structured — improved eval verifiers
    return await _eval_multidomain_reward(args, sample, **kwargs)
