from __future__ import annotations

import re

from math_verify import StringExtractionConfig, parse as math_parse, verify as math_verify_fn  # noqa: E402
from slime.rollout.rm_hub.math_utils import extract_answer  # noqa: E402


def _normalize_text_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _parse_math_text(raw: str):
    raw = str(raw).strip()
    if not raw:
        return []

    candidates = [raw]
    if not (raw.startswith("$") and raw.endswith("$")):
        candidates.append(f"${raw}$")

    if re.fullmatch(r"[A-Za-z]+", raw):
        for candidate in candidates:
            parsed = math_parse(
                candidate,
                extraction_config=[StringExtractionConfig()],
                parsing_timeout=None,
            )
            if parsed:
                return parsed
        return []

    for candidate in candidates:
        parsed = math_parse(candidate, parsing_timeout=None)
        if parsed:
            return parsed
    return []


async def reward_func(args, sample, **kwargs):
    if sample.status == sample.Status.TRUNCATED:
        return 0.0

    response = sample.response or ""
    label = sample.label
    if not label:
        return 0.0

    predicted_answer = extract_answer(response)
    if predicted_answer is None:
        return 0.0

    gold_answer = str(label).strip()

    parsed_pred = _parse_math_text(predicted_answer)
    parsed_gold = _parse_math_text(gold_answer)

    if parsed_pred and parsed_gold:
        try:
            return 1.0 if math_verify_fn(parsed_pred, parsed_gold) else 0.0
        except Exception:
            pass

    # Fallback only for plain text answers that math-verify cannot parse, e.g. Yes/No.
    return 1.0 if _normalize_text_answer(predicted_answer) == _normalize_text_answer(gold_answer) else 0.0
