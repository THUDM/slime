"""CPU unit tests for ``slime.rollout.rm_hub.math_dapo_utils``.

Pins the DAPO math scorer (``rm_type=dapo``). Distinct from
``math_utils`` in three ways the tests need to lock down:

  - ``remove_boxed`` here raises ``AssertionError`` on malformed input
    (vs ``math_utils.remove_boxed`` which returns None silently)
  - ``normalize_final_answer`` has its own pipeline (SUBSTITUTIONS list +
    REMOVED_EXPRESSIONS list + per-unit regexes); silent drift here
    causes wrong predicate matches in ``is_correct_minerva``
  - ``compute_score`` only considers the last 300 chars (efficiency
    truncation at line 280) — a regression that drops this would silently
    blow up scoring time on long traces
"""

from __future__ import annotations

import pytest

from slime.rollout.rm_hub.math_dapo_utils import (
    compute_score,
    is_correct_minerva,
    is_correct_strict_box,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
    verify,
)


NUM_GPUS = 0


# ---------------------------------------------------------------------------
# last_boxed_only_string — brace counter (separate impl from math_utils;
# this one expects ``\boxed{`` specifically, no \fbox fallback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_last_boxed_picks_rightmost():
    assert last_boxed_only_string(r"\boxed{first}, then \boxed{42}") == r"\boxed{42}"


@pytest.mark.unit
def test_last_boxed_balances_nested_braces():
    """Nested ``{}`` inside the box must not terminate the counter early."""
    assert last_boxed_only_string(r"\boxed{\frac{1}{2}}") == r"\boxed{\frac{1}{2}}"


@pytest.mark.unit
def test_last_boxed_returns_none_when_missing():
    assert last_boxed_only_string("no box") is None


@pytest.mark.unit
def test_last_boxed_returns_none_on_unterminated():
    """Imbalanced braces → ``right_brace_idx`` stays None → return None
    (math_dapo_utils.py:47)."""
    assert last_boxed_only_string(r"\boxed{never closes") is None


# ---------------------------------------------------------------------------
# remove_boxed — distinct from math_utils: this raises on bad input
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_remove_boxed_strips_wrapper():
    assert remove_boxed(r"\boxed{42}") == "42"


@pytest.mark.unit
def test_remove_boxed_raises_on_malformed():
    """Unlike ``math_utils.remove_boxed`` (try/except → None), the dapo
    version asserts (line 60-61). A consumer that catches None will
    silently break if these two implementations are later unified."""
    with pytest.raises(AssertionError, match="box error"):
        remove_boxed("not boxed")


# ---------------------------------------------------------------------------
# normalize_final_answer — substitutions + removals + regex pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        # "x = answer" → "answer" (split on '=' takes the last segment)
        ("x = 42", "42"),
        # SUBSTITUTIONS: \$ → "" (currency removal), commas in "text and" → ","
        # Articles stripped — "an apple" → "apple"
        ("an answer", "answer"),
        # Unit removal: "square", "ways", etc. dropped silently
        ("42 square", "42"),
        ("100 dollars", "100"),
        # \text{...} unwrapped via regex
        (r"\text{hello}", "hello"),
        # \boxed{...} unwrapped via regex (NOT via remove_boxed; lighter touch)
        (r"\boxed{42}", "42"),
        # Plain integer with commas → commas stripped
        ("1,234,567", "1234567"),
        # Spaces stripped via SUBSTITUTIONS (" " → "")
        ("a b c", "bc"),  # "a " is also substituted away (articles)
    ],
)
def test_normalize_final_answer_canonical_substitutions(raw, expected):
    assert normalize_final_answer(raw) == expected


@pytest.mark.unit
def test_normalize_final_answer_strips_end_tokens():
    """Generation end-markers should be stripped (REMOVED_EXPRESSIONS
    line 138-139)."""
    assert normalize_final_answer("42<|endoftext|>") == "42"
    assert normalize_final_answer("42<|im_end|>") == "42"


# ---------------------------------------------------------------------------
# is_correct_strict_box — extract & exact-match within last 100 chars
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_correct_strict_box_matches():
    """Boxed prediction at end → extracted and compared to gt."""
    score, pred = is_correct_strict_box(r"long preamble … \boxed{42}", "42")
    assert (score, pred) == (1, "42")


@pytest.mark.unit
def test_is_correct_strict_box_mismatch_returns_minus_one():
    """``compute_score`` later maps {1: 1.0, -1: -1.0} — pin both poles
    here so the dispatching doesn't drift."""
    score, pred = is_correct_strict_box(r"\boxed{43}", "42")
    assert (score, pred) == (-1, "43")


@pytest.mark.unit
def test_is_correct_strict_box_no_box_returns_minus_one():
    """No box at all → extracted_pred is None → mismatch path → -1."""
    score, pred = is_correct_strict_box("plain 42", "42")
    assert score == -1
    assert pred is None


# ---------------------------------------------------------------------------
# is_correct_minerva — regex extract via "Answer:" then dapo-normalize
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_correct_minerva_matches_int_answer():
    """Minerva path expects gt to coerce via ``int(float(gt))`` (line 210),
    so floats / int-strings collapse to canonical int strings before
    comparison."""
    correct, pred = is_correct_minerva("Long solution. Answer: 42", "42")
    assert correct is True
    assert pred == "42"


@pytest.mark.unit
def test_is_correct_minerva_takes_last_answer_match():
    """Multiple ``Answer:`` lines → ``re.findall`` returns the list and
    the function picks ``[-1]`` (line 201). Pinning this means a model
    that re-states the answer at the end is still graded on the final one.

    Note: the regex captures up to a newline (``[^\\n]+``), so the two
    candidates MUST be on separate lines — otherwise the first ``Answer:``
    greedily swallows the rest including the second one.
    """
    text = "Answer: 41 was a wrong guess.\nAnswer: 42"
    correct, pred = is_correct_minerva(text, "42")
    assert correct is True


@pytest.mark.unit
def test_is_correct_minerva_no_answer_marker_is_invalid():
    """No ``Answer: ...`` in text → ``[INVALID]`` sentinel → mismatch."""
    correct, _ = is_correct_minerva("just a number 42", "42")
    assert correct is False


# ---------------------------------------------------------------------------
# verify + compute_score — top-level public scoring entrypoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_verify_strict_box_dispatches_to_strict_box():
    """``strict_box_verify=True`` → strict-box path; correct = (score == 1)."""
    correct, pred = verify(r"\boxed{42}", "42", strict_box_verify=True)
    assert correct is True
    assert pred == "42"


@pytest.mark.unit
def test_verify_default_dispatches_to_minerva():
    correct, pred = verify("Answer: 42", "42")
    assert correct is True


@pytest.mark.unit
def test_compute_score_correct_returns_dict_with_reward_one():
    """Public API contract: dict shape with {score, acc, pred} keys."""
    out = compute_score(r"\boxed{42}", "42", strict_box_verify=True)
    assert out == {"score": 1.0, "acc": True, "pred": "42"}


@pytest.mark.unit
def test_compute_score_incorrect_returns_minus_one():
    """The -1 reward (not 0) is the deliberate signal for "wrong" — pins
    the explicit ``-1.0`` at line 285."""
    out = compute_score(r"\boxed{43}", "42", strict_box_verify=True)
    assert out["score"] == -1.0
    assert out["acc"] is False


@pytest.mark.unit
def test_compute_score_only_uses_last_300_chars():
    """Efficiency truncation at line 280: only the tail is verified. A
    correct boxed answer earlier in the string but absent in the tail
    must score as incorrect — that's the explicit design choice."""
    # Put the correct boxed expr at the very start, then 400 chars of noise.
    sol = r"\boxed{42}" + (" filler" * 60)  # >300 chars after the box
    out = compute_score(sol, "42", strict_box_verify=True)
    assert out["score"] == -1.0  # truncated away → not found


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
