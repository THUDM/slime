"""CPU unit tests for ``slime.rollout.rm_hub.math_utils``.

Pins the boxed-answer extraction + lenient normalization used by every
math-style rm_type (math / deepscaler / dapo all funnel here). The
extraction is a hand-written brace counter; ``_strip_string`` is a long
chain of LaTeX-normalization regexes/replaces. Both are prime targets
for silent regressions — a wrong brace count or one bad replace can
quietly flip rewards from 1 → 0.

This file covers the dependency-free pieces (pure string ops). The sympy
branch (``grade_answer_sympy``, ``are_equal_under_sympy``) is exercised
end-to-end through ``grade_answer_verl`` happy-path cases so we don't
have to re-derive sympy's normalization rules — just confirm the wiring
holds.
"""

from __future__ import annotations

import pytest

from slime.rollout.rm_hub.math_utils import (
    _strip_string,
    extract_answer,
    extract_boxed_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    grade_answer_verl,
    last_boxed_only_string,
    mathd_normalize_answer,
    remove_boxed,
)


# ---------------------------------------------------------------------------
# last_boxed_only_string — hand-rolled brace counter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_last_boxed_returns_last_when_multiple():
    """When two boxed expressions exist, ``rfind`` picks the last one."""
    s = r"first attempt \boxed{wrong}, final \boxed{42}"
    assert last_boxed_only_string(s) == r"\boxed{42}"


@pytest.mark.unit
def test_last_boxed_handles_nested_braces():
    """Brace counter must balance — nested braces inside the boxed expr
    should be included, not cause early termination."""
    s = r"answer: \boxed{\frac{1}{2}}"
    assert last_boxed_only_string(s) == r"\boxed{\frac{1}{2}}"


@pytest.mark.unit
def test_last_boxed_falls_back_to_fbox():
    """If no ``\\boxed`` is present, the function also accepts ``\\fbox``."""
    s = r"answer: \fbox{7}"
    assert last_boxed_only_string(s) == r"\fbox{7}"


@pytest.mark.unit
def test_last_boxed_returns_none_when_missing():
    assert last_boxed_only_string("plain text, no box") is None


@pytest.mark.unit
def test_last_boxed_returns_none_on_unterminated_box():
    """Open brace never closes → braces stay imbalanced → return None
    (not e.g. raise IndexError)."""
    assert last_boxed_only_string(r"start \boxed{never closes") is None


# ---------------------------------------------------------------------------
# remove_boxed — strip the wrapper, leave the content
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_remove_boxed_strips_wrapper():
    assert remove_boxed(r"\boxed{x+1}") == "x+1"


@pytest.mark.unit
def test_remove_boxed_preserves_inner_braces():
    """Brace count is irrelevant here — the function just strips the prefix
    and final ``}``; nested braces stay."""
    assert remove_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


@pytest.mark.unit
def test_remove_boxed_returns_none_on_malformed_input():
    """math_utils.remove_boxed wraps its asserts in try/except (412-419) —
    silently returns None on bad input. ``math_dapo_utils.remove_boxed``
    behaves differently (raises); see test_rm_math_dapo for that contract.
    """
    assert remove_boxed("not boxed at all") is None
    assert remove_boxed(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# extract_boxed_answer / extract_answer — the convenience composition
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_boxed_answer_end_to_end():
    assert extract_boxed_answer(r"Solution: \boxed{42}") == "42"


@pytest.mark.unit
def test_extract_answer_returns_none_when_no_boxed_marker():
    """``extract_answer`` only triggers when ``\\boxed`` is in the passage;
    otherwise returns None (pinning the branch at math_utils.py:479)."""
    assert extract_answer("just 42") is None


# ---------------------------------------------------------------------------
# mathd_normalize_answer / _strip_string — Hendrycks-style normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        # \\frac shorthand → \frac{a}{b}; non-frac stays
        (r"\frac12", r"\frac{1}{2}"),
        # tfrac/dfrac → frac
        (r"\tfrac{1}{2}", r"\frac{1}{2}"),
        # \left/\right wrappers removed
        (r"\left(1\right)", "(1)"),
        # degree symbol stripped (both variants)
        (r"45^{\circ}", "45"),
        (r"45^\circ", "45"),
        # leading 0 added to .N → 0.N (but NOT .5 — that collides with
        # the 0.5 → \frac{1}{2} special-case below; use .7 to isolate
        # the leading-zero rule).
        (".7", "0.7"),
        # 0.5 → \frac{1}{2} convenience replacement (pinning the special
        # case at math_utils.py:153-154). Note: .5 also lands here because
        # the leading-zero rule runs first.
        ("0.5", r"\frac{1}{2}"),
        (".5", r"\frac{1}{2}"),  # composition of leading-zero + 0.5 special
        # a/b → \frac{a}{b}
        ("3/4", r"\frac{3}{4}"),
        # spaces collapsed away
        ("1 + 1", "1+1"),
    ],
)
def test_strip_string_canonical_substitutions(raw, expected):
    assert _strip_string(raw) == expected


@pytest.mark.unit
def test_mathd_normalize_strips_text_wrapper():
    """A ``\\text{...}`` enclosing the whole answer is unwrapped before
    further normalization (math_utils.py:21-23)."""
    assert mathd_normalize_answer(r"\text{42}") == "42"


@pytest.mark.unit
def test_mathd_normalize_none_passthrough():
    assert mathd_normalize_answer(None) is None


# ---------------------------------------------------------------------------
# grade_answer_mathd — pure lexical equality after normalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "given,gt",
    [
        (r"\frac{1}{2}", "0.5"),  # 0.5 → \frac{1}{2}
        (r"\frac12", r"\frac{1}{2}"),  # \frac shorthand expansion
        (r"45^\circ", "45"),  # degree marker stripped on both sides
        ("3/4", r"\frac{3}{4}"),  # a/b normalization
    ],
)
def test_grade_answer_mathd_canonical_equivalences(given, gt):
    assert grade_answer_mathd(given, gt) is True


@pytest.mark.unit
def test_grade_answer_mathd_rejects_different():
    assert grade_answer_mathd("42", "43") is False


# ---------------------------------------------------------------------------
# grade_answer_sympy / grade_answer_verl — sympy-backed paths (happy-path
# only; this isn't a sympy regression suite, just wiring confirmation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_grade_answer_sympy_equivalent_expressions():
    """Symbolically equal expressions should compare equal under sympy."""
    assert grade_answer_sympy("x+1", "1+x") is True


@pytest.mark.unit
def test_grade_answer_sympy_fraction_must_match_exactly():
    """Reducible fractions are intentionally NOT considered equal — pinning
    the explicit ``_is_frac`` short-circuit at math_utils.py:453-456."""
    assert grade_answer_sympy(r"\frac{2}{4}", r"\frac{1}{2}") is False


@pytest.mark.unit
def test_grade_answer_verl_extracts_both_sides_from_boxed():
    """Both solution and ground_truth carry ``\\boxed{...}`` markers — the
    function extracts then grades. Confirms the wiring at
    math_utils.py:488-492 not just the inner mathd/sympy logic."""
    assert grade_answer_verl(r"answer: \boxed{42}", r"\boxed{42}") is True


@pytest.mark.unit
def test_grade_answer_verl_returns_false_on_missing_extraction():
    """No ``\\boxed`` in the solution → ``given_answer`` is None → False
    (math_utils.py:491-492)."""
    assert grade_answer_verl("just 42", "42") is False


@pytest.mark.unit
def test_grade_answer_verl_returns_false_on_empty_ground_truth():
    """Empty / falsy ground truth → False at the top guard."""
    assert grade_answer_verl(r"\boxed{42}", "") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
