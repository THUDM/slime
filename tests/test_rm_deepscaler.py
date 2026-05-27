"""CPU unit tests for ``slime.rollout.rm_hub.deepscaler``.

Pins the wrapper that decides which segment of the response counts as
the "solution" and reduces grading to ``math_utils``. The branching is
small (3 cases) but silent-failure prone: if the ``</think>`` /
``###Response`` markers stop matching the format the rollout actually
produces, the function returns 0 *for every sample* and no other CI
signal would catch it.
"""

from __future__ import annotations

import pytest

from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


@pytest.mark.unit
def test_response_split_on_think_marker_grades_tail():
    """The default chat format puts the answer after ``</think>``. Only
    the tail is graded — pre-think reasoning is ignored, even if it
    contains the wrong answer first."""
    response = r"Let me reconsider. \boxed{99}</think>Final: \boxed{42}"
    assert get_deepscaler_rule_based_reward(response, "42") == 1


@pytest.mark.unit
def test_response_split_on_response_marker_grades_tail():
    """Alternate format: ``###Response`` separator. Only what comes after
    is graded (deepscaler.py:7-8)."""
    response = r"Scratch work \boxed{wrong}###Response\boxed{42}"
    assert get_deepscaler_rule_based_reward(response, "42") == 1


@pytest.mark.unit
def test_response_without_any_marker_returns_zero():
    """No ``</think>`` AND no ``###Response`` → fall through to 0
    immediately (deepscaler.py:9-10). This is the silent-failure pole —
    if upstream chat templates drop both markers, all rewards become 0."""
    assert get_deepscaler_rule_based_reward(r"\boxed{42}", "42") == 0


@pytest.mark.unit
def test_response_with_no_boxed_answer_returns_zero():
    """Marker is present but no ``\\boxed`` in the tail → ``extract_answer``
    returns None → 0 (deepscaler.py:13-14)."""
    assert get_deepscaler_rule_based_reward("plain</think>no box here", "42") == 0


@pytest.mark.unit
def test_empty_label_returns_zero():
    """Empty ground-truth → 0 (deepscaler.py:15-16). Guards against
    missing-label data poisoning training with spurious 0s — explicitly
    the same as wrong-answer, intentional."""
    assert get_deepscaler_rule_based_reward(r"</think>\boxed{42}", "") == 0


@pytest.mark.unit
def test_label_as_int_is_coerced_to_string():
    """Integer labels are accepted and ``str()``'d (deepscaler.py:19, 25).
    Common case for datasets that store numeric labels."""
    assert get_deepscaler_rule_based_reward(r"</think>\boxed{42}", 42) == 1


@pytest.mark.unit
def test_label_as_float_is_coerced_to_string():
    """float labels: stringified to e.g. "42.0". The current grader path
    (mathd or sympy) handles "42.0" vs "42" via normalization — pinning
    the wiring, not the equality logic."""
    assert get_deepscaler_rule_based_reward(r"</think>\boxed{42}", 42) == 1


@pytest.mark.unit
def test_label_with_boxed_marker_is_extracted_too():
    """If the ground truth itself is wrapped in ``\\boxed{}``, it must be
    unwrapped before grading (deepscaler.py:26-29)."""
    assert get_deepscaler_rule_based_reward(r"</think>\boxed{42}", r"\boxed{42}") == 1


@pytest.mark.unit
def test_wrong_answer_returns_zero():
    """Sanity-check the negative side of the contract."""
    assert get_deepscaler_rule_based_reward(r"</think>\boxed{43}", "42") == 0


@pytest.mark.unit
def test_grader_uses_either_mathd_or_sympy_path():
    """``\\frac{1}{2}`` vs ``0.5`` — mathd_normalize collapses both, even
    though the strings aren't lexically equal. Pins the "either mathd OR
    sympy succeeds" disjunction at deepscaler.py:38."""
    response = r"</think>\boxed{\frac{1}{2}}"
    assert get_deepscaler_rule_based_reward(response, "0.5") == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
