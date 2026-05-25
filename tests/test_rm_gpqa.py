"""CPU unit tests for ``slime.rollout.rm_hub.gpqa``.

Pins the GPQA rule-based scorer used by the ``gpqa`` rm_type. The whole
pipeline is pure-Python regex/string normalization with multiple
fall-through branches (extract-letter → match-correct-letter → text-
contains fallback), so a regex tweak or branch-order change can silently
shift rewards from 1.0 → 0.0 without any other CI signal.

Test surface:

  - ``_extract_letter_from_response``: each of the three named patterns,
    the standalone-letter fallback (last valid letter wins), and the
    ``</think>`` chain-of-thought stripper.
  - ``compute_gpqa_reward``: every label shape it accepts
    (single-letter str, int index, full-text label) and every metadata
    shape (``choices`` as list / dict, explicit ``correct_letter``,
    explicit ``valid_letters``).

Together these cover the silent-wrong-reward classes that string-parsing
scorers regress into.
"""

from __future__ import annotations

import pytest

from slime.rollout.rm_hub.gpqa import DEFAULT_VALID_LETTERS, _extract_letter_from_response, compute_gpqa_reward

# ---------------------------------------------------------------------------
# _extract_letter_from_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "response,expected",
    [
        # Pattern 1: "answer is X" / "option: X" / "choice X"
        ("The answer is B.", "B"),
        ("Answer: C", "C"),
        ("My choice is D", "D"),
        # Pattern 2: "X is correct"
        ("A is correct here", "A"),
        ("E is the correct option", "E"),
        # Pattern 3: "final answer X"
        ("Final answer: B", "B"),
        ("the final option is C", "C"),
        # Fallback: last standalone capital letter in valid set
        ("we ruled out A, then B, settled on C", "C"),
    ],
)
def test_extract_letter_named_patterns_and_fallback(response, expected):
    assert _extract_letter_from_response(response, DEFAULT_VALID_LETTERS) == expected


@pytest.mark.unit
def test_extract_letter_strips_chain_of_thought_before_matching():
    """``</think>`` marker → keep only the trailing segment. Without the
    strip, the earlier "Answer: A" would win over the real answer "B"."""
    response = "Let me think… Answer: A is wrong.</think>The answer is B."
    assert _extract_letter_from_response(response, DEFAULT_VALID_LETTERS) == "B"


@pytest.mark.unit
def test_extract_letter_returns_none_on_no_match():
    """No pattern hit AND no valid standalone capital → None."""
    assert _extract_letter_from_response("no idea, sorry", DEFAULT_VALID_LETTERS) is None


@pytest.mark.unit
def test_extract_letter_returns_none_on_empty():
    assert _extract_letter_from_response("", DEFAULT_VALID_LETTERS) is None
    assert _extract_letter_from_response(None, DEFAULT_VALID_LETTERS) is None


@pytest.mark.unit
def test_extract_letter_respects_valid_letters_restriction():
    """Restricting valid letters to {A, B} → an ``answer is C`` match is
    rejected and the standalone-fallback used instead."""
    # The named-pattern match catches "C" but it's invalid; the fallback
    # then walks standalone letters in reverse — "B" wins.
    response = "Answer: C, but actually A was right, no wait B."
    assert _extract_letter_from_response(response, ["A", "B"]) == "B"


# ---------------------------------------------------------------------------
# compute_gpqa_reward
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reward_letter_label_match():
    """label is a single capital letter, no choices needed — exact letter match
    short-circuits to 1.0."""
    reward = compute_gpqa_reward("The answer is C.", label="C")
    assert reward == 1.0


@pytest.mark.unit
def test_reward_letter_label_mismatch_returns_zero():
    assert compute_gpqa_reward("The answer is C.", label="D") == 0.0


@pytest.mark.unit
def test_reward_int_label_maps_via_choices_length():
    """``label=2`` with 4 choices → valid_letters[2] = "C". Extracting "C"
    from response should land on 1.0."""
    reward = compute_gpqa_reward("Answer: C", label=2, metadata={"choices": ["alpha", "beta", "gamma", "delta"]})
    assert reward == 1.0


@pytest.mark.unit
def test_reward_choices_as_dict_is_accepted():
    """Some pipelines pass ``choices`` as an ordered dict — code path at
    gpqa.py:62-63 unpacks via ``.values()``."""
    reward = compute_gpqa_reward(
        "Answer: B",
        label=1,
        metadata={"choices": {"a": "first", "b": "second", "c": "third"}},
    )
    assert reward == 1.0


@pytest.mark.unit
def test_reward_full_text_label_resolves_to_letter_via_choices():
    """label is the answer *text* — code matches normalized label against
    each choice and resolves the position to a letter, then extracts the
    letter from the response."""
    reward = compute_gpqa_reward(
        "I think the answer is B.",
        label="capital of france",
        metadata={"choices": ["London", "Capital of France", "Berlin"]},
    )
    assert reward == 1.0


@pytest.mark.unit
def test_reward_text_match_fallback_when_no_letter_extracted():
    """No letter pattern, but the response contains the answer text — the
    fallback at gpqa.py:122-124 matches normalized text containment."""
    reward = compute_gpqa_reward(
        "Definitely paris.",
        label="Paris",
        metadata={"choices": ["London", "Paris", "Berlin"]},
    )
    assert reward == 1.0


@pytest.mark.unit
def test_reward_correct_letter_metadata_overrides_label():
    """``metadata.correct_letter`` takes precedence over label-derived
    letter (gpqa.py:75-79). Useful for datasets where the label is the
    full text but the correct option letter is pre-computed."""
    reward = compute_gpqa_reward(
        "Answer: A",
        label="anything",
        metadata={"correct_letter": "a"},  # lowercase ok — gets upper()'d
    )
    assert reward == 1.0


@pytest.mark.unit
def test_reward_none_response_is_zero():
    """A nil response (custom_generate failure) must score 0, not crash."""
    assert compute_gpqa_reward(None, label="A") == 0.0


@pytest.mark.unit
def test_reward_no_correct_letter_and_no_match_is_zero():
    """No metadata, label is non-letter text, response doesn't contain it —
    every branch falls through to 0.0 at gpqa.py:129."""
    assert compute_gpqa_reward("random text", label="some answer") == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
