"""CPU unit tests for ``slime.rollout.rm_hub.f1``.

Pins the token-F1 contract used by the ``f1`` rm_type. The whole pipeline
is pure Python (regex + ``collections.Counter``), so any silent drift
here directly distorts a training run's reward signal without touching a
crash log. Cover the four shapes a reward consumer cares about:

  - normalize_answer: article-strip + punctuation-strip + lowercase +
    whitespace-collapse (the order matters — ``a.`` should normalize to
    `""`, not `"a"`)
  - yes/no/noanswer special-case (exact-match required, not token F1)
  - zero-overlap path (returns the ZERO_METRIC sentinel)
  - non-trivial F1 with hand-derived precision/recall

The module ships zero existing tests; if any of the regexes or the
Counter intersection break, no other CI signal would notice.
"""

from __future__ import annotations

import pytest

from slime.rollout.rm_hub.f1 import f1_score, normalize_answer


NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The Quick Brown Fox.", "quick brown fox"),  # articles stripped, punc removed, lowercased
        ("An apple, a day", "apple day"),  # both "an" and "a" stripped
        ("HELLO   WORLD", "hello world"),  # whitespace collapsed
        ("a.b,c!", "abc"),  # punc adjacent to chars collapses without leaving a space
        ("", ""),  # empty input survives
        ("the", ""),  # all-article input collapses to empty
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


@pytest.mark.unit
def test_f1_exact_match_is_perfect():
    """Hand-derived: tokens fully overlap → precision=recall=f1=1.0."""
    f1, p, r = f1_score("Paris is the capital", "Paris is the capital")
    assert (f1, p, r) == (1.0, 1.0, 1.0)


@pytest.mark.unit
def test_f1_partial_overlap_hand_derived():
    """Hand-derived: prediction "the brown fox" → ["brown", "fox"] after
    normalize; ground truth "a quick brown fox" → ["quick", "brown", "fox"].
    Intersection = {"brown", "fox"} → num_same = 2.
        precision = 2 / 2 = 1.0   (len(pred_tokens) = 2)
        recall    = 2 / 3
        f1        = 2 * 1.0 * (2/3) / (1.0 + 2/3) = 0.8
    """
    f1, p, r = f1_score("the brown fox", "a quick brown fox")
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(0.8)


@pytest.mark.unit
def test_f1_no_token_overlap_returns_zero_metric():
    """Disjoint vocabularies → ZERO_METRIC sentinel (0, 0, 0)."""
    assert f1_score("apple banana", "carrot date") == (0, 0, 0)


@pytest.mark.unit
def test_f1_none_prediction_returns_zero_metric():
    """A failed/missing prediction is a common rm path — must be zero, not raise."""
    assert f1_score(None, "anything") == (0, 0, 0)


@pytest.mark.unit
@pytest.mark.parametrize("special", ["yes", "no", "noanswer"])
def test_f1_special_token_pred_mismatch_returns_zero(special):
    """yes/no/noanswer in the prediction but not the ground truth — must be
    zero even if token-F1 would otherwise be non-zero. Pins the asymmetric
    early-exit at f1.py:33."""
    assert f1_score(special, "some other phrase") == (0, 0, 0)


@pytest.mark.unit
@pytest.mark.parametrize("special", ["yes", "no", "noanswer"])
def test_f1_special_token_gt_mismatch_returns_zero(special):
    """Mirror check: special tokens in the ground truth (f1.py:35)."""
    assert f1_score("some other phrase", special) == (0, 0, 0)


@pytest.mark.unit
def test_f1_special_token_exact_match_uses_token_path():
    """When prediction == ground_truth == "yes", the special-case early-exit
    does NOT fire (it has ``!=`` guards), so we land on the token-F1 path
    with a single common token → f1 = 1.0."""
    f1, p, r = f1_score("yes", "yes")
    assert (f1, p, r) == (1.0, 1.0, 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
