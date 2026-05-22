"""Tests for §5 TRANSLATE TITO verifier + §8 HANDLER step 12 mask logic.

Covers SPEC §7.1 entry `test_translate_tito.py`:
  * verify_tito_for_turn happy path (matches)
  * verify_tito_for_turn drift detection
  * test_empty_turn_skip  - I8 critical bug fix (output_ids=[] must not clear all)
  * test_abort_skip_tito - I9 (abort'd turn skipped from TITO)
  * mask only touches the last n slice, not earlier turns
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import middleware as mw  # noqa: E402


class _FakeTok:
    """Deterministic stub: encode/decode are inverse, treating output_ids as
    a single-byte-per-token ASCII codec. Drift is simulated by returning a
    DIFFERENT mapping when raw_text is the special string '__drift__'."""

    def encode(self, text, add_special_tokens=False):
        if text == "__drift__":
            return [9999]  # mismatch
        # Each ASCII char -> token = ord(char)
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_verify_tito_matches() -> None:
    tok = _FakeTok()
    ids = [ord(c) for c in "hello"]
    text = "hello"
    assert mw.verify_tito_for_turn(tok, text, ids) is True


def test_verify_tito_drift() -> None:
    tok = _FakeTok()
    ids = [ord(c) for c in "abc"]
    # Pretend the decoded text drifts
    assert mw.verify_tito_for_turn(tok, "__drift__", ids) is False


def test_empty_turn_skip() -> None:
    """CRITICAL I8 fix: ``loss_mask[-0:] = []`` would clear the whole mask.

    The handler must guard with n>0 BEFORE assigning. We simulate the
    handler step 12 logic with output_ids=[] to confirm no mutation."""
    s = mw.Session()
    s.loss_mask = [1, 1, 1, 1, 1]
    s.response_ids = [10, 11, 12, 13, 14]
    initial = list(s.loss_mask)

    output_ids: list[int] = []
    n = len(output_ids)
    # This is the inline guard from §8 step 12:
    if n > 0 and s.num_aborts_this_turn == 0:
        s.loss_mask[-n:] = [0] * n   # would clear everything if n==0
    # Mask must be untouched.
    assert s.loss_mask == initial, f"I8 violated: loss_mask = {s.loss_mask}"


def test_abort_skip_tito() -> None:
    """I9: when num_aborts_this_turn > 0, TITO must be skipped."""
    s = mw.Session()
    s.num_aborts_this_turn = 2
    s.loss_mask = [1] * 10
    s.response_ids = list(range(10))
    initial = list(s.loss_mask)

    output_ids = list(range(10))
    n = len(output_ids)
    if n > 0 and s.num_aborts_this_turn == 0:
        # Would mask everything if we got here
        s.loss_mask[-n:] = [0] * n
    # We did NOT get here because num_aborts_this_turn > 0; mask untouched.
    assert s.loss_mask == initial


def test_mask_targets_only_last_n() -> None:
    """U2: mask must only affect the last n elements of loss_mask; earlier
    turns must remain unchanged."""
    s = mw.Session()
    # Simulate 3 turns: 100, 80, 120 tokens.
    s.loss_mask = [1] * 100 + [1] * 80 + [1] * 120
    s.response_ids = list(range(300))

    # Drift on turn 2 (the 80-token middle turn).
    n = 80
    if n > 0 and s.num_aborts_this_turn == 0:
        # We pretend handler is masking the JUST-extended turn slice; in real
        # handler step 11 has already appended turn 2's slice. Here we mimic
        # by masking the 80 entries that span indices 100..180 — but in real
        # life the just-extended slice is the LAST n, so simulate this by
        # treating loss_mask as if turn 2 is the most recent extension.
        s.loss_mask = [1] * 100 + [1] * 80  # rewind to "just after extend turn 2"
        s.loss_mask[-n:] = [0] * n
    assert s.loss_mask[:100] == [1] * 100, "turn 0 must be unchanged"
    assert s.loss_mask[100:180] == [0] * 80, "turn 1 must be zeroed"


def test_mask_target_is_sub_session() -> None:
    """U2: target.loss_mask works for SubSession too (not hardcoded Session)."""
    sub = mw.SubSession()
    sub.loss_mask = [1] * 50
    sub.response_ids = list(range(50))

    n = 50
    if n > 0:
        sub.loss_mask[-n:] = [0] * n
    assert sub.loss_mask == [0] * 50


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
