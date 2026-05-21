"""Smoke for _segment_reward in examples/coding_agent_rl/generate.py.

Validates the 3 reward modes against synthetic segment lists. No middleware,
no sandbox — pure function under test.

Run::

    python3 examples/coding_agent_rl/smoke_reward_modes.py
"""

from __future__ import annotations


# Inline copy of the function under test to avoid pulling in heavy imports
# (slime.utils, middleware, etc.). The smoke is a behavioral spec on the
# reward-allocation logic — kept in lock-step with generate.py:_segment_reward
# by both being one-liners. If they diverge, update both together.
def _segment_reward(reward: float, num_segments: int, seg_kind: str | None,
                    mode: str) -> float:
    if mode == "uniform":
        return float(reward) / max(1, num_segments)
    if mode == "final_only":
        return float(reward) if seg_kind == "final" else 0.0
    return float(reward)  # copy (back-compat)


def main() -> int:
    reward = 1.0
    # Three synthetic segments: [pre_wipe, pre_wipe, final]
    segments = [
        {"kind": "pre_wipe"},
        {"kind": "pre_wipe"},
        {"kind": "final"},
    ]
    n = len(segments)

    # --- uniform: reward / N each -----------------------------------------
    expected_uniform = [reward / n] * n
    actual_uniform = [_segment_reward(reward, n, s["kind"], "uniform") for s in segments]
    assert actual_uniform == expected_uniform, (actual_uniform, expected_uniform)
    print(f"[1] PASS: uniform reward={reward} N={n} -> {actual_uniform}")

    # --- copy: reward each ------------------------------------------------
    expected_copy = [reward] * n
    actual_copy = [_segment_reward(reward, n, s["kind"], "copy") for s in segments]
    assert actual_copy == expected_copy, (actual_copy, expected_copy)
    print(f"[2] PASS: copy reward={reward} N={n} -> {actual_copy}")

    # --- final_only: reward only on kind=='final' -------------------------
    expected_final_only = [0.0, 0.0, reward]
    actual_final_only = [_segment_reward(reward, n, s["kind"], "final_only") for s in segments]
    assert actual_final_only == expected_final_only, (actual_final_only, expected_final_only)
    print(f"[3] PASS: final_only reward={reward} N={n} -> {actual_final_only}")

    # --- edge: N=1 single segment under uniform should equal reward -------
    one_seg = [{"kind": "final"}]
    actual_one_uniform = _segment_reward(reward, len(one_seg), one_seg[0]["kind"], "uniform")
    assert actual_one_uniform == reward, actual_one_uniform
    print(f"[4] PASS: uniform with N=1 -> {actual_one_uniform}")

    # --- edge: N=0 defensive (max(1, 0)) ----------------------------------
    actual_zero_uniform = _segment_reward(reward, 0, None, "uniform")
    assert actual_zero_uniform == reward, actual_zero_uniform
    print(f"[5] PASS: uniform with N=0 -> {actual_zero_uniform}")

    # --- edge: final_only with no final segment -> all 0 ------------------
    no_final = [{"kind": "pre_wipe"}, {"kind": "subagent"}]
    actual_no_final = [_segment_reward(reward, len(no_final), s["kind"], "final_only") for s in no_final]
    assert actual_no_final == [0.0, 0.0], actual_no_final
    print(f"[6] PASS: final_only with no final segment -> {actual_no_final}")

    print("all reward-mode smoke cases PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
