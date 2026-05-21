"""Reclassify trajectory_tree branches on existing rollout dumps with the
patched (classifier v3) ``_classify_branch`` + ``_new_turn`` from middleware.

Usage:
    python3 reclassify_existing_rollout.py <rollout_0.pt> [<rollout_0.pt> ...]

For each dump, walks every sample's trajectory_tree.turns, rebuilds _Turn
objects with synthetic full_ids of correct length, replays them through the
patched _new_turn / _classify_branch, and emits side-by-side OLD vs NEW
Counters. Saves a sibling ``*_reclassified_v3.pt`` per input.

We can't import middleware.py directly because it pulls in
``slime.utils.aiohttp_threaded``. Patch out that import the same way
smoke_classifier_v3.py does.

The C6 assertion lives in ``_pick_initial_prompt_len``: it ASSERTS that the
dump carries a non-zero ``initial_prompt_len`` (the C1 export). Legacy dumps
(no such key) fall back to ``turns[0].input_len`` with a single warning per
dump so degraded classification is still possible but visibly flagged.
"""
from __future__ import annotations

import os
import sys
import types
from collections import Counter
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Load middleware module with the aiohttp_threaded import stubbed.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO_ROOT)
_SRC_PATH = os.path.join(_HERE, "middleware.py")
with open(_SRC_PATH) as f:
    _SRC = f.read()
_SRC = _SRC.replace(
    "from slime.utils.aiohttp_threaded import AppHandle, run_app_in_thread",
    "AppHandle = run_app_in_thread = None  # reclassify-only stub",
)
_MOD = types.ModuleType("middleware_reclassify_v3")
sys.modules["middleware_reclassify_v3"] = _MOD
exec(_SRC, _MOD.__dict__)
_Turn = _MOD.__dict__["_Turn"]
_Session = _MOD.__dict__["_Session"]
_new_turn = _MOD.__dict__["_new_turn"]
_classify_branch = _MOD.__dict__["_classify_branch"]
_is_compact_resume_request = _MOD.__dict__["_is_compact_resume_request"]
_is_summarization_request = _MOD.__dict__["_is_summarization_request"]


# ---------------------------------------------------------------------------
# Per-sample reclassify
# ---------------------------------------------------------------------------
def _pick_initial_prompt_len(tree: dict[str, Any], dump_path: str, sample_idx: int) -> int:
    """[C6] Prefer the C1-serialized ``initial_prompt_len``; assert it's > 0.
    If the dump lacks the field (legacy pre-C1 dump) or it's 0, log a warning
    once and fall back to ``turns[0].input_len`` so legacy dumps still classify
    but the degradation is visible.
    """
    initial_prompt_len = tree.get("initial_prompt_len") or 0
    turns = tree.get("turns") or []
    try:
        assert initial_prompt_len > 0, (
            f"no init_pl in {dump_path}#sample{sample_idx} (legacy dump); "
            f"enable explicit fallback"
        )
    except AssertionError as e:
        # Soft-fall: legacy dump without C1 export. Print once per dump.
        warn_key = f"_warned_init_pl_{dump_path}"
        if not globals().get(warn_key):
            print(f"  [WARN] {e}; falling back to turns[0].input_len")
            globals()[warn_key] = True
        if turns:
            initial_prompt_len = turns[0].get("input_len", 0)
    return initial_prompt_len


def _replay_via_new_turn(turns: list[dict], initial_prompt_len: int) -> tuple[list[str], Counter]:
    """Re-run each turn through the patched _new_turn by reconstructing
    synthetic ideal_ids that match the original parent_prefix_len + new bytes.

    We use a 'token-id-as-turn-id' trick: each turn's full_ids is filled with
    its own id in the [pfx:pfx+output] slice so that downstream turns that
    share a prefix share token values. Specifically:

      turn.full_ids[ : parent_prefix_len] = parent.full_ids[ : parent_prefix_len]
      turn.full_ids[parent_prefix_len : input_len] = unique-to-turn (encodes turn.id)
      turn.full_ids[input_len : ] = turn.id-tagged output marker

    For replay, we need to construct ideal_ids (the input portion = input_len)
    that, when fed into the patched _new_turn, prefix-matches parent.full_ids
    at exactly parent_prefix_len. To get exactly that match, ideal_ids' first
    parent_prefix_len tokens MUST equal parent.full_ids[: parent_prefix_len];
    tokens after must NOT match any other turn's full_ids at that position.
    We achieve this by tagging the post-prefix bytes with a (turn_id, byte_idx)
    pair encoded as a large negative integer so no real prefix accidentally
    extends.

    Returns (new_branch_kinds, new_kind_counter).
    """
    s = _Session()
    s.initial_prompt_len = initial_prompt_len
    s.record_tree = True

    # Map turn id -> reconstructed _Turn (for parent lookup).
    new_kinds: list[str] = []
    new_counter: Counter = Counter()
    # Build full_ids progressively: each replayed turn writes its own full_ids
    # based on parent's full_ids + a fresh unique tail.
    rebuilt: dict[int, list[int]] = {}

    for t in turns:
        tid = t["id"]
        parent_id = t.get("parent_id")
        # parent_prefix_len from the dump tells us where the new turn's
        # ideal_ids should match parent.full_ids; we reconstruct ideal_ids
        # accordingly.
        pfx_recorded = t.get("parent_prefix_len") or 0
        input_len = t.get("input_len") or 0
        output_len = t.get("output_len") or 0
        body = t.get("request") or {}

        if parent_id is not None and parent_id in rebuilt:
            parent_full = rebuilt[parent_id]
            head = list(parent_full[:pfx_recorded])
        else:
            head = []
        # Tag the bytes after pfx with a unique encoding so no other turn's
        # full_ids accidentally extends them. Use 2-billion-base offset + tid.
        tag_base = -(10_000_000 + tid * 1_000_000)
        tail_in = [tag_base - i for i in range(max(0, input_len - len(head)))]
        ideal_ids = head + tail_in
        # Trim/pad to exactly input_len in case pfx_recorded > input_len.
        ideal_ids = ideal_ids[:input_len]
        if len(ideal_ids) < input_len:
            ideal_ids.extend([tag_base - 999_000 - i for i in range(input_len - len(ideal_ids))])

        new_turn = _new_turn(s, ideal_ids, body)
        # Fill in output tokens with a fresh unique tag so downstream turns
        # CAN prefix-match across input+output boundary when they're linear
        # continuations. We encode output bytes with another tag offset.
        out_tag_base = -(20_000_000 + tid * 1_000_000)
        new_turn.output_len = output_len
        new_turn.full_ids = ideal_ids + [out_tag_base - i for i in range(output_len)]
        rebuilt[tid] = new_turn.full_ids

        new_kinds.append(new_turn.branch_kind)
        new_counter[new_turn.branch_kind] += 1

    return new_kinds, new_counter


def _replay_classify_only(turns: list[dict], initial_prompt_len: int) -> tuple[list[str], Counter]:
    """A simpler replay that respects the dump's parent_id assignment and only
    re-runs _classify_branch (not _new_turn). Useful as a cross-check: if this
    diverges from the _new_turn-based replay then the [C5] pass-1/pass-2
    reparenting actually changed something on this dump.
    """
    # Build a turn_id -> _Turn map with synthetic full_ids of correct length.
    turn_map: dict[int, Any] = {}
    new_kinds: list[str] = []
    new_counter: Counter = Counter()
    for t in turns:
        tid = t["id"]
        parent_id = t.get("parent_id")
        parent_full_len = 0
        parent_obj = None
        if parent_id is not None and parent_id in turn_map:
            parent_obj = turn_map[parent_id]
            parent_full_len = len(parent_obj.full_ids)
        body = t.get("request") or {}
        pfx = t.get("parent_prefix_len") or 0
        new_kind = _classify_branch(
            parent_obj,
            pfx,
            parent_full_len,
            initial_prompt_len=initial_prompt_len,
            is_summarization=_is_summarization_request(body),
            is_compact_resume=_is_compact_resume_request(body),
        )
        new_kinds.append(new_kind)
        new_counter[new_kind] += 1
        turn_map[tid] = _Turn(
            id=tid,
            parent_id=parent_id,
            parent_prefix_len=pfx,
            input_len=t.get("input_len", 0),
            output_len=t.get("output_len", 0),
            full_ids=[0] * (t.get("input_len", 0) + t.get("output_len", 0)),
        )
    return new_kinds, new_counter


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def reclassify_dump(dump_path: str, *, save: bool = True) -> dict[str, Any]:
    print(f"\n=== {dump_path} ===")
    data = torch.load(dump_path, weights_only=False, map_location="cpu")
    samples = data["samples"]

    agg_old: Counter = Counter()
    agg_new_classify: Counter = Counter()
    agg_new_full: Counter = Counter()
    per_sample = []

    for si, s in enumerate(samples):
        meta = s.get("metadata") or {}
        tree = meta.get("trajectory_tree") or {}
        turns = tree.get("turns") or []
        if not turns:
            continue
        initial_prompt_len = _pick_initial_prompt_len(tree, dump_path, si)

        old_counter: Counter = Counter()
        for t in turns:
            old_counter[t.get("branch_kind", "?")] += 1

        # Replay using both strategies for cross-checking.
        _, new_counter_classify = _replay_classify_only(turns, initial_prompt_len)
        new_kinds_full, new_counter_full = _replay_via_new_turn(turns, initial_prompt_len)

        # The "definitive" v3 result writes the _new_turn-based replay since
        # it exercises both C3/C4/[classify] AND C5 [parent selection].
        for t, k in zip(turns, new_kinds_full):
            t["branch_kind"] = k

        agg_old.update(old_counter)
        agg_new_classify.update(new_counter_classify)
        agg_new_full.update(new_counter_full)
        per_sample.append((si, len(turns), dict(old_counter), dict(new_counter_full),
                           meta.get("instance_id") or "?"))

    print(f"  AGG OLD                     = {dict(agg_old)}")
    print(f"  AGG NEW (classify-only)     = {dict(agg_new_classify)}")
    print(f"  AGG NEW (full _new_turn v3) = {dict(agg_new_full)}")

    print("  per-sample (OLD -> NEW full):")
    for si, n, co, cn, inst in per_sample:
        flag = "  CHANGED" if co != cn else ""
        print(f"    s{si:>2} n={n:>3} OLD={co} NEW={cn} inst={inst}{flag}")

    if save:
        out_path = dump_path.replace(".pt", "_reclassified_v3.pt")
        torch.save(data, out_path)
        print(f"  saved reclassified dump to {out_path}")

    return {
        "dump_path": dump_path,
        "old": dict(agg_old),
        "new_classify_only": dict(agg_new_classify),
        "new_full": dict(agg_new_full),
        "per_sample": per_sample,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for p in sys.argv[1:]:
        reclassify_dump(p)


if __name__ == "__main__":
    main()
