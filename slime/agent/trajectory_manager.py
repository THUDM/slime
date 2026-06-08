"""Per-message trajectory tree manager (C-plan: token-faithful).

Design (Plan C, 2026-06-03; strict exact-prefix rewrite 2026-06-08;
per-message routing 2026-06-08):

* The tree is a router only. DFS merge keys on ``(role, node_match_key)``
  alone, one tree node per message — no prompt_ids prefix check and no
  same-role grouping. Same conversation prefix in ``messages`` space always
  lands on the same path, regardless of any chat_template re-tokenization
  drift across turns.

* Each assistant leaf stores the THIS-TURN sglang snapshot:
  ``turn_prompt_ids`` / ``turn_response_ids`` / ``turn_response_logprobs``
  / ``turn_finish_reason`` / ``turn_index``. Non-assistant nodes carry no
  token attribution at all.

* ``get_trajectory`` linearizes each leaf turn-by-turn. Walking the leaf's
  assistant chain root→leaf, the cumulative ``(prompt + response)`` tokens
  emitted so far are matched against the next turn's ``turn_prompt_ids``:

  - **Clean continuation** — the cumulative tokens are an exact prefix of the
    turn's prompt. The new prompt tail ``prompt[len(cumulative):]`` is appended
    as loss_mask=0, then the turn's ``response`` is appended as loss_mask=1 with
    real logprobs.

  - **Drift** — the same history re-tokenized differently across turns (TITO
    drift: tool_call arg order, whitespace, reasoning-block reordering). Rather
    than raise, ``get_trajectory`` tolerates the drift by where the divergence
    index ``L`` (the common-prefix length) falls, never letting logprobs
    misalign with tokens:

      * **case A** — ``L`` lands in a prompt region (outside every recorded
        response span): a genuine prompt-level re-render → **fork** (finalize
        the current coherent segment as its own Sample, restart a fresh segment
        at this turn). Fork discards nothing.
      * **case B1** — ``L`` lands inside the most-recent response span (the
        immediately-previous turn's response got re-rendered). Let
        ``d = len(cumulative) - L`` be the drifted tail length: ``d <
        fork_threshold`` → **replace** (truncate to ``L``, silently drop the
        drifted tail, realign to this turn's prompt); ``d >= fork_threshold`` →
        **fork**.
      * **case B2** — ``L`` lands inside an *earlier* turn's response span.
        Replacing would discard that turn's tail plus every later turn, so this
        always **forks** regardless of drift size.

  A fork splits one leaf into >=2 Samples; reward is split evenly across all
  emitted Samples (see ``get_trajectory``).

* ONE tolerated exception at the ROUTING (tree) layer: an assistant-rewrite
  merge. cc sometimes re-renders a previously-recorded assistant message when
  feeding it back as prompt (tool_call arg order, whitespace). The message no
  longer matches, so DFS forks at that assistant — leaving the original short
  turn as a standalone stub leaf -> its own Sample, diluting the trajectory's
  evenly-split reward. ``_try_merge_assistant_rewrite`` absorbs such a rewrite
  onto the existing leaf when its response is short enough
  (``fork_threshold_tokens``), demoting that node to routing-only so it
  contributes 0 training tokens. This is the MESSAGE-level dual of case B1's
  TOKEN-level replace: the rewrite-merge triggers when the message dict differs
  (DFS would fork), case B1 triggers when the message is identical but its
  tokens drift inside one chain. They live at different layers and handle
  different causes, so both are kept.

* Cross-leaf dedup at the LINEARIZATION layer: a snapshot assistant node can be
  shared by >=2 sibling leaves (it is the SAME Node object on each chain, since
  ``_find_mount_point`` reuses children). Linearizing every leaf from the root
  would otherwise train that shared prefix once per leaf. Instead the first leaf
  to reach a node (DFS / build order) trains its response (loss=1); later leaves
  re-emit it as loss=0 context (``trained=False`` in ``_Segment.extend``), so the
  shared prefix is trained exactly once. The tree is left intact (unlike the
  rewrite-merge's node demotion); only the per-leaf loss signal is masked. A
  leaf's terminal turn is its own freshly-created node, never pre-claimed, so
  every leaf keeps >=1 trained turn.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from slime.agent.adapters.common import TurnRecord
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ===========================================================================
# Node
# ===========================================================================


class Node:
    """One node in the trajectory tree.

    Routing fields (every node):
        role, messages, parent, children, metadata

    Per-turn snapshot fields (assistant leaves only — None on non-assistant
    and on internal assistant nodes that aren't a turn's own leaf):
        turn_prompt_ids:        list[int]    sglang prompt as fed to /generate
        turn_response_ids:      list[int]    sglang output ids
        turn_response_logprobs: list[float]
        turn_finish_reason:     str | None
        turn_index:             int          1-based, monotonic per session
    """

    def __init__(
        self,
        *,
        role: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: Node | None = None,
    ) -> None:
        self.role = role
        self.messages = list(messages or [])
        self.metadata = dict(metadata or {})
        self.parent: Node | None = parent
        self.children: list[Node] = []
        # messages is immutable after construction (only children are appended,
        # never the message list itself), so the routing key is computed once
        # here and reused on every descent instead of re-serializing per turn.
        self.match_key = node_match_key(self.messages)
        # per-turn snapshot
        self.turn_prompt_ids: list[int] | None = None
        self.turn_response_ids: list[int] | None = None
        self.turn_response_logprobs: list[float] | None = None
        self.turn_finish_reason: str | None = None
        self.turn_index: int | None = None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: Node) -> Node:
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list[Node]:
        """Ordered list of nodes from the first non-root ancestor down to self."""
        chain: list[Node] = []
        cur: Node | None = self
        while cur is not None and not cur.is_root:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator[Node]:
        if not self.children:
            yield self
            return
        for c in self.children:
            yield from c.leaves()


# ===========================================================================
# node_match_key + message helpers
# ===========================================================================


def node_match_key(messages: list[dict[str, Any]]) -> str:
    """Identity key for a node's message list.

    json.dumps(sort_keys=True) sorts dict-internal keys recursively; list
    element order is preserved (which is what we want: message order and
    tool_calls order are both semantically significant).
    """
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


def _lcp_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix between two int lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ===========================================================================
# Segment — one coherent linearized run (a fork boundary closes it)
# ===========================================================================


class _Segment:
    """Token accumulator for one coherent linearized run within a chain.

    Holds the running ``buffer`` with aligned ``loss`` / ``logprobs`` and per-turn
    response ``spans`` (``[start, end)`` half-open). Invariant: ``absorb_turn``
    always ends by appending a response, so ``spans[-1]`` is the most-recent
    response span and the buffer ends at its end. A fork closes the segment and a
    fresh one opens at the diverging turn (see module docstring case A/B1/B2).
    """

    def __init__(self) -> None:
        self.buffer: list[int] = []
        self.loss: list[int] = []
        self.logprobs: list[float] = []
        # Each span: (start, end, turn_index) over self.buffer, half-open.
        self.spans: list[tuple[int, int, int | None]] = []
        self.first_prompt_len: int = 0

    def measure_drift(self, prompt_ids: list[int]) -> int:
        """Length of the segment's token tail this turn's ``prompt_ids`` failed to reproduce.

        Normally 0 (clean continuation). A positive value IS token-id drift — the
        same history re-tokenized differently this turn (TITO drift) — not an error.
        """
        return len(self.buffer) - _lcp_len(self.buffer, prompt_ids)

    def can_absorb_drift(self, drift: int, fork_threshold: int) -> bool:
        """Whether a ``drift``-token re-tokenization can be realigned into this segment.

        Realignable only when the drift is confined to the most-recent response
        span (case B1) and shorter than ``fork_threshold``; otherwise the caller
        forks. See module docstring for case A/B1/B2.
        """
        if drift == 0:
            return True
        realign_at = len(self.buffer) - drift
        if not self.spans or realign_at < self.spans[-1][0]:
            return False  # case A (prompt region) or B2 (earlier response span)
        return fork_threshold > 0 and drift < fork_threshold

    def absorb_turn(
        self,
        drift: int,
        prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: list[float] | None,
        turn_index: int | None,
        *,
        trained: bool = True,
    ) -> None:
        """Append one turn, dropping any re-tokenization drift first so logprobs stay aligned.

        Drop the last ``drift`` tokens so the buffer re-anchors on the prefix this
        turn's ``prompt_ids`` reproduced (shrinking the prior span if cut), then
        append this turn's prompt tail (loss=0) and response (loss=1). A truncated
        prior span stays loss=1: that region is both the prior turn's response and
        this turn's prompt context.

        ``trained=False`` appends the response as loss=0 / logprob=0.0 instead — the
        node is already owned by an earlier sibling leaf, so re-training it would
        double-count the shared prefix (see ``_chain_to_sample`` claim-on-first-visit).
        """
        realign_at = len(self.buffer) - drift
        del self.buffer[realign_at:]
        del self.loss[realign_at:]
        del self.logprobs[realign_at:]
        if self.spans and realign_at < self.spans[-1][1]:
            s, _e, j = self.spans[-1]
            self.spans[-1] = (s, realign_at, j)  # may collapse to empty (s == realign_at); harmless

        is_first_turn = not self.spans
        tail = prompt_ids[realign_at:]
        self.buffer.extend(tail)
        self.loss.extend([0] * len(tail))
        self.logprobs.extend([0.0] * len(tail))

        start = len(self.buffer)
        self.buffer.extend(response_ids)
        self.loss.extend([1 if trained else 0] * len(response_ids))
        self.logprobs.extend(
            response_logprobs if (trained and response_logprobs is not None) else [0.0] * len(response_ids)
        )
        self.spans.append((start, len(self.buffer), turn_index))

        if is_first_turn:
            self.first_prompt_len = len(prompt_ids)  # stripped at build time

    def response_strip(self) -> int:
        """Start index of the response region (the leading first-turn prompt prefix)."""
        return min(self.first_prompt_len, len(self.loss))

    def has_trained_response(self) -> bool:
        """Whether the response region carries any loss=1 token.

        False only when every turn in the segment was claimed by an earlier
        sibling leaf (cross-leaf dedup) -> no training signal to emit.
        """
        return any(self.loss[self.response_strip() :])


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    """Per-sid trajectory tree manager.

    See module docstring for the C-plan invariants. Each ``append_turn``
    mounts >=0 prompt nodes (one per message, under the deepest matching
    ancestor) + exactly 1 assistant leaf carrying that turn's sglang snapshot.
    """

    def __init__(self, *, fork_threshold_tokens: int | None = None) -> None:
        # Drift fork/replace threshold for case-B1 (see module docstring case
        # A/B1/B2). <=0 forces every B1 to fork (max fidelity); ``None`` means
        # "use the default".
        self._fork_threshold: int = 1024 if fork_threshold_tokens is None else fork_threshold_tokens
        self._trees: dict[str, Node] = {}
        self._turn_count: dict[str, int] = {}

    # -------------------- public ------------------------------------------

    def has_session(self, sid: str) -> bool:
        return sid in self._trees

    def turn_count(self, sid: str) -> int:
        return self._turn_count.get(sid, 0)

    def append_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("append_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        if turn.output_log_probs and len(turn.output_log_probs) != len(turn.output_ids):
            raise ValueError(
                f"turn.output_log_probs length {len(turn.output_log_probs)} != "
                f"turn.output_ids length {len(turn.output_ids)}"
            )

        root = self._trees.setdefault(sid, Node())

        cur, i = self._find_mount_point(root, prompt_messages)
        cur, i = self._try_merge_assistant_rewrite(sid, cur, prompt_messages, i)
        cur = self._mount_prompt_messages(cur, prompt_messages[i:], tools)
        self._attach_assistant_leaf(sid, cur, turn=turn, response_message=response_message, metadata=metadata)

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample=None,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list:
        """Drain a sid into slime ``Sample`` objects, then drop the session.

        ``get_trajectory`` is the lifecycle boundary where the message routing
        tree is linearized into token-normalized ``Sample`` objects. Each
        routing leaf yields exactly one Sample. ``reward`` is split evenly
        across all emitted samples. The sid is consumed: a second call for the
        same sid returns ``[]``.
        """
        if base_sample is None:
            base_sample = Sample(index=0, prompt="")

        root = self._trees.get(sid)
        if root is None:
            return []

        samples: list[Sample] = []
        # Cross-leaf dedup (see module docstring): a snapshot node shared by sibling
        # leaves is trained only by the first leaf to reach it. ``claimed`` carries
        # node identity (id()) across leaves to enforce that.
        claimed: set[int] = set()
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(
                self._chain_to_sample(chain, base_sample=base_sample, extra_metadata=extra_metadata, claimed=claimed)
            )

        # Reward is split evenly across every emitted sample (one per leaf); the
        # token-weighted reducer downstream then gives each loss token the
        # trajectory's full R. Assigned after the fact so the per-leaf builder
        # stays reward-agnostic.
        per_sample_reward = (reward / len(samples)) if samples else 0.0
        for s in samples:
            s.reward = per_sample_reward

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples

    # -------------------- internals ----------------------------------------

    def _find_mount_point(self, root: Node, messages: list[dict[str, Any]]) -> tuple[Node, int]:
        """DFS down the existing tree by ``(role, node_match_key)``, per message.

        Returns ``(cur, i)``: ``cur`` is the deepest node whose path matches
        ``messages[:i]`` exactly; ``i`` is the index into ``messages`` of the
        first message that diverges from anything mounted so far (i.e., where
        this turn's new content begins).
        """
        cur = root
        i = 0
        while i < len(messages):
            m = messages[i]
            m_key = node_match_key([m])
            match = next(
                (c for c in cur.children if c.role == m.get("role") and c.match_key == m_key),
                None,
            )
            if match is None:
                break
            cur = match
            i += 1
        return cur, i

    def _try_merge_assistant_rewrite(
        self,
        sid: str,
        cur: Node,
        prompt_messages: list[dict[str, Any]],
        i: int,
    ) -> tuple[Node, int]:
        """Absorb a short assistant-rewrite onto its existing node instead of forking.

        See module docstring (rewrite-merge bullet) for the why. Purely a
        reward-hygiene / de-fragmentation optimization — forking is already safe
        (the rewrite mounts as a routing-only node, skipped at linearization).

        When the diverging message is an assistant and exactly one eligible
        *short-response leaf* sibling exists, adopt the rewritten message onto that
        node and DEMOTE it to routing-only (clear its turn snapshot). Any other
        mismatch (non-assistant, long response, non-leaf or ambiguous) forks as usual.
        """
        if self._fork_threshold <= 0:
            return cur, i  # feature off
        if i >= len(prompt_messages) or prompt_messages[i].get("role") != "assistant":
            return cur, i  # genuine non-assistant history fork -> leave it

        candidates = [
            c
            for c in cur.children
            if c.role == "assistant"
            # Leaf == rewrite of the immediately-previous assistant: no later
            # turn has extended it yet. A non-leaf assistant has already grown a
            # subchain; merging onto it would tangle that history.
            and not c.children
            # A real turn leaf carrying this turn's snapshot, not an already-
            # demoted routing node (turn_prompt_ids cleared by a prior merge).
            and c.turn_prompt_ids is not None and len(c.turn_response_ids or []) < self._fork_threshold
        ]
        if len(candidates) != 1:
            if len(candidates) >= 2:
                # Ambiguous: don't pick arbitrarily — fork as usual and hint.
                logger.warning(
                    "append_turn(sid=%s turn=%s): %d eligible rewrite-merge "
                    "candidates; forking instead (ambiguous mixed state).",
                    sid,
                    self._turn_count.get(sid, 0) + 1,
                    len(candidates),
                )
            return cur, i

        sib = candidates[0]
        sib.metadata["merged_rewrite"] = {  # observability breadcrumb only
            "abandoned_turn_index": sib.turn_index,
            "abandoned_response_tokens": len(sib.turn_response_ids or []),
        }
        # Demote to routing-only: snapshot cleared -> skipped by the
        # ``turn_prompt_ids is not None`` filter at linearization, and never
        # re-selected as a merge candidate on a later turn.
        sib.turn_prompt_ids = None
        sib.turn_response_ids = None
        sib.turn_response_logprobs = None
        sib.turn_finish_reason = None
        sib.turn_index = None
        # Adopt the rewritten message; the match_key cache MUST follow messages
        # so a later turn's DFS descends through this (now rewritten) node
        # instead of forking again.
        sib.messages = [prompt_messages[i]]
        sib.match_key = node_match_key(sib.messages)
        return sib, i + 1

    def _mount_prompt_messages(
        self,
        cur: Node,
        remaining_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Node:
        """Attach each remaining prompt message as a routing node under ``cur``.

        One node per message; token attribution happens at get_trajectory time, not
        here. The tools metadata is placed only on the FIRST system node on the path
        (``_first_system_already_set`` walks ``cur → root``; ``cur`` is the deepest
        mounted node, so the walk sees every already-mounted ancestor).
        """
        for m in remaining_messages:
            role = m.get("role")
            md: dict[str, Any] = {}
            if role == "system" and tools is not None and not self._first_system_already_set(cur):
                md["tools"] = list(tools)
            cur = cur.add_child(Node(role=role, messages=[m], metadata=md))
        return cur

    def _attach_assistant_leaf(
        self,
        sid: str,
        cur: Node,
        *,
        turn: TurnRecord,
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Attach this turn's assistant leaf carrying the sglang snapshot."""
        asst = Node(
            role="assistant",
            messages=[response_message] if response_message is not None else [],
            metadata=dict(metadata or {}),
        )
        asst.turn_prompt_ids = list(turn.prompt_ids)
        asst.turn_response_ids = list(turn.output_ids)
        asst.turn_response_logprobs = list(turn.output_log_probs) if turn.output_log_probs else None
        asst.turn_finish_reason = turn.finish_reason
        asst.turn_index = self._turn_count.get(sid, 0) + 1
        cur.add_child(asst)
        self._turn_count[sid] = asst.turn_index

    def _chain_to_sample(
        self,
        chain: list[Node],
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        claimed: set[int],
    ) -> list[Sample]:
        """Linearize one root→leaf chain into >=1 Samples (see module docstring).

        Each turn either grows the current segment or forks a new one (case
        A/B1-too-long/B2). ``claimed`` deduplicates snapshot nodes shared across
        sibling leaves; a leaf's terminal node is never pre-claimed, so every leaf
        keeps >=1 trained turn. Reward is left at 0.0 and assigned by the caller.
        """
        # Only assistant leaves carrying this turn's sglang snapshot participate.
        # Routing assistant nodes mounted from prior-turn replay (turn_prompt_ids
        # is None) carry no token signal and are skipped.
        asst_chain = [n for n in chain if n.role == "assistant" and n.turn_prompt_ids is not None]

        segments: list[_Segment] = []
        seg = _Segment()
        for asst in asst_chain:
            prompt_ids = asst.turn_prompt_ids or []
            drift = seg.measure_drift(prompt_ids)
            if not seg.can_absorb_drift(drift, self._fork_threshold):
                segments.append(seg)  # fork: close this segment, start fresh here
                seg = _Segment()
                drift = 0
            trained = id(asst) not in claimed
            claimed.add(id(asst))
            seg.absorb_turn(
                drift,
                prompt_ids,
                asst.turn_response_ids or [],
                asst.turn_response_logprobs,
                asst.turn_index,
                trained=trained,
            )
        segments.append(seg)

        # Drop empty / fully-masked segments: an in-chain fork can isolate a run of
        # turns all claimed by an earlier sibling leaf (no loss=1 token), which would
        # trip the downstream "not fully masked" assert. A leaf's terminal turn is
        # never pre-claimed, so its final segment always survives.
        return [
            self._build_leaf_sample(
                base_sample=base_sample,
                extra_metadata=extra_metadata,
                tokens=seg.buffer,
                loss_mask=seg.loss,
                logprobs=seg.logprobs,
                strip=seg.response_strip(),
            )
            for seg in segments
            if seg.buffer and seg.has_trained_response()
        ]

    def _build_leaf_sample(
        self,
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        tokens: list[int],
        loss_mask: list[int],
        logprobs: list[float],
        strip: int,
    ) -> Sample:
        """Build one Sample from a linearized token segment.

        ``loss_mask`` / ``logprobs`` are clamped to the response region (``strip``
        drops the leading first-turn prompt prefix) per the slime contract:
        ``response_length == len(loss_mask)``, covering only the response region
        (see backends/megatron_utils/data.py:139, ray/rollout.py:695). ``reward``
        is left at 0.0; the caller assigns the per-sample share.

        Per-row dataset metadata and the per-turn tool / finish_reason snapshot are
        intentionally NOT propagated here (dump/analysis tooling reads them off the
        tree nodes); only ``extra_metadata`` rides along.
        """
        loss_resp, lp_resp = loss_mask[strip:], logprobs[strip:]
        metadata = dict(extra_metadata or {})
        return Sample(
            index=base_sample.index,
            group_id=base_sample.group_id if base_sample.group_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            tokens=list(tokens),
            response_length=len(loss_resp),
            loss_mask=loss_resp,
            rollout_log_probs=lp_resp,
            reward=0.0,
            status=Sample.Status.COMPLETED,
            metadata=metadata,
        )

    @staticmethod
    def _first_system_already_set(start: Node) -> bool:
        """Walk start->root looking for a system node already carrying tools."""
        cur: Node | None = start
        while cur is not None and not cur.is_root:
            if cur.role == "system" and cur.metadata.get("tools") is not None:
                return True
            cur = cur.parent
        return False


__all__ = [
    "Node",
    "TrajectoryManager",
    "node_match_key",
    "_lcp_len",
]
