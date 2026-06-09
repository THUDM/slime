"""Build a per-session training trajectory from multi-turn conversation data.

The :class:`TrajectoryManager` builds one trajectory per session. ``append_turn``
feeds in each turn (prompt messages + the served model's sglang snapshot),
routing it into a per-sid message tree; ``get_trajectory`` then linearizes that
tree into a ``list[Sample]`` of loss-masked training rows, tolerating TITO
re-tokenization drift via fork/replace.
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
    # sort_keys sorts dict-internal keys but PRESERVES list order -- message
    # order and tool_calls order are both semantically significant.
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


def _lcp_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ===========================================================================
# Segment — one coherent linearized run (a fork boundary closes it)
# ===========================================================================


class _Segment:
    """Token accumulator for one coherent linearized run within a chain."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        # Each span: (start, end, turn_index) over self.tokens, half-open.
        self.spans: list[tuple[int, int, int | None]] = []
        self.first_prompt_len: int = 0

    def measure_token_drift(self, prompt_ids: list[int]) -> int:
        return len(self.tokens) - _lcp_len(self.tokens, prompt_ids)

    def can_absorb_drift(self, drift: int, fork_threshold: int) -> bool:
        if drift == 0:
            return True
        realign_at = len(self.tokens) - drift
        if not self.spans or realign_at < self.spans[-1][0]:
            return False
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
        """Append one turn, dropping any re-tokenization drift first so logprobs stay aligned."""
        realign_at = len(self.tokens) - drift
        del self.tokens[realign_at:]
        del self.loss_mask[realign_at:]
        del self.logprobs[realign_at:]
        if self.spans and realign_at < self.spans[-1][1]:
            s, _e, j = self.spans[-1]
            # NOTE: the truncated prior span stays loss=1 -- that region is both
            # the prior turn's response and this turn's prompt context.
            self.spans[-1] = (s, realign_at, j)  # may collapse to empty (s == realign_at); harmless

        is_first_turn = not self.spans
        tail = prompt_ids[realign_at:]
        self.tokens.extend(tail)
        self.loss_mask.extend([0] * len(tail))
        self.logprobs.extend([0.0] * len(tail))

        start = len(self.tokens)
        self.tokens.extend(response_ids)
        self.loss_mask.extend([1 if trained else 0] * len(response_ids))
        self.logprobs.extend(
            response_logprobs if (trained and response_logprobs is not None) else [0.0] * len(response_ids)
        )
        self.spans.append((start, len(self.tokens), turn_index))

        if is_first_turn:
            self.first_prompt_len = len(prompt_ids)  # stripped at build time

    def response_strip(self) -> int:
        return min(self.first_prompt_len, len(self.loss_mask))

    def has_trained_response(self) -> bool:
        return any(self.loss_mask[self.response_strip() :])


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    def __init__(self, *, fork_threshold_tokens: int | None = None) -> None:
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
        """Linearize this sid's routing tree into slime ``Sample`` objects and
        consume the session.

        Each routing leaf yields one or more Samples; ``reward`` is split evenly
        across all of them. The sid is dropped afterwards, so a second call for
        the same sid returns ``[]``.
        """
        assert base_sample is not None, "get_trajectory requires a base_sample"

        root = self._trees.get(sid)
        if root is None:
            return []

        samples: list[Sample] = []
        claimed: set[int] = set()
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(
                self._chain_to_sample(chain, base_sample=base_sample, extra_metadata=extra_metadata, claimed=claimed)
            )

        # TODO custom reward func
        per_sample_reward = (reward / len(samples)) if samples else 0.0
        for s in samples:
            s.reward = per_sample_reward

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples

    # -------------------- internals ----------------------------------------

    def _find_mount_point(self, root: Node, messages: list[dict[str, Any]]) -> tuple[Node, int]:
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
        """Absorb a short assistant-rewrite onto its node instead of forking -- reward
        hygiene only; forking is already safe (a rewrite mounts as routing-only)."""
        if self._fork_threshold <= 0:
            return cur, i  # feature off
        if i >= len(prompt_messages) or prompt_messages[i].get("role") != "assistant":
            return cur, i  # genuine non-assistant history fork -> leave it

        candidates = [
            c
            for c in cur.children
            if c.role == "assistant"
            and not c.children
            and c.turn_prompt_ids is not None
            and len(c.turn_response_ids or []) < self._fork_threshold
        ]
        if len(candidates) != 1:
            if len(candidates) >= 2:
                logger.warning(
                    "append_turn(sid=%s turn=%s): %d eligible rewrite-merge "
                    "candidates; forking instead (ambiguous mixed state).",
                    sid,
                    self._turn_count.get(sid, 0) + 1,
                    len(candidates),
                )
            return cur, i

        sib = candidates[0]
        sib.metadata["merged_rewrite"] = {
            "abandoned_turn_index": sib.turn_index,
            "abandoned_response_tokens": len(sib.turn_response_ids or []),
        }

        sib.turn_prompt_ids = None
        sib.turn_response_ids = None
        sib.turn_response_logprobs = None
        sib.turn_finish_reason = None
        sib.turn_index = None
        sib.messages = [prompt_messages[i]]
        sib.match_key = node_match_key(sib.messages)
        return sib, i + 1

    def _mount_prompt_messages(
        self,
        cur: Node,
        remaining_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Node:
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
        asst_chain = [n for n in chain if n.role == "assistant" and n.turn_prompt_ids is not None]

        segments: list[_Segment] = []
        seg = _Segment()
        for asst in asst_chain:
            prompt_ids = asst.turn_prompt_ids or []
            drift = seg.measure_token_drift(prompt_ids)
            if not seg.can_absorb_drift(drift, self._fork_threshold):
                segments.append(seg)
                seg = _Segment()
                drift = 0
            trained = id(asst) not in claimed
            claimed.add(id(asst))
            # A snapshot node can be shared by sibling leaves (same Node object).
            # Train it only on the first leaf to reach it; later leaves re-emit it
            # as loss=0 context so the shared prefix isn't double-counted.
            seg.absorb_turn(
                drift,
                prompt_ids,
                asst.turn_response_ids or [],
                asst.turn_response_logprobs,
                asst.turn_index,
                trained=trained,
            )
        segments.append(seg)

        # Drop fully-masked segments (all turns claimed by an earlier leaf) -- they'd
        # trip the downstream "not fully masked" assert. A leaf's terminal turn is
        # never pre-claimed, so its final segment always survives.
        return [
            self._build_leaf_sample(
                base_sample=base_sample,
                extra_metadata=extra_metadata,
                tokens=seg.tokens,
                loss_mask=seg.loss_mask,
                logprobs=seg.logprobs,
                strip=seg.response_strip(),
            )
            for seg in segments
            if seg.tokens and seg.has_trained_response()
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
        loss_resp, lp_resp = loss_mask[strip:], logprobs[strip:]
        metadata = dict(extra_metadata or {})
        return Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
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
