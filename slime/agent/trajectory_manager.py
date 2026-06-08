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

* ``get_trajectory`` linearizes each leaf turn-by-turn with a STRICT
  exact-prefix contract: walking the leaf's assistant chain root→leaf, the
  cumulative ``(prompt + response)`` tokens emitted so far MUST be an exact
  prefix of the next turn's ``turn_prompt_ids``. When it is, the new prompt
  tail ``prompt[len(cumulative):]`` is appended as loss_mask=0, then the
  turn's ``response`` is appended as loss_mask=1 with real logprobs.

* When the prefix does NOT match, the upstream tokenization drifted (the
  same history re-tokenized differently across turns). That is a bug to
  surface, not to paper over: ``get_trajectory`` raises ``ValueError`` with
  the sid, turn_index, common-prefix length, and drift size so the
  offending turn is locatable. Note a drift introduced at an early turn can
  surface several turns later — the prefix check catches it whenever the
  re-rendered early region first diverges from the accumulated tokens.

  (History: an earlier design tolerated drift via LCP drop-and-replace plus
  optional drift-fork / drop-accounting / fork-merge. That machinery is
  removed here in favor of failing loudly; it can be re-added as an explicit
  layer later if real drift turns out to be unavoidable.)
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

    __slots__ = (
        # routing
        "role",
        "messages",
        "metadata",
        "parent",
        "children",
        "match_key",
        # per-turn snapshot (assistant leaves)
        "turn_prompt_ids",
        "turn_response_ids",
        "turn_response_logprobs",
        "turn_finish_reason",
        "turn_index",
    )

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
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    """Per-sid trajectory tree manager.

    See module docstring for the C-plan invariants. Each ``append_turn``
    mounts >=0 prompt nodes (one per message, under the deepest matching
    ancestor) + exactly 1 assistant leaf carrying that turn's sglang snapshot.
    """

    def __init__(self) -> None:
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
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(self._chain_to_sample(sid, chain, base_sample=base_sample, extra_metadata=extra_metadata))

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

    def _mount_prompt_messages(
        self,
        cur: Node,
        remaining_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Node:
        """Attach each remaining prompt message as a routing node under ``cur``.

        One node per message (``node.messages`` is a singleton list). Token
        attribution happens at get_trajectory time, not here. The tools
        metadata is placed only on the FIRST system node on the path —
        ``_first_system_already_set(cur)`` walks ``cur → root`` looking for a
        system ancestor that already carries it, and ``cur`` here is the
        deepest node from descent (+ optional merge), so the walk sees every
        ancestor that's already mounted.
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
        sid: str,
        chain: list[Node],
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
    ) -> list[Sample]:
        """Linearize one root→leaf chain into a single Sample (strict exact-prefix).

        Walk the chain's assistant nodes root→leaf, accumulating tokens. Each
        turn's cumulative ``(prompt + response)`` so far MUST be an exact prefix
        of the next turn's ``turn_prompt_ids``; otherwise the upstream
        tokenization drifted and we raise (see module docstring). Reward is left
        at 0.0 here and assigned by the caller.
        """
        # Only assistant leaves carrying this turn's sglang snapshot participate.
        # Routing assistant nodes mounted from prior-turn replay (turn_prompt_ids
        # is None) carry no token signal and are skipped.
        asst_chain = [n for n in chain if n.role == "assistant" and n.turn_prompt_ids is not None]

        tokens: list[int] = []
        loss_mask: list[int] = []
        logprobs: list[float] = []
        for asst in asst_chain:
            prompt = asst.turn_prompt_ids or []
            response = asst.turn_response_ids or []
            response_logprobs = asst.turn_response_logprobs

            # Strict prefix check: tokens accumulated so far must be an exact
            # prefix of this turn's prompt. For the first turn `tokens` is empty
            # so this trivially holds.
            n = len(tokens)
            if prompt[:n] != tokens:
                self._raise_prefix_drift(sid, asst_chain, asst, tokens, prompt)

            new_prompt = prompt[n:]
            tokens.extend(new_prompt)
            loss_mask.extend([0] * len(new_prompt))
            logprobs.extend([0.0] * len(new_prompt))

            tokens.extend(response)
            loss_mask.extend([1] * len(response))
            logprobs.extend(response_logprobs if response_logprobs is not None else [0.0] * len(response))

        first_prompt_len = len(asst_chain[0].turn_prompt_ids or []) if asst_chain else 0
        return [
            self._build_leaf_sample(
                base_sample=base_sample,
                extra_metadata=extra_metadata,
                tokens=tokens,
                loss_mask=loss_mask,
                logprobs=logprobs,
                first_prompt_len=first_prompt_len,
            )
        ]

    @staticmethod
    def _raise_prefix_drift(
        sid: str,
        asst_chain: list[Node],
        asst: Node,
        tokens: list[int],
        prompt: list[int],
    ) -> None:
        """Raise on a TITO drift: accumulated tokens are not a prefix of prompt.

        Reports the common-prefix length and drift size, plus which earlier
        assistant turn's prompt region the divergence falls in — a drift
        introduced at an early turn can surface only when a later turn
        re-renders that early region differently.
        """
        L = _lcp_len(tokens, prompt)
        drift = len(tokens) - L
        # Locate which turn's prompt region L lands in, so an early-turn drift
        # that surfaces several turns later is still attributable.
        drift_in_turn = None
        for prior in asst_chain:
            if prior is asst:
                break
            if L < len(prior.turn_prompt_ids or []):
                drift_in_turn = prior.turn_index
                break
        raise ValueError(
            f"get_trajectory(sid={sid} turn={asst.turn_index}): TITO drift — "
            f"accumulated tokens are not a prefix of this turn's prompt "
            f"(common_prefix_len={L}, drift={drift} tokens; divergence falls in "
            f"turn {drift_in_turn}'s prompt region). The same history "
            f"re-tokenized differently across turns; refusing to silently "
            f"drop/realign."
        )

    def _build_leaf_sample(
        self,
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        tokens: list[int],
        loss_mask: list[int],
        logprobs: list[float],
        first_prompt_len: int,
    ) -> Sample:
        """Build one Sample from a linearized token segment.

        ``loss_mask`` / ``logprobs`` are clamped to the response region (the
        leading first-turn prompt prefix is stripped) per the slime contract:
        ``response_length == len(loss_mask)`` and loss_mask/logprobs cover only
        the response region. ``reward`` is left at 0.0; the caller assigns the
        per-sample share.

        Sample metadata carries only ``extra_metadata`` (empty on the
        production path): the per-row dataset metadata and per-turn tool /
        finish_reason snapshot are intentionally NOT propagated onto the leaf
        Sample. Dump/analysis tooling reads those off the tree nodes instead.
        """
        # Clamp loss_mask/logprobs to the response region: strip the leading
        # first-turn prompt prefix so response_length == len(loss_mask), per the
        # slime contract (see backends/megatron_utils/data.py:139,
        # ray/rollout.py:695).
        strip = min(first_prompt_len, len(loss_mask))
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
