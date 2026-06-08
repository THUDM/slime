"""Per-role chunk-merging trajectory tree manager (C-plan: token-faithful).

Design (Plan C, 2026-06-03):

* The tree is a router only. DFS merge keys on ``(role, node_match_key)``
  alone — no prompt_ids prefix check. Same conversation prefix in
  ``messages`` space always lands on the same path, regardless of any
  chat_template re-tokenization drift across turns.

* Each assistant leaf stores the THIS-TURN sglang snapshot:
  ``turn_prompt_ids`` / ``turn_response_ids`` / ``turn_response_logprobs``
  / ``turn_finish_reason`` / ``turn_index``. Non-assistant nodes carry no
  token attribution at all.

* ``get_trajectory`` linearizes each leaf turn-by-turn using LCP-aligned
  drop-and-replace: the cumulative tokens emitted so far are clamped to
  the longest common prefix with the next turn's prompt; any prior tokens
  past that LCP (the TITO drift suffix, including the previous turn's
  response if it lands in the drift region) are DROPPED along with their
  logprobs, then ``prompt[LCP:]`` is appended as loss_mask=0 (chat
  template's authoritative re-rendering wins), then the current turn's
  ``response`` is appended as loss_mask=1 with real logprobs.

* Trade-off: previous-turn response tokens that fall inside the drift
  region lose their training signal. In exchange, the final tokens
  sequence matches what the live model actually conditioned on for every
  later turn — logprobs stay coherent, no duplicated-content forks, no
  reliance on chat_template being position-invariant.

* Drift fork (gated by ``drift_fork_min_loss_tokens``, default 1024 — ON
  by default): when a drift would drop >= N loss_mask=1 tokens, the leaf
  FORKS. This is the primary drift path. We emit an extra synthetic
  "drift_fork" Sample at drain time alongside the main Sample.
  Fork tokens = cumulative pre-drop; fork loss_mask is COMPLEMENTARY — 1
  only at positions that the main leaf is about to drop, 0 elsewhere. Fork
  reward = main-leaf share; fork group_id = main-leaf group_id. Fork ∪ main
  on loss_mask=1 tokens never overlap and their union equals the virtual
  no-drift trajectory. The forked drift is NOT counted in the main sample's
  ``tito_dropped_*`` (it wasn't truly lost).

  When the drift would drop < N loss_mask=1 tokens (or is a pure-prompt
  drift losing 0 loss tokens), the secondary DROP path applies instead:
  drop-and-replace on the main leaf only, accounted in ``tito_dropped_*``.

* On drift, ``Sample.metadata`` records:
    ``tito_dropped_tokens``         — total tokens dropped (NOT including
                                      drifts that produced a fork)
    ``tito_dropped_turns``          — number of turns that triggered a drop
    ``tito_drift_forks_emitted``    — set on main leaf when >=1 drift-fork
                                      sibling was emitted for the same leaf
    ``tito_drift_fork``             — True on a drift-fork Sample
    ``tito_drift_fork_at_turn``     — turn index whose drift triggered it
    ``tito_drift_fork_loss_tokens`` — count of loss_mask=1 tokens in fork

* Fork-merge rescue (gated by ``fork_merge_max_response_tokens``, default
  1024 — ON by default; set <=0 to disable): a routing-time mechanism,
  independent of the
  linearization-time drift_fork/drop above. When DFS breaks at an assistant
  group (a later replay reformats an earlier assistant message, e.g.
  tool_call arg ordering or whitespace) and exactly one leaf sibling carries
  a per-turn response shorter than the threshold, the rewrite is collapsed
  onto that sibling instead of spawning a new sibling subtree. The merged
  sibling's stale response then enters trajectories with loss_mask=0,
  recorded in ``fork_merge_masked_tokens`` / ``fork_merge_turns``. drift_fork
  handles token alignment; fork-merge prevents a routing fork from forming.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
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
# node_match_key + role-grouping helpers
# ===========================================================================


def node_match_key(messages: list[dict[str, Any]]) -> str:
    """Identity key for a node's message list.

    json.dumps(sort_keys=True) sorts dict-internal keys recursively; list
    element order is preserved (which is what we want: message order and
    tool_calls order are both semantically significant).
    """
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


@dataclass
class _PromptGroup:
    role: str
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _DriftFork:
    """Pre-drop fork collected when TITO drift would lose >= threshold loss tokens."""

    tokens: list[int]
    # Complementary mask: 0 at positions main leaf keeps, 1 at positions main leaf will drop.
    loss_mask: list[int]
    logprobs: list[float]
    # asst.turn_index whose prompt triggered the drift.
    drift_turn_index: int | None
    # finish_reason of the prior assistant turn — describes "what the prefix
    # looked like" before drift. Captured at fork creation so the build
    # path doesn't have to look back into the chain.
    prev_finish_reason: str | None


@dataclass
class _LeafAccum:
    """Result of walking one leaf's assistant chain."""

    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    drift_forks: list[_DriftFork] = field(default_factory=list)
    # accounting (emitted to main sample's metadata when > 0)
    dropped_tokens: int = 0
    dropped_turns: int = 0
    fork_merge_masked_tokens: int = 0
    fork_merge_turns: int = 0


def _group_messages_by_role(
    messages: list[dict[str, Any]],
) -> list[_PromptGroup]:
    groups: list[_PromptGroup] = []
    for m in messages:
        role = m.get("role")
        if not isinstance(role, str):
            logger.warning("skipping message without string role: %r", m)
            continue
        if groups and groups[-1].role == role:
            groups[-1].messages.append(m)
        else:
            groups.append(_PromptGroup(role=role, messages=[m]))
    return groups


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
    mounts >=0 prompt nodes (under the deepest matching ancestor) + exactly
    1 assistant leaf carrying that turn's sglang snapshot.
    """

    def __init__(
        self,
        *,
        drift_fork_min_loss_tokens: int = 1024,
        fork_merge_max_response_tokens: int = 1024,
    ) -> None:
        # Drift-fork threshold (loss_mask=1 token count inside drift suffix).
        # When a drift would drop >= this many loss tokens, fork instead of
        # dropping. Always an int (default 1024 — drift-fork ON by default).
        # Set <=0 to effectively disable (combined with the drift_loss_tokens>0
        # guard, only true drifts above the threshold ever fork).
        self._fork_threshold: int = drift_fork_min_loss_tokens
        # Fork-merge threshold: when DFS would break at an assistant group and
        # exactly one non-leaf assistant sibling has turn_response_ids length
        # STRICTLY LESS than this value, collapse the would-be fork onto that
        # sibling (its response then enters trajectories with loss_mask=0).
        # Default 1024 — fork-merge ON by default; set <=0 to disable.
        self._fork_merge_threshold: int = fork_merge_max_response_tokens
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
        groups = _group_messages_by_role(prompt_messages)

        cur, i = self._find_mount_point(root, groups)
        cur, i = self._try_fork_merge_assistant(sid, cur, groups, i)
        cur = self._mount_prompt_groups(cur, groups[i:], tools)
        self._attach_assistant_leaf(sid, cur, turn=turn, response_message=response_message, metadata=metadata)

    def _find_mount_point(self, root: Node, groups: list[_PromptGroup]) -> tuple[Node, int]:
        """DFS down the existing tree by ``(role, node_match_key)``.

        Returns ``(cur, i)``: ``cur`` is the deepest node whose path matches
        ``groups[:i]`` exactly; ``i`` is the index into ``groups`` of the first
        group that diverges from anything mounted so far (i.e., where this
        turn's new content begins).
        """
        cur = root
        i = 0
        while i < len(groups):
            g_key = node_match_key(groups[i].messages)
            match = next(
                (c for c in cur.children if c.role == groups[i].role and node_match_key(c.messages) == g_key),
                None,
            )
            if match is None:
                break
            cur = match
            i += 1
        return cur, i

    def _try_fork_merge_assistant(self, sid: str, cur: Node, groups: list[_PromptGroup], i: int) -> tuple[Node, int]:
        """Optionally collapse an assistant-rewrite onto a single short leaf sibling.

        The typical claude-code pattern is: a later replay reformats an earlier
        assistant message (e.g. tool_call arg ordering, whitespace), which
        breaks DFS at that assistant group. Without rescue, every such reformat
        spawns a new sibling subtree; with rescue, when the existing sibling's
        per-turn response is short enough that masking it out is the cheaper
        trade-off, we collapse onto that sibling and mark it for mask=0 at
        linearization.
        """
        if self._fork_merge_threshold <= 0:
            return cur, i  # feature off
        if i >= len(groups) or groups[i].role != "assistant":
            return cur, i  # feature on, but this turn isn't an asst rewrite

        candidates = [
            c
            for c in cur.children
            if c.role == "assistant"
            # Real rewrite footprint = the original turn's leaf assistant
            # node: it was inserted by step 3 of a prior turn (so
            # turn_response_ids is populated), and no later turn has
            # extended it (so it is still a leaf). Once a rewrite collapses
            # onto it via this rescue, the new turn's user/tool + asst leaf
            # are appended underneath as children — so the merge target
            # MUST be a leaf at decision time, otherwise it has already
            # diverged into mixed subchains and merging would tangle them.
            and not c.children
            and c.turn_response_ids is not None
            and len(c.turn_response_ids) < self._fork_merge_threshold
        ]
        if len(candidates) != 1:
            if len(candidates) >= 2:
                # Legacy / pathological state: fork_merge wasn't on during
                # earlier rewrites, or threshold was widened mid-session.
                # Don't pick arbitrarily — fork as usual and surface a hint.
                logger.warning(
                    "append_turn(sid=%s turn=%s): multiple eligible fork-merge "
                    "candidates (%d), refusing to merge — likely a legacy state "
                    "from a prior run without fork_merge enabled; forking instead.",
                    sid,
                    self._turn_count.get(sid, 0) + 1,
                    len(candidates),
                )
            return cur, i

        sib = candidates[0]
        masked = len(sib.turn_response_ids or [])
        logger.warning(
            "append_turn(sid=%s turn=%s): fork-merging assistant rewrite "
            "into existing sibling (turn_index=%s, masked_response_tokens=%d)",
            sid,
            self._turn_count.get(sid, 0) + 1,
            sib.turn_index,
            masked,
        )
        sib.metadata["fork_merged"] = True
        sib.metadata["fork_merge_masked_tokens"] = masked
        sib.messages = list(groups[i].messages)
        return sib, i + 1

    def _mount_prompt_groups(
        self,
        cur: Node,
        remaining_groups: list[_PromptGroup],
        tools: list[dict[str, Any]] | None,
    ) -> Node:
        """Attach each remaining prompt group as a routing node under ``cur``.

        Token attribution happens at get_trajectory time, not here. The tools
        metadata is placed only on the FIRST system node on the path —
        ``_first_system_already_set(cur)`` walks ``cur → root`` looking for a
        system ancestor that already carries it, and ``cur`` here is the
        deepest node from descent (+ optional merge), so the walk sees every
        ancestor that's already mounted.
        """
        for g in remaining_groups:
            md: dict[str, Any] = {}
            if g.role == "system" and tools is not None and not self._first_system_already_set(cur):
                md["tools"] = list(tools)
            cur = cur.add_child(Node(role=g.role, messages=list(g.messages), metadata=md))
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
        routing leaf yields one main Sample plus one extra Sample per drift
        fork. ``reward`` is split evenly across all emitted samples. The sid is
        consumed: a second call for the same sid returns ``[]``.
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
            samples.extend(
                self._normalize_routing_leaf(sid, routing_leaf, base_sample=base_sample, extra_metadata=extra_metadata)
            )

        # Reward is split evenly across every emitted sample (main + forks);
        # the token-weighted reducer downstream then gives each loss token the
        # trajectory's full R. Assigned after the fact so the per-leaf builder
        # stays reward-agnostic.
        per_sample_reward = (reward / len(samples)) if samples else 0.0
        for s in samples:
            s.reward = per_sample_reward

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples

    # -------------------- internals ----------------------------------------

    def _normalize_routing_leaf(
        self,
        sid: str,
        leaf: Node,
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
    ) -> list[Sample]:
        """Linearize one routing leaf into its main Sample (+ drift-fork Samples).

        Drift forks come first, then the main sample, matching the original
        drain order. Reward is left at 0.0 here and assigned by the caller.
        """
        chain = leaf.path_from_root()
        # Only assistant leaves carrying this turn's sglang snapshot
        # participate in TITO accumulation. Routing assistant nodes mounted
        # from prior-turn replay (turn_prompt_ids is None) carry no token
        # signal and would otherwise be misread as a full-trajectory drift.
        asst_chain = [n for n in chain if n.role == "assistant" and n.turn_prompt_ids is not None]
        accum = self._accumulate_chain(sid, asst_chain)
        first_sys = next((n for n in chain if n.role == "system"), None)
        base_md = {"tools": first_sys.metadata.get("tools") if first_sys else None}
        first_prompt_len = len(asst_chain[0].turn_prompt_ids or []) if asst_chain else 0

        # Build one Sample from a linearized segment (a _DriftFork or the accum
        # itself — both expose tokens/loss_mask/logprobs). The base/extra/clamp
        # args are constant across segments, so this closure carries them.
        def build(seg: _DriftFork | _LeafAccum, leaf_md: dict[str, Any]) -> Sample:
            return self._build_leaf_sample(
                base_sample=base_sample,
                extra_metadata=extra_metadata,
                leaf_metadata={**base_md, **leaf_md},
                tokens=seg.tokens,
                loss_mask=seg.loss_mask,
                logprobs=seg.logprobs,
                first_prompt_len=first_prompt_len,
            )

        # Drift forks first, then the main sample — matching the original drain
        # order. A fork's loss_mask is complementary (0 on [0:L], L >=
        # first_prompt_len), so the response-region clamp never touches a loss=1
        # position and sum(loss_mask) is the final loss-token count.
        samples = [
            build(
                fork,
                {
                    "finish_reason": fork.prev_finish_reason,
                    "tito_drift_fork": True,
                    "tito_drift_fork_at_turn": fork.drift_turn_index,
                    "tito_drift_fork_loss_tokens": sum(fork.loss_mask),
                },
            )
            for fork in accum.drift_forks
        ]
        samples.append(build(accum, self._main_leaf_metadata(accum, asst_chain)))
        return samples

    @staticmethod
    def _main_leaf_metadata(accum: _LeafAccum, asst_chain: list[Node]) -> dict[str, Any]:
        """Assemble the main sample's leaf metadata (conditional drift/merge keys)."""
        last_asst = asst_chain[-1] if asst_chain else None
        md: dict[str, Any] = {"finish_reason": last_asst.turn_finish_reason if last_asst else None}
        if accum.dropped_tokens > 0:
            md["tito_dropped_tokens"] = accum.dropped_tokens
            md["tito_dropped_turns"] = accum.dropped_turns
        if accum.drift_forks:
            md["tito_drift_forks_emitted"] = len(accum.drift_forks)
        if accum.fork_merge_masked_tokens > 0:
            md["fork_merge_masked_tokens"] = accum.fork_merge_masked_tokens
            md["fork_merge_turns"] = accum.fork_merge_turns
        return md

    def _build_leaf_sample(
        self,
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        leaf_metadata: dict[str, Any],
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
        """
        loss_resp, lp_resp = self._response_region(loss_mask, logprobs, first_prompt_len)
        metadata = {
            **(base_sample.metadata or {}),
            **(extra_metadata or {}),
            **leaf_metadata,
        }
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

    def _accumulate_chain(self, sid: str, asst_chain: list[Node]) -> _LeafAccum:
        """Apply LCP drop-and-replace to an assistant chain in turn order.

        Mutates and returns ``accum``. See the module docstring for the
        algorithm, drift-fork contract, and fork-merge masking rule.
        """
        accum = _LeafAccum()
        for k, asst in enumerate(asst_chain, start=1):
            # Read-only views: only fed to _lcp_len / .extend(), never mutated
            # in place, so no defensive copy is needed here (the leaf already
            # owns isolated copies from _attach_assistant_leaf).
            prompt_ids = asst.turn_prompt_ids or []
            response_ids = asst.turn_response_ids or []
            response_logprobs = asst.turn_response_logprobs
            is_merged = bool(asst.metadata.get("fork_merged"))

            # k == 1 falls out of the general case: LCP([], prompt) == 0 so
            # _fork_or_drop_drift trivially returns the full prompt with no drop.
            prev_finish_reason = asst_chain[k - 2].turn_finish_reason if k >= 2 else None
            emit_prompt = self._fork_or_drop_drift(
                sid,
                accum,
                prev_finish_reason=prev_finish_reason,
                drift_turn_index=asst.turn_index,
                prompt=prompt_ids,
            )

            accum.tokens.extend(emit_prompt)
            accum.loss_mask.extend([0] * len(emit_prompt))
            accum.logprobs.extend([0.0] * len(emit_prompt))

            accum.tokens.extend(response_ids)
            # fork-merged sibling: its response is "stale" — present in the tree
            # only as a routing placeholder for the rewrites that collapsed onto
            # it; mask it out of training.
            accum.loss_mask.extend([0 if is_merged else 1] * len(response_ids))
            accum.logprobs.extend(response_logprobs if response_logprobs is not None else [0.0] * len(response_ids))

            if is_merged:
                accum.fork_merge_masked_tokens += len(response_ids)
                accum.fork_merge_turns += 1
        return accum

    def _fork_or_drop_drift(
        self,
        sid: str,
        accum: _LeafAccum,
        *,
        prev_finish_reason: str | None,
        drift_turn_index: int | None,
        prompt: list[int],
    ) -> list[int]:
        """Resolve a TITO drift between cumulative tokens and the next prompt.

        Compute LCP ``L``; if there's no drift suffix, just return the prompt
        tail. Otherwise decide FORK vs DROP (fork is the primary path), then
        truncate ``accum.{tokens,loss_mask,logprobs}`` to ``L`` and return
        ``prompt[L:]`` for the caller to emit.

        FORK (drift loses >= ``self._fork_threshold`` loss_mask=1 tokens):
            append a complementary-mask ``_DriftFork`` to ``accum`` so the
            dropped training signal survives as a sibling output leaf. The
            drift is NOT counted toward ``accum.dropped_*``.
        DROP (below threshold, or a pure-prompt drift losing 0 loss tokens):
            drop-and-replace only, counted toward ``accum.dropped_*``.
        """
        L = _lcp_len(accum.tokens, prompt)
        drift = len(accum.tokens) - L
        if drift == 0:
            return prompt[L:]

        drift_loss_tokens = sum(accum.loss_mask[L:])
        # PRIMARY: fork — losing >= threshold loss tokens is worth a
        # complementary-mask fork leaf so the dropped signal survives. The
        # drift_loss_tokens>0 guard keeps pure-prompt drift on the DROP path
        # even if the threshold is set to <=0.
        forked = drift_loss_tokens > 0 and drift_loss_tokens >= self._fork_threshold
        if forked:
            accum.drift_forks.append(
                _DriftFork(
                    tokens=list(accum.tokens),
                    loss_mask=[0] * L + list(accum.loss_mask[L:]),
                    logprobs=[0.0] * L
                    + [(accum.logprobs[i] if accum.loss_mask[i] == 1 else 0.0) for i in range(L, len(accum.tokens))],
                    drift_turn_index=drift_turn_index,
                    prev_finish_reason=prev_finish_reason,
                )
            )
        else:
            # SECONDARY: drop-and-replace only.
            accum.dropped_tokens += drift
            accum.dropped_turns += 1

        logger.warning(
            "get_trajectory(sid=%s leaf turn=%s): TITO drift detected, "
            "dropping %d prior tokens (incl. previous-turn response) to "
            "realign with this turn's prompt%s",
            sid,
            drift_turn_index,
            drift,
            f"; forked {drift_loss_tokens} loss tokens" if forked else "",
        )

        del accum.tokens[L:]
        del accum.loss_mask[L:]
        del accum.logprobs[L:]
        return prompt[L:]

    @staticmethod
    def _response_region(
        loss_mask: list[int],
        logprobs: list[float],
        first_prompt_len: int,
    ) -> tuple[list[int], list[float]]:
        """Strip the leading first-turn prompt prefix from loss_mask / logprobs
        so what remains is the response-region view slime expects (see
        slime/backends/megatron_utils/data.py:139, slime/ray/rollout.py:695)."""
        strip = min(first_prompt_len, len(loss_mask))
        return loss_mask[strip:], logprobs[strip:]

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
    "_group_messages_by_role",
    "_lcp_len",
]
