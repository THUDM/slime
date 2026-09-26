"""Build a per-session training trajectory from multi-turn conversation data.

The :class:`TrajectoryManager` builds one trajectory per session. ``record_turn``
feeds in each turn (prompt messages + the served model's sglang snapshot),
routing it into a per-sid message tree; ``get_trajectory`` then linearizes that
tree into a ``list[Sample]`` of loss-masked training rows, tolerating TITO
re-tokenization drift via fork/replace.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from collections.abc import Iterator
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ===========================================================================
# TurnRecord
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """One sglang ``/generate`` snapshot: the contract between an adapter and the
    manager. Adapters build it from a turn's prompt/output token ids; ``record_turn``
    consumes it."""

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    ill_formed: bool = False


# ===========================================================================
# MessageNode
# ===========================================================================


class MessageNode:
    """One node in a session's routing tree, carrying a single chat message
    (``None`` for the dummy root and for an assistant leaf we generated but
    whose ``response_message`` was empty).

    The two kinds are distinguished by whether ``turn`` is set, which reflects
    WHERE the message came from:

    * **generated** (``turn is not None``): an assistant message the model
      actually generated this turn, fed in via ``record_turn``. ``turn`` holds
      its :class:`TurnRecord` -- the prompt/output ids, logprobs and finish
      reason that ``get_trajectory`` linearizes into training tokens.
    * **routing-only** (``turn is None``): the message came from the prompt, not
      from generation, so it only exists to route. This is every
      system/user/tool node, AND any assistant we did NOT generate: a foreign
      assistant the client replayed in a later prompt, or a prior generated turn
      demoted by the rewrite-merge in ``_try_merge_assistant_rewrite``.
    """

    def __init__(
        self,
        *,
        role: str | None = None,
        message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: MessageNode | None = None,
    ) -> None:
        self.role = role
        self.message = message
        self.metadata = dict(metadata or {})
        self.parent: MessageNode | None = parent
        self.children: list[MessageNode] = []
        self.turn: TurnRecord | None = None  # the generated TurnRecord, else None (routing-only)
        self.turn_index: int | None = None
        # Shared by sibling leaf paths; the first to reach it trains on it, the rest
        # re-emit it as loss_mask=0 context -- so each response is trained exactly once.
        self.response_trained: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: MessageNode) -> MessageNode:
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list[MessageNode]:
        """Ordered list of nodes from the first non-root ancestor down to self."""
        chain: list[MessageNode] = []
        node: MessageNode | None = self
        while node is not None and not node.is_root:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator[MessageNode]:
        if not self.children:
            yield self
            return
        for c in self.children:
            yield from c.leaves()


def _tool_defaults(tools_schema: list[dict] | None) -> dict[str, dict[str, Any]]:
    """``{tool name: {argument: default}}`` from a chat-template tool schema.

    Only properties that actually declare a ``default`` are collected; a tool with
    none contributes an empty mapping, and an absent schema yields ``{}``.
    """
    out: dict[str, dict[str, Any]] = {}
    for entry in tools_schema or []:
        fn = (entry or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        if not isinstance(props, dict):
            continue
        defaults = {k: v["default"] for k, v in props.items() if isinstance(v, dict) and "default" in v}
        if defaults:
            out[name] = defaults
    return out


def _strip_trailing_ws(args: Any) -> Any:
    """Right-strip every line of every string argument, recursively.

    A client may right-strip a multi-line code payload on replay. How much it
    removed is unknowable, so instead of trying to put it back, both sides are
    compared with all trailing whitespace gone.
    """
    if isinstance(args, dict):
        return {k: _strip_trailing_ws(v) for k, v in sorted(args.items())}
    if isinstance(args, list):
        return [_strip_trailing_ws(v) for v in args]
    if isinstance(args, str):
        return "\n".join(line.rstrip() for line in args.split("\n"))
    return args


def _drop_schema_defaults(args: Any, defaults: dict[str, Any]) -> Any:
    """Drop top-level arguments whose value equals the tool's declared default.

    Omitting an argument and passing its declared default are the same call, so a
    client that fills one in on replay has changed nothing the tool can observe.
    Dropping those on both sides makes the two compare equal, whatever the default
    happens to be -- ``False``, ``True``, ``0``, or a string.

    ``True is 1`` in Python, so identity rather than ``==`` guards booleans here:
    an argument of ``1`` must not be mistaken for a declared default of ``True``.
    """
    if not isinstance(args, dict) or not defaults:
        return args

    def is_default(key: str, value: Any) -> bool:
        if key not in defaults:
            return False
        default = defaults[key]
        if isinstance(default, bool) or isinstance(value, bool):
            return value is default
        return value == default

    return {k: v for k, v in args.items() if not is_default(k, v)}


def _norm_tool_args(args: Any, defaults: dict[str, Any] | None = None) -> Any:
    """Project tool-call arguments onto a coarser view, for comparison only.

    A harness may replay a prior assistant tool call re-rendered rather than
    byte-identical. Keeping arguments a dict in ``tool_call_dict`` already makes
    the comparison immune to key order; two further client edits are equally free
    of semantic content but still break equality:

    * an argument filled in with the value the tool schema already declares as its
      default (``replace_all`` on Edit, for instance), where the model omitted it;
    * per-line right-stripping of string arguments carrying code payloads.

    Neither edit is reversed -- the original bytes are never reconstructed from the
    echo, and the caller substitutes the message it stored instead (see
    ``restore_generated_messages``). Both sides are merely projected onto a view
    that cannot see the difference.

    ``defaults`` comes from the tool's own schema (see :func:`_tool_defaults`).
    Without it, only the whitespace projection applies, so a filled-in default
    still reads as a different call -- a missed restore, never a wrong one.
    """
    return _strip_trailing_ws(_drop_schema_defaults(args, defaults or {}))


def _is_same_tool_call_echo(
    held: dict[str, Any] | None,
    incoming: dict[str, Any],
    tool_defaults: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Whether ``incoming`` is ``held`` re-rendered by the harness.

    Only tool-call arguments are compared through the coarser view of
    :func:`_norm_tool_args`; everything else -- role, content, reasoning, tool
    name, call count, and any other key -- still has to match exactly. A message
    that differs in any of those is a different action and must not be mistaken
    for an echo.
    """
    if not isinstance(held, dict) or held.get("role") != incoming.get("role"):
        return False
    h_tcs, i_tcs = held.get("tool_calls"), incoming.get("tool_calls")
    if not h_tcs or not i_tcs or len(h_tcs) != len(i_tcs):
        return False
    if {k: v for k, v in held.items() if k != "tool_calls"} != {
        k: v for k, v in incoming.items() if k != "tool_calls"
    }:
        return False
    for h, i in zip(h_tcs, i_tcs, strict=True):  # equal lengths checked above
        hf, if_ = h.get("function") or {}, i.get("function") or {}
        name = hf.get("name")
        if h.get("type") != i.get("type") or name != if_.get("name"):
            return False
        defaults = (tool_defaults or {}).get(name, {})
        if _norm_tool_args(hf.get("arguments"), defaults) != _norm_tool_args(if_.get("arguments"), defaults):
            return False
    return True


# ===========================================================================
# drift classification — how an incoming turn's prompt relates to held tokens
# ===========================================================================


def _common_prefix_len(a: list[int], b: list[int], chunk: int = 4096) -> int:
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit:
        chunk_end = min(matched + chunk, limit)
        if a[matched:chunk_end] == b[matched:chunk_end]:
            matched = chunk_end
        else:
            while matched < chunk_end and a[matched] == b[matched]:
                matched += 1
            return matched
    return matched


class DriftKind(enum.Enum):
    CLEAN = "clean"  # drift == 0: prompt_ids exactly extends held tokens; append the tail beyond them
    REALIGN = "realign"  # drift inside the most-recent response span and short incoming response; replace that span (loss_mask=0)
    FORK = "fork"  # everything else: close this builder, open a fresh one as a fork


# ===========================================================================
# SampleBuilder — accumulates turns into one trainable Sample (fork closes it)
# ===========================================================================


class _SampleBuilder:
    """Accumulates a chain's turns into the token sequence of one ``Sample``.

    A chain of turns is appended one at a time via :meth:`append_turn`. Ideally
    each turn's prompt exactly extends the tokens we already hold, but a replayed
    turn rarely re-tokenizes byte-for-byte: TITO round-trips and chat-template
    re-rendering both perturb the ids of content we've already seen. The builder
    handles this drift in a source-agnostic way, classified by where and how far
    the prompt diverges from the held tokens (see :meth:`classify_token_drift`):

    * **CLEAN** -- no drift; append the prompt tail beyond what we hold.
    * **REALIGN** -- a short divergence inside the most-recent response span;
      overwrite that span from the prompt as loss_mask=0 and keep accumulating.
    * **FORK** -- divergence too large or too early to absorb; this builder is
      rejected and the caller closes it and opens a fresh one. That boundary is
      the "fork".

    Each surviving builder yields one Sample.
    """

    def __init__(self, fork_threshold: int) -> None:
        self._fork_threshold = fork_threshold
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.last_response_start_idx: int | None = None
        self.leading_prompt_len: int = 0

    def classify_token_drift(self, turn: TurnRecord) -> DriftKind:
        """Decide how this builder should absorb ``turn``'s prompt.

        The incoming turn's prompt is expected to match the tokens this builder
        already holds as an exact prefix. When token drift has occurred -- the
        prompt diverges from the held tokens -- we decide whether to REALIGN
        (heal a short divergence inside the most-recent response span) or to FORK
        (``len(turn.output_ids) >= fork_threshold``, or the divergence sits too
        early to absorb). With no drift the turn is handled the CLEAN way -- a
        plain prefix extension.
        """
        realign_at = _common_prefix_len(self.tokens, turn.prompt_ids)
        drift = len(self.tokens) - realign_at

        if drift == 0:
            return DriftKind.CLEAN

        # REALIGN only heals drift that falls inside the most-recent response span
        # (and is short); divergence anywhere earlier, or an empty builder, forks.
        start = self.last_response_start_idx
        if start is not None and realign_at >= start and len(turn.output_ids) < self._fork_threshold:
            return DriftKind.REALIGN
        return DriftKind.FORK

    def append_turn(self, turn: TurnRecord, kind: DriftKind, *, trained: bool = True) -> None:
        """Append one turn into this SampleBuilder, branching on ``kind``: for REALIGN
        we overwrite the already-saved response span, for CLEAN we just append this
        turn's prompt tail."""
        assert kind is not DriftKind.FORK, "append_turn called on a builder that would fork"

        is_first_turn = self.last_response_start_idx is None

        # --- append this turn's prompt tail (loss_mask=0) ---
        if kind is DriftKind.REALIGN:
            self._align_to_prompt(turn.prompt_ids)  # drop the drifted tail, re-append from prompt
        else:  # CLEAN: held tokens are an exact prefix of prompt_ids; append the tail beyond them
            self._append_tokens(turn.prompt_ids[len(self.tokens) :], loss_mask=0)

        # --- append this turn's generated response (loss_mask=1 unless re-emitted as context) ---
        self.last_response_start_idx = len(self.tokens)
        self._append_tokens(
            turn.output_ids, loss_mask=int(trained), logprobs=turn.output_log_probs if trained else None
        )

        if is_first_turn:
            self.leading_prompt_len = len(turn.prompt_ids)

    def _align_to_prompt(self, prompt_ids: list[int]) -> None:
        """Heal REALIGN drift by overwriting the most-recent response span with
        ``prompt_ids`` as loss_mask=0: the drifted tokens carry no signal, and re-appending
        from the prompt keeps the builder contiguous. Earlier turns are untouched."""
        response_start = self.last_response_start_idx
        tail = prompt_ids[response_start:]
        self.tokens[response_start:] = tail
        self.loss_mask[response_start:] = [0] * len(tail)
        self.logprobs[response_start:] = [0.0] * len(tail)

    def _append_tokens(self, ids: list[int], *, loss_mask: int, logprobs: list[float] | None = None) -> None:
        self.tokens.extend(ids)
        self.loss_mask.extend([loss_mask] * len(ids))
        self.logprobs.extend(logprobs if logprobs else [0.0] * len(ids))

    def has_trained_response(self) -> bool:
        return any(self.loss_mask[self.leading_prompt_len :])

    def to_sample(
        self, base_sample: Sample, extra_metadata: dict[str, Any] | None, max_sample_tokens: int = 0
    ) -> Sample:
        """Emit the accumulated tokens as one ``Sample``, stripping the first-turn
        prompt so loss_mask / logprobs cover only the response region."""
        start = self.leading_prompt_len  # first-turn prompt stripped; response region starts here
        tokens = list(self.tokens)
        loss_mask = self.loss_mask
        logprobs = self.logprobs
        if max_sample_tokens and len(tokens) > max_sample_tokens:
            tokens = tokens[:max_sample_tokens]
            loss_mask = loss_mask[:max_sample_tokens]
            logprobs = logprobs[:max_sample_tokens]
        md = dict(extra_metadata or {})
        return Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            tokens=tokens,
            response_length=len(loss_mask) - start,
            loss_mask=loss_mask[start:],
            rollout_log_probs=logprobs[start:],
            reward=0.0,
            status=Sample.Status.COMPLETED,
            metadata=md,
        )


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    def __init__(self, *, fork_threshold_tokens: int | None = None) -> None:
        self._fork_threshold: int = 1024 if fork_threshold_tokens is None else fork_threshold_tokens
        self._trees: dict[str, MessageNode] = {}
        self._turn_count: dict[str, int] = {}

    # -------------------- public ------------------------------------------

    def has_session(self, sid: str) -> bool:
        return sid in self._trees

    def turn_count(self, sid: str) -> int:
        return self._turn_count.get(sid, 0)

    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        assert not turn.output_log_probs or len(turn.output_log_probs) == len(turn.output_ids), (
            f"turn.output_log_probs length {len(turn.output_log_probs)} != "
            f"turn.output_ids length {len(turn.output_ids)}"
        )

        root = self._trees.setdefault(sid, MessageNode())

        node, depth = self._find_mount_point(root, prompt_messages)
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth)
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample: Sample,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:
        """Linearize this sid's routing tree into slime ``Sample`` objects and
        consume the session.

        Each routing leaf yields one or more Samples; ``reward`` is assigned in
        full to every emitted Sample (not split across them), so each trained
        turn carries the trajectory's outcome reward. The sid is dropped
        afterwards, so a second call for the same sid returns ``[]``.
        """
        root = self._trees.get(sid)
        if root is None:
            return []

        samples: list[Sample] = []
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(
                self._chain_to_samples(
                    chain, base_sample=base_sample, extra_metadata=extra_metadata, max_sample_tokens=max_sample_tokens
                )
            )

        for s in samples:
            s.reward = reward

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples

    def drop_session(self, sid: str) -> None:
        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)

    def restore_generated_messages(
        self,
        sid: str,
        messages: list[dict[str, Any]],
        tools_schema: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Swap a harness's re-rendered assistant echoes back to what we generated.

        A harness that replays a prior assistant turn re-rendered rather than
        byte-identical makes the next prompt diverge from the tokens we hold on
        both layers at once: the routing walk misses the node (message
        inequality, so ``_try_merge_assistant_rewrite`` demotes that turn to
        routing-only and its generated tokens stop training) and the token prefix
        drifts (the re-rendered arguments tokenize differently, so the turn
        classifies REALIGN and ``_align_to_prompt`` zeroes the span). Restoring
        before the prompt is rendered removes the divergence at its source, so
        the turn stays CLEAN and keeps its gradient.

        Only an assistant message recognized as the same tool call re-rendered
        is swapped (see :func:`_is_same_tool_call_echo`); anything else is left
        as the harness sent it, and the walk stops there. Returns a new list;
        ``messages`` is not mutated.

        ``tools_schema`` supplies each tool's declared argument defaults, so an
        argument the client filled in with its default is recognized exactly
        rather than guessed at. Passing nothing is safe: the comparison just gets
        stricter, costing a restore rather than making a wrong one.
        """
        root = self._trees.get(sid)
        if root is None:
            return messages

        defaults = _tool_defaults(tools_schema)
        out = list(messages)
        node, depth = root, 0
        while depth < len(out):
            msg = out[depth]
            exact = next((c for c in node.children if c.role == msg.get("role") and c.message == msg), None)
            if exact is not None:
                node, depth = exact, depth + 1
                continue
            # No exact child: this may be our own generated turn, echoed back
            # re-rendered. Require a single generated assistant candidate --
            # with more than one we cannot tell which the echo refers to, the
            # same ambiguity _try_merge_assistant_rewrite declines to guess at.
            echoes = [
                c
                for c in node.children
                if c.role == "assistant" and c.turn is not None and _is_same_tool_call_echo(c.message, msg, defaults)
            ]
            if len(echoes) != 1:
                break
            out[depth] = echoes[0].message
            node, depth = echoes[0], depth + 1
        return out

    # -------------------- internals ----------------------------------------

    def _find_mount_point(self, root: MessageNode, messages: list[dict[str, Any]]) -> tuple[MessageNode, int]:
        """Walk down the tree matching each message by role and dict equality (==),
        returning the deepest node that still matches and where to mount the rest."""
        node = root
        depth = 0
        while depth < len(messages):
            msg = messages[depth]
            next_child = None
            for child in node.children:
                if child.role == msg.get("role") and child.message == msg:
                    next_child = child
                    break
            if next_child is None:
                break
            node = next_child
            depth += 1
        return node, depth

    def _try_merge_assistant_rewrite(
        self,
        sid: str,
        node: MessageNode,
        prompt_messages: list[dict[str, Any]],
        depth: int,
    ) -> tuple[MessageNode, int]:
        """Merge a short assistant-rewrite onto its node instead of forking.

        A harness may replay a prior assistant message slightly re-rendered (e.g.
        whitespace) in a later prompt. It no longer matches the node we generated,
        so it would fork -- stranding the original generated turn as a dead-end
        leaf that still emits its own training Sample. Instead we overwrite that
        node's message in place and stop training its generated content (demote to
        routing-only), so only the live branch trains. This only applies below
        ``fork_threshold``: a long abandoned response carries enough real signal
        to fork and train standalone.

        Forking is always safe (a rewrite mounts as routing-only); this is purely
        a cleanup. So we merge only when the mount point has exactly one assistant
        child that is a leaf, generated (``turn`` set), and short (response <
        ``fork_threshold``), and fork otherwise, since absorbing destroys a
        generated TurnRecord irreversibly.
        """
        if self._fork_threshold <= 0:
            return node, depth  # feature off
        if depth >= len(prompt_messages) or prompt_messages[depth].get("role") != "assistant":
            return node, depth  # genuine non-assistant history fork -> leave it

        asst_children = [c for c in node.children if c.role == "assistant"]
        if len(asst_children) != 1:
            if len(asst_children) > 1:
                logger.warning(
                    "record_turn(sid=%s turn=%s): %d assistant children at mount "
                    "point; can't tell which the rewrite targets, so forking.",
                    sid,
                    self._turn_count.get(sid, 0) + 1,
                    len(asst_children),
                )
            return node, depth

        rewritten_node = asst_children[0]
        if (
            rewritten_node.children
            or rewritten_node.turn is None
            or len(rewritten_node.turn.output_ids) >= self._fork_threshold
        ):
            return node, depth

        rewritten_node.metadata["merged_rewrite"] = {
            "abandoned_turn_index": rewritten_node.turn_index,
            "abandoned_response_tokens": len(rewritten_node.turn.output_ids),
        }
        rewritten_node.turn = None
        rewritten_node.turn_index = None
        rewritten_node.message = prompt_messages[depth]
        return rewritten_node, depth + 1

    def _mount_prompt_messages(
        self,
        node: MessageNode,
        remaining_messages: list[dict[str, Any]],
    ) -> MessageNode:
        for m in remaining_messages:
            node = node.add_child(MessageNode(role=m.get("role"), message=m))
        return node

    def _attach_assistant_leaf(
        self,
        sid: str,
        node: MessageNode,
        *,
        turn: TurnRecord,
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        asst = MessageNode(
            role="assistant",
            message=response_message,
            metadata=dict(metadata or {}),
        )
        asst.turn = turn
        asst.turn_index = self._turn_count.get(sid, 0) + 1
        node.add_child(asst)
        self._turn_count[sid] = asst.turn_index

    def _split_chain_into_builders(self, chain: list[MessageNode]) -> list[_SampleBuilder]:
        """Pack the chain's generated turns into per-Sample token builders.

        Turns flow into the current builder until one can't extend it as an
        exact prefix (re-tokenization drift past what we can drop); that turn
        opens a new builder -- a fork. A generated turn shared by sibling leaves
        is trained only on the first leaf to claim it; later leaves re-emit it
        as loss_mask=0 context so the shared prefix isn't double-counted.
        """
        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]

        builders: list[_SampleBuilder] = []
        for asst_node in asst_nodes:
            trained = not asst_node.response_trained
            asst_node.response_trained = True

            if not builders or (kind := builders[-1].classify_token_drift(asst_node.turn)) is DriftKind.FORK:
                builders.append(_SampleBuilder(self._fork_threshold))
                builders[-1].append_turn(asst_node.turn, DriftKind.CLEAN, trained=trained)
            else:
                builders[-1].append_turn(asst_node.turn, kind, trained=trained)
        return builders

    def _chain_to_samples(
        self,
        chain: list[MessageNode],
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:

        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]
        truncated = bool(asst_nodes) and asst_nodes[-1].turn.finish_reason == "length"
        use_tool = any(bool((n.message or {}).get("tool_calls")) for n in asst_nodes)
        ill_formed = any(n.turn.ill_formed for n in asst_nodes)
        md = {
            **(extra_metadata or {}),
            "truncated": truncated,
            "use_tool": use_tool,
            "ill_formed": ill_formed,
        }
        return [
            builder.to_sample(base_sample, md, max_sample_tokens)
            for builder in self._split_chain_into_builders(chain)
            if builder.has_trained_response()
        ]


__all__ = [
    "TrajectoryManager",
    "TurnRecord",
]
