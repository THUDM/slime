"""Turn-node trajectory manager for agent rollouts.

A session's trajectory is modeled as a **text-prefix tree**: each ``Node`` is one
turn (incremental prompt segment + response segment). Text-prefix matching is the
primary signal; token-id comparison is secondary and only handles TITO
(text-in-token-out) drift after the text prefix already matches. ``export`` walks
the tree once and yields full-length ``(tokens, masks, logprobs)`` triples, one
sample per leaf.

The core algorithm (``PromptSeg``/``RespSeg``/``Node``/``TrajectoryTree``/
``MatchResult`` and ``match_prefix``/``attach_turn``/``record_turn``/``export``)
is the standalone turn-node design; see
``0601-Trajectory-manager/02-turn-node/02-turn-node-design.md``.

The slime glue below the core (``TokenSegment``/``export_token_segments``/
``fan_out_sample_segments``/``write_segment_to_sample``) is the only intentional
deviation from the design's zero-dependency constraint: it imports ``Sample`` and
adapts the full-length export into slime's prompt/response training segments.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# =============================================================================
# Core turn-node tree (standalone design; zero training-side dependency)
# =============================================================================


@dataclass
class PromptSeg:
    text: str
    tokens: list[int]


@dataclass
class RespSeg:
    text: str
    tokens: list[int]
    logprobs: list[float]


@dataclass
class Node:
    id: str
    prompt_delta_text: str
    prompt_delta_tokens: list[int]
    resp_text: str
    resp_tokens: list[int]
    resp_logprobs: list[float]
    parent: Node | None
    children: list[Node] = field(default_factory=list)
    loss_mask: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    replaced_count: int = 0
    warning: str | None = None
    kind_is_root: bool = False

    @staticmethod
    def new_root() -> Node:
        now = time.time()
        return Node(
            id=uuid.uuid4().hex[:12],
            prompt_delta_text="",
            prompt_delta_tokens=[],
            resp_text="",
            resp_tokens=[],
            resp_logprobs=[],
            parent=None,
            children=[],
            created_at=now,
            updated_at=now,
            kind_is_root=True,
        )

    @staticmethod
    def new_turn(prompt: PromptSeg, resp: RespSeg, parent: Node) -> Node:
        now = time.time()
        node = Node(
            id=uuid.uuid4().hex[:12],
            prompt_delta_text=prompt.text,
            prompt_delta_tokens=list(prompt.tokens),
            resp_text=resp.text,
            resp_tokens=list(resp.tokens),
            resp_logprobs=list(resp.logprobs),
            parent=parent,
            children=[],
            created_at=now,
            updated_at=now,
        )
        parent.children.append(node)
        return node

    @property
    def node_text(self) -> str:
        return self.prompt_delta_text + self.resp_text


@dataclass
class TrajectoryTree:
    root: Node = field(default_factory=Node.new_root)


@dataclass
class MatchResult:
    case: str
    anchor: Node | None = None
    text_matched_len: int = 0
    drift_nodes: list[Node] = field(default_factory=list)
    lca: Node | None = None
    fork_node: Node | None = None
    residual_count: int = 0


# ---- helpers ----


def _text_lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _chain_nodes(node: Node) -> list[Node]:
    """root->node path, non-root nodes, ordered root to leaf."""
    out = []
    cur = node
    while cur is not None and not cur.kind_is_root:
        out.append(cur)
        cur = cur.parent
    out.reverse()
    return out


def _chain_text(nodes: list[Node]) -> str:
    return "".join(n.node_text for n in nodes)


def _collect_leaves(root: Node) -> list[Node]:
    """Pre-order collect of all leaves (excluding root; empty if root childless)."""
    leaves = []

    def walk(n: Node):
        if not n.children:
            if not n.kind_is_root:
                leaves.append(n)
            return
        for c in n.children:
            walk(c)

    walk(root)
    return leaves


def _log_replace(case, reason, node, before_text, after_text, before_tokens, after_tokens):
    logger.warning(
        "trajectory replace | case=%s reason=%s node=%s\n" "  text : %r -> %r\n" "  token: %s -> %s",
        case,
        reason,
        node.id,
        before_text,
        after_text,
        before_tokens,
        after_tokens,
    )


# ---- match_prefix ----


def match_prefix(tree: TrajectoryTree, turns) -> MatchResult:
    incoming_text = "".join(p.text + r.text for (p, r) in turns)

    # For each root->leaf chain take the longest text prefix; pick the longest,
    # ties broken by the latest created_at leaf.
    best_leaf = None
    best_len = -1
    for leaf in _collect_leaves(tree.root):
        nodes = _chain_nodes(leaf)
        ctext = _chain_text(nodes)
        lcp = _text_lcp(ctext, incoming_text)
        if lcp > best_len or (lcp == best_len and best_leaf is not None and leaf.created_at > best_leaf.created_at):
            best_leaf = leaf
            best_len = lcp

    # Case 1: no text overlap at all
    if best_leaf is None or best_len == 0:
        return MatchResult(case="case1", anchor=tree.root, text_matched_len=0)

    nodes = _chain_nodes(best_leaf)
    ctext = _chain_text(nodes)

    # §7.1: incoming text <= chain text and fully covered (no new tail, incl. equal)
    if best_len == len(incoming_text) and len(incoming_text) <= len(ctext):
        return MatchResult(case="substring", anchor=best_leaf, text_matched_len=best_len)

    # Text reached leaf and incoming has a new tail -> TITO decision (Case 2/3/4)
    if best_len == len(ctext) and len(incoming_text) > len(ctext):
        drift = []
        for i, node in enumerate(nodes):
            p, r = turns[i]
            if list(p.tokens) != node.prompt_delta_tokens or list(r.tokens) != node.resp_tokens:
                drift.append(node)
        if not drift:
            return MatchResult(case="case2", anchor=best_leaf, text_matched_len=best_len)
        only_tail = len(drift) == 1 and drift[0] is nodes[-1]
        return MatchResult(
            case="case3" if only_tail else "case4",
            anchor=best_leaf,
            text_matched_len=best_len,
            drift_nodes=drift,
        )

    # Text stops inside the chain (0 < best_len < len(ctext)) -> Case 5/6
    cum = 0
    fork_node = None
    fork_idx = 0
    for idx, node in enumerate(nodes):
        seg_len = len(node.node_text)
        if cum + seg_len > best_len:
            fork_node = node
            fork_idx = idx
            break
        cum += seg_len
    if fork_node is None:
        fork_node = nodes[-1]
        fork_idx = len(nodes) - 1
    residual_count = len(nodes) - fork_idx  # X..leaf
    lca = fork_node.parent
    return MatchResult(
        case="case5" if residual_count <= 1 else "case6",
        anchor=best_leaf,
        text_matched_len=best_len,
        lca=lca,
        fork_node=fork_node,
        residual_count=residual_count,
    )


# ---- attach_turn ----


def attach_turn(tree: TrajectoryTree, match: MatchResult, turns) -> Node | None:
    last_p, last_r = turns[-1]

    if match.case == "substring":
        logger.warning(
            "trajectory drop | reason=substring-prefix redelivery; "
            "incoming is a true prefix of an existing chain, dropped"
        )
        return None

    if match.case == "case1":
        return Node.new_turn(last_p, last_r, parent=tree.root)

    if match.case == "case2":
        return Node.new_turn(last_p, last_r, parent=match.anchor)

    if match.case in ("case3", "case4"):
        nodes = _chain_nodes(match.anchor)
        # Replace each drifted node's segment in place (same-index turn segment).
        idx_by_node = {id(n): i for i, n in enumerate(nodes)}
        for node in match.drift_nodes:
            i = idx_by_node[id(node)]
            p, r = turns[i]
            before_text, before_tokens = node.node_text, list(node.resp_tokens)
            node.prompt_delta_text = p.text
            node.prompt_delta_tokens = list(p.tokens)
            node.resp_text = r.text
            node.resp_tokens = list(r.tokens)
            node.resp_logprobs = [0.0] * len(r.tokens)  # §2.4 placeholder
            node.replaced_count += 1
            node.updated_at = time.time()
            reason = (
                "tokenizer TITO drift at tail turn"
                if match.case == "case3"
                else "multi-turn token drift (template changed?)"
            )
            _log_replace(match.case, reason, node, before_text, node.node_text, before_tokens, node.resp_tokens)
        # mask interval
        if match.case == "case3":
            match.drift_nodes[0].loss_mask = True
        else:
            first = match.drift_nodes[0]
            first.warning = "multi-turn token drift"
            start = idx_by_node[id(first)]
            for node in nodes[start:]:
                node.loss_mask = True
        # append the new turn at the tail
        return Node.new_turn(last_p, last_r, parent=nodes[-1])

    if match.case == "case5":
        nodes = _chain_nodes(match.anchor)
        x = match.fork_node
        xi = nodes.index(x)
        p, r = turns[xi]
        before_text, before_tokens = x.node_text, list(x.resp_tokens)
        x.prompt_delta_text = p.text
        x.prompt_delta_tokens = list(p.tokens)
        x.resp_text = r.text
        x.resp_tokens = list(r.tokens)
        x.resp_logprobs = [0.0] * len(r.tokens)
        x.loss_mask = True
        x.replaced_count += 1
        x.updated_at = time.time()
        _log_replace(
            "case5", "upstream response format rewrite", x, before_text, x.node_text, before_tokens, x.resp_tokens
        )
        # if turns has newer turns after x, append them in order
        parent = x
        for j in range(xi + 1, len(turns)):
            pj, rj = turns[j]
            parent = Node.new_turn(pj, rj, parent=parent)
        return parent

    if match.case == "case6":
        nodes = _chain_nodes(match.anchor)
        x = match.fork_node
        xi = nodes.index(x)
        lca = match.lca  # = x.parent
        parent = lca
        last = None
        for j in range(xi, len(turns)):
            pj, rj = turns[j]
            last = Node.new_turn(pj, rj, parent=parent)
            parent = last
        return last

    raise ValueError(f"unknown match case: {match.case}")


# ---- record_turn ----


def record_turn(tree: TrajectoryTree, turns) -> Node | None:
    match = match_prefix(tree, turns)
    return attach_turn(tree, match, turns)


# ---- export ----


def _subtree_has_trainable(node: Node) -> bool:
    """Whether any node in the subtree (rooted at ``node``) carries a trainable
    response segment (``loss_mask`` False and non-empty ``resp_tokens``). Used by
    owner selection so a fork's owner branch is not later dropped by the slice
    layer for being all-mask."""
    stack = [node]
    while stack:
        cur = stack.pop()
        if not cur.kind_is_root and not cur.loss_mask and cur.resp_tokens:
            return True
        stack.extend(cur.children)
    return False


def _decide_fork_owners(root: Node) -> dict:
    owners = {}

    def walk(n: Node):
        if len(n.children) >= 2:
            # Decision E: prefer the earliest child whose subtree still has a
            # trainable leaf, so the shared prefix it owns survives slicing.
            # Fall back to the earliest child when every branch is all-mask
            # (then the shared prefix has nowhere trainable to go anyway).
            trainable = [c for c in n.children if _subtree_has_trainable(c)]
            pool = trainable or list(n.children)
            owner = min(pool, key=lambda c: c.created_at)
            owners[id(n)] = owner
        for c in n.children:
            walk(c)

    walk(root)
    return owners


def _masked_by_fork(node: Node, leaf: Node, fork_owner: dict) -> bool:
    """Whether ``node`` is force-masked because some fork ancestor ``F`` on the
    leaf's path chose a non-owner branch toward ``leaf``: i.e. ``node`` is at or
    before ``F`` (incl. ``F`` itself) and at ``F`` the leaf's branch != owner."""
    # leaf's root->leaf path (incl. root)
    path = []
    cur = leaf
    while cur is not None:
        path.append(cur)
        cur = cur.parent
    path.reverse()  # root ... leaf
    pos = {id(n): i for i, n in enumerate(path)}
    node_pos = pos.get(id(node))
    if node_pos is None:
        return False
    # for every fork point F on path (present in fork_owner)
    for i, F in enumerate(path):
        if id(F) not in fork_owner:
            continue
        # leaf's chosen child at F = path[i+1]
        if i + 1 >= len(path):
            continue
        chosen_child = path[i + 1]
        owner = fork_owner[id(F)]
        if chosen_child is not owner:
            # non-owner branch: mask nodes at or before F (root..F)
            if node_pos <= i:
                return True
    return False


def export(tree: TrajectoryTree):
    samples, masks, logprobs = [], [], []
    fork_owner = _decide_fork_owners(tree.root)
    for leaf in _collect_leaves(tree.root):
        nodes = _chain_nodes(leaf)
        toks, mask, lp = [], [], []
        for node in nodes:
            toks += node.prompt_delta_tokens
            mask += [0] * len(node.prompt_delta_tokens)
            lp += [0.0] * len(node.prompt_delta_tokens)
            toks += node.resp_tokens
            if _masked_by_fork(node, leaf, fork_owner):
                bit = 0
            elif node.loss_mask:
                bit = 0
            else:
                bit = 1
            mask += [bit] * len(node.resp_tokens)
            lp += list(node.resp_logprobs)
        samples.append(toks)
        masks.append(mask)
        logprobs.append(lp)
    return samples, masks, logprobs


# =============================================================================
# slime glue: full-length export -> prompt/response TokenSegment + Sample fan-out
# =============================================================================


@dataclasses.dataclass(frozen=True)
class TokenSegment:
    """One training segment assembled from an agent trajectory.

    slime training invariants: ``tokens = prompt_ids + response_ids``,
    ``response_length = len(response_ids)`` and
    ``len(loss_mask) == len(rollout_log_probs) == response_length``.
    """

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    rollout_log_probs: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def export_token_segments(tree: TrajectoryTree, *, metadata: dict[str, Any] | None = None) -> list[TokenSegment]:
    """Turn the tree's full-length export into slime ``TokenSegment``s.

    Each leaf chain is split at the first trainable token (design §6,
    ``mask.index(1)``): tokens before the cut are the prompt, tokens from the cut
    on are the response. Leaves whose whole chain is masked (no trainable token,
    e.g. fork non-owner or Case4/5 fully-masked) are dropped -- they have no
    trainable response and would break slime's ``response_length`` contract.
    """
    out: list[TokenSegment] = []
    tokens_list, masks_list, logprobs_list = export(tree)
    for tokens, mask, logprobs in zip(tokens_list, masks_list, logprobs_list, strict=True):
        if 1 not in mask:
            continue  # all-mask chain: nothing trainable, drop
        cut = mask.index(1)
        response_ids = tokens[cut:]
        if not response_ids:
            continue
        segment = TokenSegment(
            prompt_ids=list(tokens[:cut]),
            response_ids=list(response_ids),
            loss_mask=list(mask[cut:]),
            rollout_log_probs=list(logprobs[cut:]),
            metadata={**(metadata or {}), "segment_kind": "leaf"},
        )
        assert len(segment.loss_mask) == len(segment.rollout_log_probs) == len(segment.response_ids)
        out.append(segment)
    return out


def write_segment_to_sample(sample: Sample, segment: TokenSegment, reward: float, tokenizer) -> None:
    """Populate token, mask, response, reward, and status fields from a segment."""
    sample.tokens = list(segment.prompt_ids) + list(segment.response_ids)
    sample.response_length = len(segment.response_ids)
    sample.loss_mask = list(segment.loss_mask)
    sample.rollout_log_probs = list(segment.rollout_log_probs)
    sample.response = tokenizer.decode(segment.response_ids, skip_special_tokens=False)
    sample.reward = float(reward)
    sample.status = Sample.Status.COMPLETED


def fan_out_sample_segments(
    sample: Sample,
    segments: list[TokenSegment],
    reward: float,
    tokenizer,
    *,
    metadata: dict[str, Any] | None = None,
) -> list[Sample]:
    """Emit one Sample per segment, splitting reward uniformly across them.

    Sibling samples share ``group_id`` so reducers that average by group do
    not over-count trajectories split by compaction or sub-agent dispatch.
    """
    k = len(segments)
    per_segment_reward = float(reward) / max(1, k)
    shared_group_id = sample.group_id if sample.group_id is not None else sample.index
    base_metadata = {**(sample.metadata or {}), **(metadata or {})}

    out: list[Sample] = []
    for i, segment in enumerate(segments):
        sub = sample if i == 0 else copy.copy(sample)
        write_segment_to_sample(sub, segment, per_segment_reward, tokenizer)
        sub.group_id = shared_group_id
        sub.metadata = {
            **base_metadata,
            **(segment.metadata or {}),
            "segment_idx": i,
            "num_segments": k,
        }
        out.append(sub)
    return out
