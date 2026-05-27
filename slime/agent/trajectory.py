"""Token-level trajectory helpers for agent rollouts."""

from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any

from slime.utils.types import Sample


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """Exact token snapshot for one assistant generation.

    ``prompt_ids`` is the full tokenized prompt sent to the generator for that
    turn. ``output_ids`` is the raw generated output.
    """

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str


@dataclasses.dataclass(frozen=True)
class TokenSegment:
    """One training segment assembled from an agent trajectory."""

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class TurnSegment:
    """A frozen group of turns before token-level merge."""

    turns: list[TurnRecord]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def make_turn_segment(
    turns: list[TurnRecord],
    *,
    kind: str = "",
    metadata: dict[str, Any] | None = None,
) -> TurnSegment:
    """Freeze turns and attach conventional segment metadata."""
    frozen_turns = list(turns)
    segment_metadata = dict(metadata or {})
    if kind:
        segment_metadata.setdefault("segment_kind", kind)
    segment_metadata.setdefault("finish_reason", frozen_turns[-1].finish_reason if frozen_turns else "")
    return TurnSegment(turns=frozen_turns, metadata=segment_metadata)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def merge_turns(turns: list[TurnRecord], *, metadata: dict[str, Any] | None = None) -> TokenSegment | None:
    """Replay turn records into one linear training segment.

    The first turn's prompt becomes the segment prompt. Later turn prompts are
    stitched against ``prompt + response_so_far``. Any new prompt suffix is
    non-model context and receives loss mask 0. If a later prompt only matches
    an earlier prefix, the unstitched response suffix is discarded so stale
    model output is not trained and does not condition later turns.
    """
    if not turns:
        return None

    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []

    for i, turn in enumerate(turns):
        if i > 0:
            if turn.prompt_ids[: len(prompt_ids)] != prompt_ids:
                logger.warning("[trajectory] merge prompt base changed; starting segment from drifted prompt")
                prompt_ids = list(turn.prompt_ids)
                response_ids = []
                loss_mask = []
            else:
                prompt_suffix = turn.prompt_ids[len(prompt_ids) :]
                matched_len = _common_prefix_len(response_ids, prompt_suffix)
                if matched_len < len(response_ids):
                    logger.warning(
                        "[trajectory] merge prefix drift; truncating %d unstitched response tokens",
                        len(response_ids) - matched_len,
                    )
                    response_ids = response_ids[:matched_len]
                    loss_mask = loss_mask[:matched_len]

                context_tail = prompt_suffix[matched_len:]
                response_ids.extend(context_tail)
                loss_mask.extend([0] * len(context_tail))

        response_ids.extend(turn.output_ids)
        loss_mask.extend([1] * len(turn.output_ids))

    return TokenSegment(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        metadata=dict(metadata or {}),
    )


def merge_turn_segments(
    segments: list[TurnSegment],
    *,
    max_context_tokens: int = 0,
) -> list[TokenSegment]:
    """Merge frozen turn segments and drop empty or oversized outputs."""
    out: list[TokenSegment] = []
    for turn_segment in segments:
        token_segment = merge_turns(turn_segment.turns, metadata=turn_segment.metadata)
        if token_segment is None:
            continue
        total_tokens = len(token_segment.prompt_ids) + len(token_segment.response_ids)
        if token_segment.response_ids and (max_context_tokens <= 0 or total_tokens <= max_context_tokens):
            out.append(token_segment)
    return out


def write_segment_to_sample(sample: Sample, segment: TokenSegment, reward: float, tokenizer) -> None:
    """Populate token, mask, response, reward, and status fields from a segment."""
    sample.tokens = list(segment.prompt_ids) + list(segment.response_ids)
    sample.response_length = len(segment.response_ids)
    sample.loss_mask = list(segment.loss_mask)
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
    rollout_id: int | None = None,
) -> list[Sample]:
    """Emit one Sample per segment, splitting reward uniformly across them.

    Sibling samples share ``rollout_id`` so reducers that average by rollout do
    not over-count trajectories split by compaction or sub-agent dispatch.
    """
    k = len(segments)
    per_segment_reward = float(reward) / max(1, k)
    shared_rollout_id = getattr(sample, "index", None) if rollout_id is None else rollout_id
    base_metadata = {**(sample.metadata or {}), **(metadata or {})}

    out: list[Sample] = []
    for i, segment in enumerate(segments):
        sub = sample if i == 0 else copy.copy(sample)
        write_segment_to_sample(sub, segment, per_segment_reward, tokenizer)
        sub.rollout_id = shared_rollout_id
        sub.metadata = {
            **base_metadata,
            **(segment.metadata or {}),
            "segment_idx": i,
            "num_segments": k,
        }
        out.append(sub)
    return out
