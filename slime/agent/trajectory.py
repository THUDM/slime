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
    turn. ``output_ids`` is the raw generated output. ``output_loss_mask`` is
    normally all 1s, but can be zeroed by per-turn validation.
    """

    prompt_ids: list[int]
    output_ids: list[int]
    output_loss_mask: list[int]
    finish_reason: str


@dataclasses.dataclass(frozen=True)
class TokenSegment:
    """One training segment assembled from an agent trajectory."""

    prompt_ids: list[int]
    response_ids: list[int]
    loss_mask: list[int]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def _prompt_matches_current(prompt_ids: list[int], base_prompt: list[int], response_ids: list[int]) -> bool:
    expected_len = len(base_prompt) + len(response_ids)
    return (
        prompt_ids[: len(base_prompt)] == base_prompt and prompt_ids[len(base_prompt) : expected_len] == response_ids
    )


def _output_mask(turn: TurnRecord) -> list[int]:
    if len(turn.output_loss_mask) == len(turn.output_ids):
        return list(turn.output_loss_mask)
    logger.warning(
        "[trajectory] turn mask length mismatch; zeroing output mask (%d ids, %d mask)",
        len(turn.output_ids),
        len(turn.output_loss_mask),
    )
    return [0] * len(turn.output_ids)


def merge_turns(turns: list[TurnRecord], *, metadata: dict[str, Any] | None = None) -> TokenSegment | None:
    """Replay turn records into one linear training segment.

    The first turn's prompt becomes the segment prompt. Later turn prompts are
    expected to start with ``prompt + response_so_far``; their suffix is new
    non-model context and receives loss mask 0, followed by the turn output and
    its per-turn output mask.
    """
    if not turns:
        return None

    prompt_ids = list(turns[0].prompt_ids)
    response_ids: list[int] = []
    loss_mask: list[int] = []

    for i, turn in enumerate(turns):
        if i > 0:
            expected_len = len(prompt_ids) + len(response_ids)
            if _prompt_matches_current(turn.prompt_ids, prompt_ids, response_ids):
                context_tail = turn.prompt_ids[expected_len:]
                response_ids.extend(context_tail)
                loss_mask.extend([0] * len(context_tail))
            elif turn.prompt_ids[: len(prompt_ids)] == prompt_ids:
                logger.warning("[trajectory] merge prefix drift; rebaselining segment")
                response_ids = list(turn.prompt_ids[len(prompt_ids) :])
                loss_mask = [0] * len(response_ids)
            else:
                logger.warning("[trajectory] merge prompt base changed; starting segment from drifted prompt")
                prompt_ids = list(turn.prompt_ids)
                response_ids = []
                loss_mask = []

        response_ids.extend(turn.output_ids)
        loss_mask.extend(_output_mask(turn))

    return TokenSegment(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        metadata=dict(metadata or {}),
    )


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
