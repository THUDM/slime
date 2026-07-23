from __future__ import annotations

from argparse import Namespace

import pytest

from slime.utils.types import Sample

NUM_GPUS = 0


def _make_args() -> Namespace:
    return Namespace(sglang_speculative_algorithm=False)


def _resume_sample(
    sample: Sample,
    *,
    partial_rollout: bool,
    tokens: list[int],
    log_probs: list[float],
    mask_offpolicy_in_partial_rollout: bool = True,
) -> Sample:
    if partial_rollout and mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.mask_response_tokens(0)

    sample.append_response_tokens(
        _make_args(),
        tokens=tokens,
        log_probs=log_probs,
        trainable=True,
        meta_info={"finish_reason": {"type": "stop"}},
        text=" new",
    )
    sample.reward = 1.0
    return sample


@pytest.mark.unit
def test_partial_resume_masks_old_tokens_and_appends_trainable_loss_mask():
    sample = Sample(
        tokens=[1, 2, 10, 11],
        response="old",
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[-0.7, -0.8],
        status=Sample.Status.ABORTED,
    )

    _resume_sample(
        sample,
        partial_rollout=True,
        tokens=[30, 31, 32],
        log_probs=[-0.3, -0.4, -0.5],
    )

    assert sample.response_length == 5
    assert sample.loss_mask == [0, 0, 1, 1, 1]
    assert len(sample.loss_mask) == sample.response_length
    assert sample.rollout_log_probs == [-0.7, -0.8, -0.3, -0.4, -0.5]
    assert sample.status is Sample.Status.COMPLETED


@pytest.mark.unit
def test_non_partial_resume_preserves_existing_loss_mask():
    sample = Sample(
        tokens=[1, 2, 10, 11],
        response="old",
        response_length=2,
        loss_mask=[1, 0],
        rollout_log_probs=[-0.7, -0.8],
        status=Sample.Status.ABORTED,
    )

    _resume_sample(sample, partial_rollout=False, tokens=[30], log_probs=[-0.3])

    assert sample.response_length == 3
    assert sample.loss_mask == [1, 0, 1]
    assert len(sample.loss_mask) == sample.response_length


@pytest.mark.unit
def test_fresh_full_rollout_still_gets_all_trainable_loss_mask():
    sample = Sample(tokens=[1, 2], status=Sample.Status.PENDING)

    _resume_sample(sample, partial_rollout=False, tokens=[30, 31], log_probs=[-0.3, -0.4])

    assert sample.response_length == 2
    assert sample.loss_mask == [1, 1]
    assert len(sample.loss_mask) == sample.response_length
