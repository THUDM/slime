"""CPU unit tests for zero-length response handling in rollout post-processing.

Pins the contract that a response of length 0 must produce an *empty*
loss_mask / teacher_log_probs, not the whole sequence. Python's slicing
semantics make ``seq[-0:]`` return the entire sequence, so every
``seq[-response_length:]`` in the rollout path needs an explicit guard
for ``response_length == 0``:

  * ``slime/rollout/sft_rollout.py`` — a fully-masked SFT sample (no
    trainable tokens) used to end up with ``loss_mask`` covering the
    whole prompt, violating the downstream invariant
    ``len(loss_mask) == response_length`` (asserted in
    ``slime/ray/rollout.py`` and ``slime/utils/types.py``) and crashing
    the rollout.

  * ``slime/rollout/on_policy_distillation.py`` — an empty generation
    (e.g. immediate EOS) used to keep the full-length teacher log-prob
    tensor instead of an empty one, misaligning the OPD KL computation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import slime.rollout.sft_rollout as sft_rollout
from slime.rollout.on_policy_distillation import post_process_rewards
from slime.utils.mask_utils import get_response_lengths
from slime.utils.types import Sample

NUM_GPUS = 0


class _FakeMaskGenerator:
    """Stands in for MultiTurnLossMaskGenerator without needing a tokenizer."""

    def __init__(self, token_ids, loss_mask):
        self._token_ids = token_ids
        self._loss_mask = loss_mask

    def get_loss_mask(self, messages, tools=None):
        return self._token_ids, self._loss_mask

    def get_response_lengths(self, loss_masks):
        return get_response_lengths(loss_masks)


class _FakeDataBuffer:
    def __init__(self, samples):
        self._samples = samples

    def get_samples(self, n):
        return self._samples[:n]


def _run_sft_rollout(monkeypatch, token_ids, loss_mask):
    monkeypatch.setattr(sft_rollout, "TOKENIZER", object())
    monkeypatch.setattr(sft_rollout, "PROCESSOR", object())
    monkeypatch.setattr(sft_rollout, "MASK_GENERATOR", _FakeMaskGenerator(token_ids, loss_mask))
    monkeypatch.setattr(sft_rollout, "SAMPLE_PRINTED", True)

    args = SimpleNamespace(
        hf_checkpoint=None,
        loss_mask_type=None,
        rollout_global_dataset=True,
        rollout_batch_size=1,
    )
    data_buffer = _FakeDataBuffer([(Sample(prompt=[{"role": "user", "content": "hi"}]),)])
    (sample,) = sft_rollout.generate_rollout(args, 0, data_buffer)[0]
    return sample


@pytest.mark.unit
def test_sft_rollout_fully_masked_sample_yields_empty_loss_mask(monkeypatch):
    # No assistant turn / template mismatch -> no trainable tokens at all.
    sample = _run_sft_rollout(monkeypatch, [10, 11, 12, 13], [0, 0, 0, 0])

    assert sample.response_length == 0
    # Must be empty; the buggy `loss_mask[-0:]` returned the whole mask here.
    assert sample.loss_mask == []
    assert len(sample.loss_mask) == sample.response_length


@pytest.mark.unit
def test_sft_rollout_normal_sample_keeps_tail_loss_mask(monkeypatch):
    sample = _run_sft_rollout(monkeypatch, [10, 11, 12, 13], [0, 0, 1, 1])

    assert sample.response_length == 2
    assert sample.loss_mask == [1, 1]
    assert len(sample.loss_mask) == sample.response_length


class _StubOPDSample:
    """Minimal stand-in for Sample as used by post_process_rewards."""

    def __init__(self, reward_payload, response_length):
        self._reward_payload = reward_payload
        self.response_length = response_length
        self.teacher_log_probs = None

    def get_reward_value(self, args):
        return self._reward_payload


def _opd_reward_payload(logprobs):
    # Mirrors sglang's return_logprob format: the first entry has no logprob.
    return {"meta_info": {"input_token_logprobs": [(None, 0)] + [(lp, i + 1) for i, lp in enumerate(logprobs)]}}


@pytest.mark.unit
def test_opd_zero_length_response_yields_empty_teacher_log_probs():
    sample = _StubOPDSample(_opd_reward_payload([-0.1, -0.2, -0.3]), response_length=0)

    post_process_rewards(None, [sample])

    assert sample.teacher_log_probs.shape == (0,)


@pytest.mark.unit
def test_opd_normal_response_keeps_tail_teacher_log_probs():
    sample = _StubOPDSample(_opd_reward_payload([-0.1, -0.2, -0.3]), response_length=2)

    post_process_rewards(None, [sample])

    assert torch.allclose(sample.teacher_log_probs, torch.tensor([-0.2, -0.3]))
