from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_NAME = "examples.geo3k_vlm_multi_turn.rollout"
pytestmark = pytest.mark.unit


class _FakeTokenizer:
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [100 + idx for idx, _ in enumerate(str(text).split())]


class _FakeSample(SimpleNamespace):
    def __init__(self, *, prompt="a b c", tokens=None, loss_mask=None, rollout_log_probs=None):
        super().__init__(
            prompt=prompt,
            tokens=list(tokens or []),
            multimodal_inputs=None,
            multimodal_train_inputs=None,
            loss_mask=loss_mask,
            rollout_log_probs=rollout_log_probs,
            response_length=0,
        )


@pytest.fixture()
def rollout(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    fake_torch.cat = lambda values, dim=0: list(values)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_sglang_rollout = types.ModuleType("slime.rollout.sglang_rollout")
    fake_sglang_rollout.GenerateState = object
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", fake_sglang_rollout)

    fake_http = types.ModuleType("slime.utils.http_utils")
    fake_http.post = None
    monkeypatch.setitem(sys.modules, "slime.utils.http_utils", fake_http)

    fake_processing = types.ModuleType("slime.utils.processing_utils")
    fake_processing.encode_image_for_rollout_engine = lambda image: image
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", fake_processing)

    fake_types = types.ModuleType("slime.utils.types")
    fake_types.Sample = object
    monkeypatch.setitem(sys.modules, "slime.utils.types", fake_types)

    sys.modules.pop(MODULE_NAME, None)
    try:
        yield importlib.import_module(MODULE_NAME)
    finally:
        sys.modules.pop(MODULE_NAME, None)


def _start_state(rollout, sample, *, context_len=None, max_new_tokens=5):
    state = SimpleNamespace(processor=None, tokenizer=_FakeTokenizer())
    return rollout._prepare_start_state(
        sample,
        state,
        Namespace(rollout_max_context_len=context_len),
        {"max_new_tokens": max_new_tokens},
    )


def test_geo3k_multi_turn_response_budget_does_not_charge_prompt_tokens(rollout):
    sample = _FakeSample(prompt="a b c")

    _image_data, response_tokens, budget, _mm_buffer = _start_state(rollout, sample)

    assert response_tokens == []
    assert sample.tokens == [100, 101, 102]
    assert budget == 5


def test_geo3k_multi_turn_response_budget_counts_existing_response_only(rollout):
    sample = _FakeSample(
        prompt="a b c",
        tokens=[100, 101, 102, 201, 202],
        loss_mask=[1, 1],
        rollout_log_probs=[-0.1, -0.2],
    )

    _image_data, response_tokens, budget, _mm_buffer = _start_state(rollout, sample)

    assert response_tokens == [201, 202]
    assert sample.response_length == 2
    assert budget == 3


def test_geo3k_multi_turn_budget_respects_context_and_response_limits(rollout):
    sample = _FakeSample(
        prompt="a b c",
        tokens=[100, 101, 102, 201, 202, 203, 204],
        loss_mask=[1, 1, 1, 1],
        rollout_log_probs=[-0.1, -0.2, -0.3, -0.4],
    )

    _image_data, response_tokens, budget, _mm_buffer = _start_state(rollout, sample, context_len=10)

    assert response_tokens == [201, 202, 203, 204]
    assert budget == 1
