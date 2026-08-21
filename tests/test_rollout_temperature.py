"""CPU tests for SGLang rollout-temperature log-prob semantics."""

from __future__ import annotations

import sys
import types
from argparse import Namespace
from collections.abc import Iterator

import pytest
import torch


NUM_GPUS = 0


def _stub_megatron(monkeypatch) -> None:
    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_stub
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)


def _import_loss(monkeypatch):
    previous_loss = sys.modules.pop("slime.backends.megatron_utils.loss", None)
    previous_cp_utils = sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
    _stub_megatron(monkeypatch)
    from slime.backends.megatron_utils import loss as loss_module

    return loss_module, previous_loss, previous_cp_utils


def _restore_loss(previous_loss, previous_cp_utils) -> None:
    if previous_loss is None:
        sys.modules.pop("slime.backends.megatron_utils.loss", None)
    else:
        sys.modules["slime.backends.megatron_utils.loss"] = previous_loss
    if previous_cp_utils is None:
        sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
    else:
        sys.modules["slime.backends.megatron_utils.cp_utils"] = previous_cp_utils


@pytest.fixture
def loss_module(monkeypatch) -> Iterator[types.ModuleType]:
    module, previous_loss, previous_cp_utils = _import_loss(monkeypatch)
    try:
        yield module
    finally:
        _restore_loss(previous_loss, previous_cp_utils)


def test_apply_rollout_temperature_matches_sglang_logprob_semantics(loss_module):
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    torch.testing.assert_close(loss_module.apply_rollout_temperature(logits, 0.0), logits)
    torch.testing.assert_close(loss_module.apply_rollout_temperature(logits, 1.0), logits)
    torch.testing.assert_close(
        loss_module.apply_rollout_temperature(logits, 0.5),
        logits / 0.5,
    )

    with pytest.raises(ValueError, match="temperature must be >= 0"):
        loss_module.apply_rollout_temperature(logits, -0.1)


def test_recomputed_logprobs_follow_sglang_temperature_convention(loss_module):
    # Packed prompt+response: logits[t] predicts tokens[t+1]. The chosen
    # response token is vocab index 2, whose untempered model log-prob is
    # log_softmax([1, 2, 3])[2] < 0, not the greedy delta log q = 0.
    logits = torch.tensor([[[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]]], dtype=torch.float32)
    tokens = [torch.tensor([0, 2], dtype=torch.long)]
    target_logits = logits[0, 0]
    expected_t0 = torch.log_softmax(target_logits, dim=-1)[2]
    expected_t05 = torch.log_softmax(target_logits / 0.5, dim=-1)[2]
    expected_t1 = expected_t0

    def recompute(temperature: float) -> torch.Tensor:
        args = Namespace(
            rollout_temperature=temperature,
            allgather_cp=False,
            log_probs_chunk_size=-1,
            entropy_coef=0.0,
        )
        _, result = loss_module.get_log_probs_and_entropy(
            logits.clone(),
            args=args,
            unconcat_tokens=tokens,
            total_lengths=[2],
            response_lengths=[1],
        )
        return result["log_probs"][0]

    lp_t0 = recompute(0.0)
    lp_t05 = recompute(0.5)
    lp_t1 = recompute(1.0)

    torch.testing.assert_close(lp_t0, expected_t0.unsqueeze(0))
    torch.testing.assert_close(lp_t05, expected_t05.unsqueeze(0))
    torch.testing.assert_close(lp_t1, expected_t1.unsqueeze(0))
    assert lp_t0.item() < 0


def test_get_responses_leaves_greedy_logits_unscaled(loss_module):
    tokens = [torch.tensor([10, 11, 12, 13], dtype=torch.long)]
    policy = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]], dtype=torch.float32)
    chunk, _ = next(
        loss_module.get_responses(
            policy,
            args=Namespace(rollout_temperature=0.0, allgather_cp=False),
            unconcat_tokens=tokens,
            total_lengths=[4],
            response_lengths=[2],
        )
    )
    torch.testing.assert_close(chunk, torch.tensor([[0.3, 0.4], [0.5, 0.6]]))

    scaled, _ = next(
        loss_module.get_responses(
            policy,
            args=Namespace(rollout_temperature=0.5, allgather_cp=False),
            unconcat_tokens=tokens,
            total_lengths=[4],
            response_lengths=[2],
        )
    )
    torch.testing.assert_close(scaled, torch.tensor([[0.6, 0.8], [1.0, 1.2]]))


def test_greedy_rollout_skips_top_p_nucleus_replay(loss_module):
    batch = {
        "rollout_top_p_token_ids": [[0, 1]],
        "rollout_top_p_token_offsets": [[0, 2]],
    }
    greedy = Namespace(rollout_top_p=0.9, rollout_temperature=0.0)
    sampled = Namespace(rollout_top_p=0.9, rollout_temperature=0.5)
    disabled = Namespace(rollout_top_p=1.0, rollout_temperature=0.5)

    assert loss_module.get_rollout_top_p_logprob_kwargs(greedy, batch) == {}
    assert loss_module.get_rollout_top_p_logprob_kwargs(disabled, batch) == {}
    kwargs = loss_module.get_rollout_top_p_logprob_kwargs(sampled, batch)
    assert kwargs["top_p_token_ids"] == batch["rollout_top_p_token_ids"]
    assert kwargs["top_p_token_offsets"] == batch["rollout_top_p_token_offsets"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
