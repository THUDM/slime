import sys
import types
from argparse import Namespace

import pytest
import torch


NUM_GPUS = 0


@pytest.fixture
def loss_module(monkeypatch):
    previous_loss = sys.modules.pop("slime.backends.megatron_utils.loss", None)
    previous_cp_utils = sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
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
    try:
        from slime.backends.megatron_utils import loss as module

        yield module
    finally:
        if previous_loss is None:
            sys.modules.pop("slime.backends.megatron_utils.loss", None)
        else:
            sys.modules["slime.backends.megatron_utils.loss"] = previous_loss
        if previous_cp_utils is None:
            sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
        else:
            sys.modules["slime.backends.megatron_utils.cp_utils"] = previous_cp_utils


@pytest.mark.parametrize("temperature,scale", [(0.0, 1.0), (1.0, 1.0), (0.5, 0.5)])
def test_recomputed_logprobs_match_sglang_temperature(loss_module, temperature, scale):
    logits = torch.tensor([[[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]]], dtype=torch.float32)
    args = Namespace(
        rollout_temperature=temperature,
        allgather_cp=False,
        log_probs_chunk_size=-1,
        entropy_coef=0.0,
    )
    _, result = loss_module.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[torch.tensor([0, 2], dtype=torch.long)],
        total_lengths=[2],
        response_lengths=[1],
    )
    expected = torch.log_softmax(logits[0, 0] / scale, dim=-1)[2]
    torch.testing.assert_close(result["log_probs"][0], expected.unsqueeze(0))


def test_get_responses_does_not_scale_at_zero(loss_module):
    policy = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]], dtype=torch.float32)
    chunk, _ = next(
        loss_module.get_responses(
            policy,
            args=Namespace(rollout_temperature=0.0, allgather_cp=False),
            unconcat_tokens=[torch.tensor([10, 11, 12, 13], dtype=torch.long)],
            total_lengths=[4],
            response_lengths=[2],
        )
    )
    torch.testing.assert_close(chunk, torch.tensor([[0.3, 0.4], [0.5, 0.6]]))


def test_greedy_skips_top_p_replay(loss_module):
    batch = {
        "rollout_top_p_token_ids": [[0, 1]],
        "rollout_top_p_token_offsets": [[0, 2]],
    }
    greedy = loss_module.get_rollout_top_p_logprob_kwargs(
        Namespace(rollout_top_p=0.9, rollout_temperature=0.0),
        batch,
    )
    sampled = loss_module.get_rollout_top_p_logprob_kwargs(
        Namespace(rollout_top_p=0.9, rollout_temperature=0.5),
        batch,
    )
    assert greedy == {}
    assert sampled["top_p_token_ids"] == batch["rollout_top_p_token_ids"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
