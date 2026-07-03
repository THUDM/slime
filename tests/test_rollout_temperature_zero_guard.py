import sys
import types
from argparse import Namespace

import pytest
import torch


NUM_GPUS = 0


def _install_mpu_stub(monkeypatch):
    """Install a single-rank megatron.core.mpu stub and return the loss module."""
    sys.modules.pop("slime.backends.megatron_utils.loss", None)
    sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)

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

    import slime.backends.megatron_utils.loss as loss

    return loss


def test_get_responses_zero_temperature_stays_finite(monkeypatch):
    loss = _install_mpu_stub(monkeypatch)

    args = Namespace(rollout_temperature=0.0, allgather_cp=False, true_on_policy_mode=False)
    logits = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]], dtype=torch.float32)
    tokens = [torch.tensor([10, 11, 12, 13], dtype=torch.long)]

    chunks = [
        chunk
        for chunk, _ in loss.get_responses(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=[4],
            response_lengths=[2],
        )
    ]

    assert len(chunks) == 1
    out = chunks[0]
    torch.testing.assert_close(out, logits.squeeze(0)[1:3])


def test_get_log_probs_zero_temperature_stays_finite(monkeypatch):
    loss = _install_mpu_stub(monkeypatch)

    captured = {}

    def fake_calculate(scaled_logits, tokens, tp_group, **kwargs):
        captured["logits"] = scaled_logits
        T = scaled_logits.size(0)
        return scaled_logits.new_zeros((T, 1)), None

    monkeypatch.setattr(loss, "calculate_log_probs_and_entropy", fake_calculate)

    args = Namespace(rollout_temperature=0.0, allgather_cp=False, log_probs_chunk_size=-1)
    logits = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]], dtype=torch.float32)
    tokens = [torch.tensor([10, 11, 12, 13], dtype=torch.long)]

    loss.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=tokens,
        total_lengths=[4],
        response_lengths=[2],
    )

    scaled = captured["logits"]
    torch.testing.assert_close(scaled, logits.squeeze(0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
