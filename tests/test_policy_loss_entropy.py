import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch


NUM_GPUS = 0

_MISSING = object()
_MODULES_TO_RESTORE = (
    "slime.backends.megatron_utils.loss",
    "slime.backends.megatron_utils.cp_utils",
    "megatron",
    "megatron.core",
    "megatron.core.mpu",
)


@pytest.fixture
def loss_module():
    previous_modules = {name: sys.modules.get(name, _MISSING) for name in _MODULES_TO_RESTORE}
    for name in ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils"):
        sys.modules.pop(name, None)

    mpu_mod = types.ModuleType("megatron.core.mpu")
    mpu_mod.get_context_parallel_world_size = lambda: 1
    mpu_mod.get_context_parallel_rank = lambda: 0
    mpu_mod.get_tensor_model_parallel_group = lambda: None

    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_mod
    megatron_mod = types.ModuleType("megatron")
    megatron_mod.core = core_mod

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.mpu"] = mpu_mod

    try:
        yield importlib.import_module("slime.backends.megatron_utils.loss")
    finally:
        for name, module in previous_modules.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _make_args(entropy_coef: float) -> Namespace:
    return Namespace(
        entropy_coef=entropy_coef,
        use_rollout_logprobs=False,
        use_opsm=False,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        use_kl_loss=False,
    )


def _make_batch() -> dict:
    return {
        "advantages": [torch.ones(2)],
        "unconcat_tokens": [torch.tensor([1, 2, 3], dtype=torch.long)],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
    }


def _mean_reducer(x: torch.Tensor) -> torch.Tensor:
    return x.mean()


def test_policy_loss_skips_entropy_when_coef_is_zero(loss_module, monkeypatch):
    logits = torch.randn(1, 3, 4, dtype=torch.float32, requires_grad=True)
    calls = []

    def fake_get_log_probs_and_entropy(logits_arg, *, with_entropy, **kwargs):
        calls.append(with_entropy)
        if with_entropy:
            raise AssertionError("policy_loss_function requested entropy with entropy_coef=0")
        return torch.empty((0,), device=logits_arg.device), {"log_probs": [logits_arg[0, 1:3, 0]]}

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)

    loss, metrics = loss_module.policy_loss_function(
        _make_args(entropy_coef=0.0),
        _make_batch(),
        logits,
        _mean_reducer,
    )

    assert calls == [False]
    assert metrics["entropy_loss"].shape == torch.Size([])
    assert metrics["entropy_loss"].item() == 0.0
    assert not metrics["entropy_loss"].requires_grad
    loss.backward()
    assert logits.grad is not None


def test_policy_loss_requests_entropy_when_coef_is_nonzero(loss_module, monkeypatch):
    logits = torch.randn(1, 3, 4, dtype=torch.float32, requires_grad=True)
    calls = []

    def fake_get_log_probs_and_entropy(logits_arg, *, with_entropy, **kwargs):
        calls.append(with_entropy)
        result = {"log_probs": [logits_arg[0, 1:3, 0]]}
        if with_entropy:
            result["entropy"] = [logits_arg[0, 1:3, 1]]
        return torch.empty((0,), device=logits_arg.device), result

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)

    loss, metrics = loss_module.policy_loss_function(
        _make_args(entropy_coef=0.5),
        _make_batch(),
        logits,
        _mean_reducer,
    )

    assert calls == [True]
    torch.testing.assert_close(metrics["entropy_loss"], logits.detach()[0, 1:3, 1].mean())
    loss.backward()
    torch.testing.assert_close(logits.grad[0, 1:3, 1], torch.full((2,), -0.25))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
