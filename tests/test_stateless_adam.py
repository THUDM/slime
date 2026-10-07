import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from slime.backends.megatron_utils.stateless_adam import StatelessAdam

NUM_GPUS = 0


def _run_with_reinitialized_adam(param, grads, *, adam_w_mode):
    param = param.clone().detach()
    optimizer_cls = torch.optim.AdamW if adam_w_mode else torch.optim.Adam
    for grad in grads:
        param = param.detach().requires_grad_(True)
        optimizer = optimizer_cls(
            [param],
            lr=0.03,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.1,
        )
        param.grad = grad.clone()
        optimizer.step()
        param = param.detach()
    return param


@pytest.mark.unit
@pytest.mark.parametrize("adam_w_mode", [True, False])
def test_stateless_adam_matches_reinitialized_adam_each_step(adam_w_mode):
    torch.manual_seed(0)
    initial_param = torch.randn(8, dtype=torch.float64)
    grads = [torch.randn_like(initial_param) for _ in range(4)]
    param = initial_param.clone()
    optimizer = StatelessAdam(
        [param],
        lr=0.03,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.1,
        adam_w_mode=adam_w_mode,
    )

    for grad in grads:
        param.grad = grad.clone()
        optimizer.step()
        optimizer.zero_grad()

    expected = _run_with_reinitialized_adam(initial_param, grads, adam_w_mode=adam_w_mode)
    torch.testing.assert_close(param, expected)


@pytest.mark.unit
def test_stateless_adam_does_not_persist_moment_tensors():
    param = torch.tensor([1.0, -2.0])
    optimizer = StatelessAdam([param])

    assert optimizer.state == {}
    assert optimizer.state_dict()["state"] == {}


@pytest.fixture
def checkpoint_module(tmp_path, monkeypatch):
    # Only Megatron's model/collective boundary is replaced. The Slime adapter
    # writes real checkpoint files, and the optimizer/scheduler below are real.
    args = SimpleNamespace(
        save=str(tmp_path), load=str(tmp_path), use_stateless_adam=True, finetune=False, ckpt_step=0
    )

    def save_model(iteration, *a, **kw):
        (tmp_path / f"iter_{iteration:07d}").mkdir()
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    native = SimpleNamespace(save_checkpoint=Mock(side_effect=save_model), load_checkpoint=Mock(return_value=(0, 0)))
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", native)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", SimpleNamespace(get_args=lambda: args))
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/checkpoint.py"
    spec = importlib.util.spec_from_file_location("stateless_checkpoint_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    return module, args, native


@pytest.mark.parametrize("step", [0, 7])
def test_stateless_checkpoint_restores_scheduler_progress(checkpoint_module, step):
    module, args, native = checkpoint_module
    args.ckpt_step = step
    param = torch.tensor([1.0])
    optimizer = StatelessAdam([param], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda count: 0.8**count)
    for _ in range(3):
        param.grad = torch.ones_like(param)
        optimizer.step()
        scheduler.step()
    module.save_checkpoint(step, [], optimizer, scheduler, num_floating_point_operations_so_far=0)
    expected = scheduler.state_dict()
    param.grad = torch.ones_like(param)
    optimizer.step()
    scheduler.step()
    expected_next_lr = scheduler.get_last_lr()

    restored_optimizer = StatelessAdam([torch.tensor([1.0])], lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda count: 0.8**count)
    native.load_checkpoint.return_value = (step, 0)
    assert module.load_checkpoint([], restored_optimizer, restored_scheduler, {}) == (step, 0)
    assert args.no_load_optim and restored_optimizer.state == {}
    assert restored_scheduler.state_dict() == expected
    restored_optimizer.step()
    restored_scheduler.step()
    assert restored_scheduler.get_last_lr() == expected_next_lr


def test_stateless_resume_rejects_missing_scheduler_but_finetune_can_reset(checkpoint_module):
    module, args, native = checkpoint_module
    native.save_checkpoint(0)
    scheduler = Mock()
    with pytest.raises(FileNotFoundError, match="opt_param_scheduler"):
        module.load_checkpoint([], Mock(), scheduler, {})
    scheduler.load_state_dict.assert_not_called()
    args.finetune = True
    assert module.load_checkpoint([], Mock(), scheduler, {}) == (0, 0)
    scheduler.load_state_dict.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
