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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
