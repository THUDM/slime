"""CUDA correctness and workspace coverage for StatelessAdam."""

import math

import pytest
import torch

from slime.backends.megatron_utils.stateless_adam import StatelessAdam

NUM_GPUS = 1
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None, reason="NVIDIA CUDA is required"
)


def test_step_does_not_allocate_parameter_sized_workspace():
    param = torch.ones(4 * 1024 * 1024, device="cuda")
    param.grad = torch.full_like(param, 0.25)
    optimizer = StatelessAdam([param], lr=0.01, weight_decay=0.1)
    optimizer.step()  # Exclude compilation and lazy initialization.
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    optimizer.step()
    torch.cuda.synchronize()

    workspace = torch.cuda.max_memory_allocated() - allocated
    assert workspace < 1024 * 1024, f"Optimizer allocated {workspace} bytes of temporary tensors"


@pytest.mark.parametrize("adam_w_mode", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("decoupled_grad", [False, True])
def test_cuda_updates_match_reset_adam(adam_w_mode, maximize, decoupled_grad):
    torch.manual_seed(7)
    initial = torch.randn(8193, device="cuda")
    param = initial.clone()
    expected = initial.clone()
    kwargs = dict(lr=0.003, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1, maximize=maximize)
    optimizer = StatelessAdam([param], adam_w_mode=adam_w_mode, use_decoupled_grad=decoupled_grad, **kwargs)
    reference_cls = torch.optim.AdamW if adam_w_mode else torch.optim.Adam

    for _ in range(4):
        grad = torch.randn_like(param)
        grad[:4] = torch.tensor([0.0, 1e-9, -1e-9, 1.0], device="cuda")
        if decoupled_grad:
            param.decoupled_grad = grad.clone()
        else:
            param.grad = grad.clone()
        expected.grad = grad.clone()
        reference = reference_cls([expected], foreach=False, fused=False, **kwargs)
        reference.step()
        optimizer.step()
        torch.testing.assert_close(param, expected, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(grad, param.decoupled_grad if decoupled_grad else param.grad, rtol=0, atol=0)
        assert optimizer.state == {}


@pytest.mark.parametrize("adam_w_mode", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_cuda_without_bias_correction(adam_w_mode, weight_decay):
    param = torch.tensor([0.3, -0.7, 0.0, 1.0], device="cuda")
    expected = param.double()
    optimizer = StatelessAdam(
        [param],
        lr=0.02,
        betas=(0.8, 0.95),
        eps=1e-5,
        weight_decay=weight_decay,
        adam_w_mode=adam_w_mode,
        bias_correction=False,
    )
    for values in ([0.0, 0.4, -1e-8, -0.6], [0.2, -0.1, 0.3, 0.0]):
        param.grad = torch.tensor(values, device="cuda")
        grad = param.grad.double()
        if adam_w_mode:
            expected = expected * (1 - 0.02 * weight_decay)
        else:
            grad = grad + weight_decay * expected
        # Independent Adam moments, reset before each step, without correction.
        first_moment = 0.2 * grad
        second_moment = 0.05 * grad.square()
        expected = expected - 0.02 * first_moment / (second_moment.sqrt() + 1e-5)
        optimizer.step()
        torch.testing.assert_close(param.double(), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("adam_w_mode", [False, True])
@pytest.mark.parametrize("bias_correction", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_cuda_preserves_eager_rounding(adam_w_mode, bias_correction, maximize):
    torch.manual_seed(23)
    param = torch.randn(8193, device="cuda") * 0.01
    expected = param.clone()
    weight_decay = 0.13
    optimizer = StatelessAdam(
        [param],
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        adam_w_mode=adam_w_mode,
        bias_correction=bias_correction,
        maximize=maximize,
    )
    for lr in (0.003, 1e-5, 0.001):
        optimizer.param_groups[0]["lr"] = lr
        param.grad = torch.randn_like(param) * 1e-6
        # Exercise cancellation in coupled decay, where a rounding difference
        # can change the sign of the normalized gradient.
        if not adam_w_mode:
            param.grad[:1024] = expected[:1024] * (weight_decay if maximize else -weight_decay)
        grad = param.grad.neg() if maximize else param.grad
        if adam_w_mode:
            expected.mul_(1.0 - lr * weight_decay)
        else:
            grad = grad.add(expected, alpha=weight_decay)
        numerator_scale = 1.0 if bias_correction else 0.2
        denominator_scale = 1.0 if bias_correction else math.sqrt(1.0 - 0.95)
        denom = grad.abs().mul(denominator_scale).add_(1e-8)
        expected.addcdiv_(grad, denom, value=-lr * numerator_scale)

        optimizer.step()

        torch.testing.assert_close(param, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("strided", ["parameter", "gradient", "neither"])
def test_cuda_fallback_preserves_layout_and_updates(dtype, strided):
    storage = torch.ones(7, 6, device="cuda", dtype=dtype)
    param = storage[:, ::2] if strided == "parameter" else storage[:, :3].contiguous()
    grad_storage = torch.full((7, 6), 0.5, device="cuda", dtype=dtype)
    param.grad = grad_storage[:, ::2] if strided == "gradient" else grad_storage[:, :3].contiguous()
    optimizer = StatelessAdam([param], lr=0.125, eps=0, weight_decay=0)
    optimizer.step()
    torch.testing.assert_close(param, torch.full_like(param, 0.875), rtol=0, atol=0)
    if strided == "parameter":
        torch.testing.assert_close(storage[:, 1::2], torch.ones_like(param), rtol=0, atol=0)
    torch.testing.assert_close(grad_storage, torch.full_like(grad_storage, 0.5), rtol=0, atol=0)


def test_cuda_parameter_groups_closure_and_missing_gradients():
    param = torch.tensor([1.0], device="cuda", requires_grad=True)
    untouched = torch.tensor([2.0], device="cuda")
    empty = torch.empty(0, device="cuda")
    empty.grad = torch.empty_like(empty)
    optimizer = StatelessAdam([{"params": [param], "lr": 0.125}, {"params": [untouched, empty]}], eps=0)

    def closure():
        loss = param.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.item() == 1.0
    torch.testing.assert_close(param, torch.tensor([0.875], device="cuda"), rtol=0, atol=0)
    torch.testing.assert_close(untouched, torch.tensor([2.0], device="cuda"), rtol=0, atol=0)
    optimizer.param_groups[0]["lr"] = 0.25
    optimizer.step()
    torch.testing.assert_close(param, torch.tensor([0.625], device="cuda"), rtol=0, atol=0)
    assert optimizer.state_dict()["state"] == {}


def test_cuda_update_uses_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        param = torch.ones(8193, device="cuda")
        param.grad = torch.full_like(param, 0.5)
        optimizer = StatelessAdam([param], lr=0.125, eps=0)
        optimizer.step()
    stream.synchronize()
    torch.testing.assert_close(param, torch.full_like(param, 0.875), rtol=0, atol=0)


def test_cuda_decoupled_gradient_broadcast_uses_eager_fallback():
    param = torch.ones(2, 3, device="cuda")
    grad_storage = torch.tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]], device="cuda")
    param.decoupled_grad = grad_storage[:1]
    optimizer = StatelessAdam([param], lr=0.125, eps=0, use_decoupled_grad=True)
    optimizer.step()
    torch.testing.assert_close(param, torch.full_like(param, 0.875), rtol=0, atol=0)
    torch.testing.assert_close(param.decoupled_grad, torch.full_like(param.decoupled_grad, 0.5), rtol=0, atol=0)


def test_cuda_update_invalidates_saved_autograd_parameters():
    param = torch.ones(3, device="cuda", requires_grad=True)
    loss = param.square().sum()
    loss.backward(retain_graph=True)
    StatelessAdam([param], lr=0.1).step()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
