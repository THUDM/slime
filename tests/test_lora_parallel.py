"""Two-process CPU/Gloo LoRA forward, input-gradient and factor-gradient parity."""

from __future__ import annotations

import sys
import types
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from _lora_fakes import (
    FakeColumnParallelLinear,
    FakeFusedLayerNormColumnLinear,
    FakeRowParallelLinear,
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
    make_config,
)

from slime.backends.megatron_utils.lora import layers

NUM_GPUS = 0


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        dist.all_reduce(grad)
        return grad


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, reduce_grad):
        ctx.reduce_grad = reduce_grad
        shards = [torch.empty_like(x) for _ in range(2)]
        dist.all_gather(shards, x.contiguous())
        return torch.cat(shards, dim=0)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        if ctx.reduce_grad:
            dist.all_reduce(grad)
        return grad.chunk(2, dim=0)[dist.get_rank()].contiguous(), None


class _ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous().clone()
        dist.all_reduce(x)
        return x.chunk(2, dim=0)[dist.get_rank()].contiguous()

    @staticmethod
    def backward(ctx, grad):
        shards = [torch.empty_like(grad) for _ in range(2)]
        dist.all_gather(shards, grad.contiguous())
        return torch.cat(shards, dim=0)


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous().clone()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


def _check_network(sequence_parallel, fused_norm=False, implementation="cpu"):
    """Compare two column/tanh/row blocks with an unsharded differentiable reference."""
    rank = dist.get_rank()
    device = torch.device("cpu" if implementation == "cpu" else f"cuda:{rank}")
    dtype = torch.float64 if implementation == "cpu" else torch.float32
    torch.manual_seed(137)
    hidden, ffn, r = 16, 32, 4
    config = make_config(lora_rank=r, lora_alpha=8)
    x_full = torch.randn(4, 1, hidden, device=device, dtype=dtype)
    x_ref = x_full.clone().requires_grad_()
    x = (x_full.chunk(2, dim=0)[rank] if sequence_parallel else x_full).clone().requires_grad_()
    target = torch.randn_like(x_full)
    reference, actual = x_ref, x
    parameters = []
    modules = []
    for block in range(2):
        if implementation == "cpu":
            col = (FakeFusedLayerNormColumnLinear if fused_norm else FakeColumnParallelLinear)(hidden, ffn // 2)
            col.sequence_parallel = sequence_parallel
            row = FakeRowParallelLinear(ffn // 2, hidden, sequence_parallel=sequence_parallel)
        else:
            from megatron.core.transformer.transformer_config import TransformerConfig

            mconfig = TransformerConfig(
                num_layers=2,
                hidden_size=hidden,
                num_attention_heads=4,
                tensor_model_parallel_size=dist.get_world_size(),
                sequence_parallel=sequence_parallel,
                use_cpu_initialization=False,
                params_dtype=dtype,
                normalization="RMSNorm",
                gradient_accumulation_fusion=False,
            )
            if implementation == "te":
                from megatron.core.extensions.transformer_engine import (
                    TELayerNormColumnParallelLinear,
                    TERowParallelLinear,
                )

                col = TELayerNormColumnParallelLinear(
                    hidden,
                    ffn,
                    config=mconfig,
                    init_method=torch.nn.init.normal_,
                    gather_output=False,
                    bias=False,
                    skip_bias_add=False,
                    is_expert=False,
                )
                row_cls = TERowParallelLinear
            else:
                from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

                col = ColumnParallelLinear(
                    hidden,
                    ffn,
                    config=mconfig,
                    init_method=torch.nn.init.normal_,
                    gather_output=False,
                    bias=False,
                )
                row_cls = RowParallelLinear
            row = row_cls(
                ffn,
                hidden,
                config=mconfig,
                init_method=torch.nn.init.normal_,
                input_is_parallel=True,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
        col, row = col.to(device=device, dtype=dtype), row.to(device=device, dtype=dtype)
        for module, suffix in [(col, "linear_fc1"), (row, "linear_fc2")]:
            with torch.no_grad():
                module.weight.zero_()
            module.requires_grad_(False)
            spec = layers.describe_lora_target(f"decoder.layers.{block}.mlp.{suffix}", module, config)
            layers.attach_lora_adapter(module, spec, config)
            modules.append(module)
        # Generate one reference on all ranks independently of backend RNG consumption.
        torch.manual_seed(700 + block)
        a1 = (torch.randn(r, hidden, device=device, dtype=dtype) * 0.2).requires_grad_()
        b1 = (torch.randn(ffn, r, device=device, dtype=dtype) * 0.2).requires_grad_()
        a2 = (torch.randn(r, ffn, device=device, dtype=dtype) * 0.2).requires_grad_()
        b2 = (torch.randn(hidden, r, device=device, dtype=dtype) * 0.2).requires_grad_()
        shards = [(col.lora_A, a1, 1), (col.lora_B, b1, 0), (row.lora_A, a2, 1), (row.lora_B, b2, 1)]
        with torch.no_grad():
            for local, full, axis in shards:
                local.copy_(full.chunk(2, dim=axis)[rank])
        parameters.extend(shards)
        normed = reference
        if fused_norm:
            normed = reference.float() * torch.rsqrt(reference.float().square().mean(-1, keepdim=True) + col.eps)
            normed = (normed * col.layer_norm_weight.float()).to(dtype)
        ref_hidden = torch.tanh(F.linear(F.linear(normed, a1), b1) * config.scale)
        reference = reference + F.linear(F.linear(ref_hidden, a2), b2) * config.scale
        if implementation == "cpu":
            out = layers.compute_lora_delta(col, actual)
            out = layers.compute_lora_delta(row, torch.tanh(out))
        else:
            out = row(torch.tanh(col(actual)[0]))[0]
        actual = actual + out

    ref_local = reference.chunk(2, dim=0)[rank] if sequence_parallel else reference
    atol = 2e-6 if implementation == "cpu" else 2e-4
    torch.testing.assert_close(actual, ref_local, rtol=2e-4, atol=atol)
    (actual * (target.chunk(2, dim=0)[rank] if sequence_parallel else target)).sum().backward()
    (reference * target).sum().backward()
    expected_dx = x_ref.grad.chunk(2, dim=0)[rank] if sequence_parallel else x_ref.grad
    torch.testing.assert_close(x.grad, expected_dx, rtol=2e-4, atol=atol)
    for local, full, axis in parameters:
        torch.testing.assert_close(local.grad, full.grad.chunk(2, dim=axis)[rank], rtol=2e-4, atol=atol)
    return modules


def _check_routed_expert_network(implementation="cpu"):
    """Verify the shared routed-expert adapter against an unsharded reference.

    Routed experts differ from dense layers in that the MoE token dispatcher owns
    all sequence-parallel communication: the activation reaching the experts
    already carries the full hidden width, and the expert output is reduced by the
    dispatcher. The adapter must therefore add neither an input gather nor an
    output reduction, only the expert-TP reduction of the low-rank intermediate.
    """
    rank = dist.get_rank()
    device = torch.device("cpu")
    dtype = torch.float64
    torch.manual_seed(137)
    hidden, ffn, r, tokens = 16, 32, 4, 6
    config = make_config(lora_rank=r, lora_alpha=8)

    a1 = (torch.randn(r, hidden, device=device, dtype=dtype) * 0.2).requires_grad_()
    b1 = (torch.randn(ffn, r, device=device, dtype=dtype) * 0.2).requires_grad_()
    a2 = (torch.randn(r, ffn, device=device, dtype=dtype) * 0.2).requires_grad_()
    b2 = (torch.randn(hidden, r, device=device, dtype=dtype) * 0.2).requires_grad_()
    x_full = torch.randn(tokens, hidden, device=device, dtype=dtype)
    target = torch.randn(tokens, hidden, device=device, dtype=dtype)

    # Unsharded reference: fc1 -> tanh -> fc2, all in full width.
    x_ref = x_full.clone().requires_grad_()
    ref_hidden = torch.tanh(F.linear(F.linear(x_ref, a1), b1) * config.scale)
    reference = F.linear(F.linear(ref_hidden, a2), b2) * config.scale

    fc1 = TEColumnParallelGroupedLinear(2, hidden, ffn // 2).to(device=device, dtype=dtype)
    fc2 = TERowParallelGroupedLinear(2, ffn // 2, hidden).to(device=device, dtype=dtype)
    modules = []
    for module, suffix in [(fc1, "linear_fc1"), (fc2, "linear_fc2")]:
        with torch.no_grad():
            for index in range(module.num_gemms):
                getattr(module, f"weight{index}").zero_()
        module.requires_grad_(False)
        spec = layers.describe_lora_target(f"decoder.layers.0.mlp.experts.{suffix}", module, config)
        layers.attach_lora_adapter(module, spec, config)
        modules.append(module)

    # Shard the reference factors exactly as the implementation expects: A of the
    # column layer is replicated (full hidden width), everything else is sharded.
    shards = [(fc1.lora_A, a1, None), (fc1.lora_B, b1, 0), (fc2.lora_A, a2, 1), (fc2.lora_B, b2, 1)]
    with torch.no_grad():
        for local, full, axis in shards:
            local.copy_(full if axis is None else full.chunk(2, dim=axis)[rank])

    x = x_full.clone().requires_grad_()
    out = layers.compute_lora_delta(fc1, x)
    out = layers.compute_lora_delta(fc2, torch.tanh(out))
    # The row path leaves a partial sum per rank; the dispatcher's reduce-scatter
    # is what combines them, so sum here to compare against the full reference.
    actual = _Reduce.apply(out)

    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-6)

    (actual * target).sum().backward()
    (reference * target).sum().backward()
    # ``x`` is replicated across expert-TP ranks (the dispatcher gathers along the
    # token axis), so each rank legitimately holds a partial input gradient; the
    # dispatcher's reduce-scatter backward is what sums them.
    summed_input_grad = x.grad.clone()
    dist.all_reduce(summed_input_grad)
    torch.testing.assert_close(summed_input_grad, x_ref.grad, rtol=2e-4, atol=2e-6)
    for local, full, axis in shards:
        expected = full.grad if axis is None else full.grad.chunk(2, dim=axis)[rank]
        torch.testing.assert_close(local.grad, expected, rtol=2e-4, atol=2e-6)
    return modules


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    mappings = types.ModuleType("megatron.core.tensor_parallel.mappings")
    mappings.copy_to_tensor_model_parallel_region = _Copy.apply
    mappings.gather_from_sequence_parallel_region = lambda x, tensor_parallel_output_grad=True: _Gather.apply(
        x, tensor_parallel_output_grad
    )
    mappings.reduce_scatter_to_sequence_parallel_region = _ReduceScatter.apply
    mappings.reduce_from_tensor_model_parallel_region = _Reduce.apply
    try:
        with (
            patch.dict(sys.modules, {mappings.__name__: mappings}),
            patch.object(layers, "tensor_model_parallel_world_size", lambda: 2),
            patch.object(layers, "tensor_model_parallel_rank", lambda: rank),
            patch.object(layers, "tensor_model_parallel_group", lambda: dist.group.WORLD),
            patch.object(layers, "expert_tensor_parallel_world_size", lambda: 2),
            patch.object(layers, "expert_tensor_parallel_rank", lambda: rank),
            patch.object(layers, "expert_tensor_parallel_group", lambda: dist.group.WORLD),
            # Expert parallelism is exercised separately; here EP=1 so the shared
            # adapter needs no cross-EP gradient summation.
            patch.object(layers, "expert_model_parallel_world_size", lambda: 1),
        ):
            for sequence_parallel in [False, True]:
                for fused_norm in [False, True]:
                    _check_network(sequence_parallel, fused_norm)
            _check_routed_expert_network()
    finally:
        dist.destroy_process_group()


def _expert_parallel_worker(rank, rendezvous):
    """EP=2, expert-TP=1: one adapter shared by the local experts of both ranks.

    Each expert-parallel rank only routes its own tokens, so the gradient of the
    shared adapter is the sum over the expert-parallel group. Anything less would
    train each replica on a different objective and let them drift apart.
    """
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        with (
            patch.object(layers, "expert_tensor_parallel_world_size", lambda: 1),
            patch.object(layers, "expert_tensor_parallel_rank", lambda: 0),
            patch.object(layers, "expert_model_parallel_world_size", lambda: 2),
            patch.object(layers, "expert_model_parallel_group", lambda: dist.group.WORLD),
        ):
            dtype = torch.float64
            hidden, ffn, r, tokens = 8, 12, 4, 5
            config = make_config(lora_rank=r, lora_alpha=8)
            torch.manual_seed(0)
            a = (torch.randn(r, hidden, dtype=dtype) * 0.2).requires_grad_()
            b = (torch.randn(ffn, r, dtype=dtype) * 0.2).requires_grad_()
            # Distinct tokens per EP rank: this is the whole point of the test.
            torch.manual_seed(100 + rank)
            x_local = torch.randn(tokens, hidden, dtype=dtype)
            torch.manual_seed(7)
            upstream = torch.randn(tokens, ffn, dtype=dtype)

            fc1 = TEColumnParallelGroupedLinear(2, hidden, ffn).to(dtype=dtype)
            with torch.no_grad():
                for index in range(fc1.num_gemms):
                    getattr(fc1, f"weight{index}").zero_()
            fc1.requires_grad_(False)
            spec = layers.describe_lora_target("decoder.layers.0.mlp.experts.linear_fc1", fc1, config)
            layers.attach_lora_adapter(fc1, spec, config)
            with torch.no_grad():
                fc1.lora_A.copy_(a)
                fc1.lora_B.copy_(b)

            layers.compute_lora_delta(fc1, x_local).backward(upstream)

            # Reference: the shared adapter sees every EP rank's tokens.
            all_x = [torch.empty_like(x_local) for _ in range(2)]
            dist.all_gather(all_x, x_local.contiguous())
            loss = sum((F.linear(F.linear(xi, a), b) * config.scale * upstream).sum() for xi in all_x)
            loss.backward()

            torch.testing.assert_close(fc1.lora_A.grad, a.grad, rtol=2e-4, atol=2e-6)
            torch.testing.assert_close(fc1.lora_B.grad, b.grad, rtol=2e-4, atol=2e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
def test_two_rank_lora_forward_and_backward(tmp_path):
    torch.multiprocessing.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


@pytest.mark.integration
def test_expert_parallel_shared_adapter_sums_gradients(tmp_path):
    torch.multiprocessing.spawn(
        _expert_parallel_worker, args=((tmp_path / "rendezvous-ep").as_uri(),), nprocs=2, join=True
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
