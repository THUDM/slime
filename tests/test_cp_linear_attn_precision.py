#!/usr/bin/env python3
"""
Test CP linear attention implementation for correctness.

Verifies that CPLinearAttnFunction produces identical forward outputs and
backward gradients compared to a non-CP reference, for different CP sizes.

Run with:
    torchrun --nproc_per_node=2 tests/test_cp_linear_attn_precision.py
    torchrun --nproc_per_node=4 tests/test_cp_linear_attn_precision.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.distributed as dist
import torch.nn as nn

from slime_plugins.models.qwen3_5 import CPLinearAttnFunction, _cp_all_gather_zigzag, _cp_slice_zigzag


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def test_gather_slice_roundtrip():
    """Test that slice -> all_gather reconstructs the original tensor exactly."""
    rank = dist.get_rank()
    cp_size = dist.get_world_size()
    cp_group = dist.group.WORLD

    torch.manual_seed(42)
    seq_len = 128
    hidden_dim = 32
    full_tensor = torch.randn(seq_len, 1, hidden_dim, device="cuda")
    dist.broadcast(full_tensor, src=0)

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int64, device="cuda")

    local_tensor = _cp_slice_zigzag(full_tensor, cu_seqlens, rank, cp_size)
    assert local_tensor.shape[0] == seq_len // cp_size

    reconstructed = _cp_all_gather_zigzag(local_tensor, cu_seqlens, cp_group, cp_size)
    max_diff = (reconstructed - full_tensor).abs().max().item()
    log(rank, f"[roundtrip single-seq] max diff = {max_diff}")
    assert max_diff == 0.0, f"Roundtrip failed: {max_diff}"


def test_gather_slice_multi_seq():
    """Test roundtrip with multiple packed sequences."""
    rank = dist.get_rank()
    cp_size = dist.get_world_size()
    cp_group = dist.group.WORLD

    torch.manual_seed(42)
    # Two sequences: 64 and 128 tokens
    seq_lens = [64, 128]
    total_len = sum(seq_lens)
    hidden_dim = 32
    full_tensor = torch.randn(total_len, 1, hidden_dim, device="cuda")
    dist.broadcast(full_tensor, src=0)

    cu_seqlens = torch.tensor([0, 64, 192], dtype=torch.int64, device="cuda")

    local_tensor = _cp_slice_zigzag(full_tensor, cu_seqlens, rank, cp_size)
    reconstructed = _cp_all_gather_zigzag(local_tensor, cu_seqlens, cp_group, cp_size)
    max_diff = (reconstructed - full_tensor).abs().max().item()
    log(rank, f"[roundtrip multi-seq] max diff = {max_diff}")
    assert max_diff == 0.0, f"Multi-seq roundtrip failed: {max_diff}"


def test_cp_forward_backward():
    """Test that CP forward/backward match non-CP reference exactly."""
    rank = dist.get_rank()
    cp_size = dist.get_world_size()
    cp_group = dist.group.WORLD

    torch.manual_seed(42)
    hidden_dim = 64
    model = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
    ).cuda()

    # Ensure all ranks have identical model
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    seq_len = 32 * cp_size  # divisible by 2 * cp_size
    full_hidden = torch.randn(seq_len, 1, hidden_dim, device="cuda")
    dist.broadcast(full_hidden, src=0)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int64, device="cuda")

    class FakeParams:
        cu_seqlens_q = cu_seqlens

    packed_params = FakeParams()

    class FakeModule(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def hf_forward(self, hidden_states, packed_seq_params):
            return self.net(hidden_states)

    module = FakeModule(model)

    # --- Reference: full forward/backward (no CP) ---
    model.zero_grad()
    full_ref = full_hidden.clone().detach().requires_grad_(True)
    ref_out = model(full_ref.permute(1, 0, 2)).permute(1, 0, 2)
    ref_out.sum().backward()
    ref_output = ref_out.detach().clone()
    ref_grad = full_ref.grad.clone()
    ref_pgrads = [p.grad.clone() for p in model.parameters()]

    # --- CP: forward/backward ---
    model.zero_grad()
    local_hidden = _cp_slice_zigzag(full_hidden, cu_seqlens, rank, cp_size).clone().detach().requires_grad_(True)

    local_output = CPLinearAttnFunction.apply(local_hidden, cu_seqlens, module, packed_params, cp_group, cp_size, rank)
    local_output.sum().backward()

    # Gather output and compare
    gathered_out = _cp_all_gather_zigzag(local_output.detach(), cu_seqlens, cp_group, cp_size)
    out_diff = (gathered_out - ref_output).abs().max().item()
    log(rank, f"[CP fwd] output diff = {out_diff}")
    assert out_diff < 5e-5, f"Output mismatch: {out_diff}"

    # Gather hidden grad and compare
    gathered_grad = _cp_all_gather_zigzag(local_hidden.grad.detach(), cu_seqlens, cp_group, cp_size)
    grad_diff = (gathered_grad - ref_grad).abs().max().item()
    log(rank, f"[CP bwd] hidden grad diff = {grad_diff}")
    assert grad_diff < 5e-5, f"Hidden grad mismatch: {grad_diff}"

    # Compare param grads
    for i, p in enumerate(model.parameters()):
        pdiff = (p.grad - ref_pgrads[i]).abs().max().item()
        log(rank, f"[CP bwd] param {i} grad diff = {pdiff}")
        assert pdiff < 5e-5, f"Param grad {i} mismatch: {pdiff}"


def test_cp_multi_seq_forward_backward():
    """Test CP with multiple packed sequences."""
    rank = dist.get_rank()
    cp_size = dist.get_world_size()
    cp_group = dist.group.WORLD

    torch.manual_seed(42)
    hidden_dim = 64
    model = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
    ).cuda()

    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # Two sequences, both divisible by 2 * cp_size
    chunk = 2 * cp_size
    s1, s2 = 4 * chunk, 6 * chunk
    total_len = s1 + s2
    full_hidden = torch.randn(total_len, 1, hidden_dim, device="cuda")
    dist.broadcast(full_hidden, src=0)
    cu_seqlens = torch.tensor([0, s1, s1 + s2], dtype=torch.int64, device="cuda")

    class FakeParams:
        cu_seqlens_q = cu_seqlens

    class FakeModule(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def hf_forward(self, hidden_states, packed_seq_params):
            return self.net(hidden_states)

    module = FakeModule(model)

    # Reference
    model.zero_grad()
    full_ref = full_hidden.clone().detach().requires_grad_(True)
    ref_out = model(full_ref.permute(1, 0, 2)).permute(1, 0, 2)
    ref_out.sum().backward()
    ref_output = ref_out.detach().clone()
    ref_grad = full_ref.grad.clone()

    # CP
    model.zero_grad()
    local_hidden = _cp_slice_zigzag(full_hidden, cu_seqlens, rank, cp_size).clone().detach().requires_grad_(True)
    local_output = CPLinearAttnFunction.apply(local_hidden, cu_seqlens, module, FakeParams(), cp_group, cp_size, rank)
    local_output.sum().backward()

    gathered_out = _cp_all_gather_zigzag(local_output.detach(), cu_seqlens, cp_group, cp_size)
    out_diff = (gathered_out - ref_output).abs().max().item()
    log(rank, f"[CP multi-seq fwd] diff = {out_diff}")
    assert out_diff < 5e-5, f"Multi-seq output mismatch: {out_diff}"

    gathered_grad = _cp_all_gather_zigzag(local_hidden.grad.detach(), cu_seqlens, cp_group, cp_size)
    grad_diff = (gathered_grad - ref_grad).abs().max().item()
    log(rank, f"[CP multi-seq bwd] diff = {grad_diff}")
    assert grad_diff < 5e-5, f"Multi-seq grad mismatch: {grad_diff}"


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    rank = dist.get_rank()

    log(rank, f"=== Testing CP with {dist.get_world_size()} ranks ===")

    test_gather_slice_roundtrip()
    test_gather_slice_multi_seq()
    test_cp_forward_backward()
    test_cp_multi_seq_forward_backward()

    log(rank, "=== ALL TESTS PASSED ===")
    dist.destroy_process_group()
