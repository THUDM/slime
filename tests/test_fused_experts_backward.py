"""
Test script to compare Triton backward implementation with Python backward implementation.

This test compares:
1. Triton implementation (from fused_experts.py) - uses invoke_fused_moe_backward_kernel
2. Python reference implementation (defined in this file) - uses pure PyTorch operations
"""

import pytest
import torch

# ============================================================================
# Python Reference Implementation (Pure PyTorch)
# ============================================================================


class GateUpProjFunctionPython(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        num_tokens, D_in = hidden_states.shape
        E, N, K = w1.shape
        assert D_in == K, f"hidden_states dim {D_in} != w1 dim {K}"

        topk = topk_ids.shape[1]

        # Output: (num_tokens * topk, N)
        intermediate_cache1 = torch.empty(
            (num_tokens * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Python implementation: iterate over tokens and their topk experts
        # For each token t and expert k:
        #   intermediate_cache1[t*topk + k] = hidden_states[t] @ w1[expert_id].T
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_ids[t, k].item()
                x_t = hidden_states[t]  # shape: (D_in,)
                W1_e = w1[expert_id]  # shape: (N, K)
                intermediate_cache1[t * topk + k] = x_t @ W1_e.T

        ctx.save_for_backward(hidden_states, w1, topk_weights, topk_ids)
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for GateUpProjFunction - Pure Python implementation.

        Forward: output = input @ w1 (without topk_weight multiplication)
        Backward:
            - grad_hidden_states = grad_output @ w1
            - grad_w1 = grad_output.T @ input (note: transposed)
            - grad_topk_weights = zeros (not needed in this stage)

        Args:
            grad_output: shape (num_tokens * topk, N)

        Returns:
            (grad_hidden_states, grad_w1, grad_topk_weights, None)
        """
        hidden_states, w1, topk_weights, topk_ids = ctx.saved_tensors
        topk = ctx.topk

        num_tokens, D_in = hidden_states.shape
        E, N, _ = w1.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_hidden_states = torch.zeros_like(hidden_states)
        # Use float32 for grad_w1 accumulation to avoid bfloat16 precision loss
        grad_w1 = torch.zeros(w1.shape, dtype=torch.float32, device=w1.device)
        # GateUpProj stage doesn't compute topk_weights gradient
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx * topk : end_chunk_idx * topk]

            # 1. Calculate grad_hidden_states: grad_output @ w1
            # For each token t and expert k:
            #   grad_hidden_states[t] += grad_output[t*topk+k] @ w1[expert_id]
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    grad_y_tk = curr_grad_output[t * topk + k]  # shape: (N,)
                    W1_e = w1[expert_id]  # shape: (N, D_in)
                    # grad_x: (N,) @ (N, D_in) -> (D_in,)
                    grad_hidden_states[begin_chunk_idx + t] += grad_y_tk @ W1_e

            # 2. Calculate grad_w1: input.T @ grad_output
            # For each token t and expert k:
            #   grad_w1[expert_id] += input[t].T @ grad_output[t*topk+k]
            # Which is: grad_w1[expert_id] += outer(grad_output[t*topk+k], input[t])
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    x_t = curr_hidden_states[t]  # shape: (D_in,)
                    grad_y_tk = curr_grad_output[t * topk + k]  # shape: (N,)
                    # grad_W1: outer(grad_y_tk, x_t) -> (N, D_in)
                    # Accumulate in float32
                    grad_w1[expert_id] += torch.outer(grad_y_tk, x_t).to(torch.float32)

        # Convert grad_w1 back to original dtype (bfloat16)
        grad_w1 = grad_w1.to(hidden_states.dtype)

        return grad_hidden_states, grad_w1, grad_topk_weights, None


class DownProjFunctionPython(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache2: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        total_tokens, intermediate_size = intermediate_cache2.shape
        topk = topk_ids.shape[1]
        num_tokens = total_tokens // topk
        E, hidden_size, K = w2.shape
        assert intermediate_size == K, f"intermediate_cache2 dim {intermediate_size} != w2 dim {K}"

        # Output: (num_tokens, topk, hidden_size)
        intermediate_cache3 = torch.empty(
            (num_tokens, topk, hidden_size),
            device=intermediate_cache2.device,
            dtype=intermediate_cache2.dtype,
        )

        # Python implementation: iterate over tokens and their topk experts
        # For each token t and expert k:
        #   intermediate_cache3[t, k] = topk_weights[t, k] * (intermediate_cache2[t*topk+k] @ w2[expert_id].T)
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_ids[t, k].item()
                x_tk = intermediate_cache2[t * topk + k]  # shape: (intermediate_size,)
                W2_e = w2[expert_id]  # shape: (hidden_size, intermediate_size)
                weight_tk = topk_weights[t, k]  # scalar

                intermediate_cache3[t, k] = weight_tk * (x_tk @ W2_e.T)

        ctx.save_for_backward(intermediate_cache2, w2, topk_weights, topk_ids)
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DownProjFunction - Pure Python implementation.

        Forward: output = topk_weights * (input @ w2) (with topk_weight multiplication)
        Backward:
            - grad_intermediate_cache2 = topk_weights * (grad_output @ w2)
            - grad_w2 = topk_weights * (grad_output.T @ intermediate_cache2)
            - grad_topk_weights = dot(grad_output, forward_output_before_weighting)

        Args:
            grad_output: shape (num_tokens, topk, hidden_size)

        Returns:
            (grad_intermediate_cache2, grad_w2, grad_topk_weights, None)
        """
        intermediate_cache2, w2, topk_weights, topk_ids = ctx.saved_tensors
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, hidden_size, intermediate_size = w2.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
        # Use float32 for grad_w2 accumulation to avoid bfloat16 precision loss
        grad_w2 = torch.zeros(w2.shape, dtype=torch.float32, device=w2.device)
        # Compute grad_topk_weights in DownProjFunction backward
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            # 1. Calculate grad_intermediate_cache2: topk_weights * (grad_output @ w2)
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    grad_y_tk = curr_grad_output[t, k]  # shape: (hidden_size,)
                    W2_e = w2[expert_id]  # shape: (hidden_size, intermediate_size)
                    weight_tk = curr_topk_weights[t, k]  # scalar

                    grad_intermediate_cache2[(begin_chunk_idx + t) * topk + k] = weight_tk * (grad_y_tk @ W2_e)

            # 2. Calculate grad_w2: topk_weights * (grad_output.T @ intermediate_cache2)
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    grad_y_tk = curr_grad_output[t, k]  # shape: (hidden_size,)
                    x_tk = curr_intermediate_cache2[t * topk + k]  # shape: (intermediate_size,)
                    weight_tk = curr_topk_weights[t, k]  # scalar

                    # Accumulate in float32
                    grad_w2[expert_id] += (weight_tk * torch.outer(grad_y_tk, x_tk)).to(torch.float32)

            # 3. Calculate grad_topk_weights: dot(grad_output, forward_output_before_weighting)
            for t in range(curr_num_tokens):
                for k in range(topk):
                    expert_id = curr_topk_ids[t, k].item()
                    grad_y_tk = curr_grad_output[t, k]  # shape: (hidden_size,)
                    x_tk = curr_intermediate_cache2[t * topk + k]  # shape: (intermediate_size,)
                    W2_e = w2[expert_id]  # shape: (hidden_size, intermediate_size)

                    # Compute forward output before weighting
                    forward_output_unweighted = x_tk @ W2_e.T  # shape: (hidden_size,)

                    # grad_topk_weights: dot product
                    grad_topk_weights[begin_chunk_idx + t, k] += torch.sum(grad_y_tk * forward_output_unweighted)

        # Convert grad_w2 back to original dtype (bfloat16)
        grad_w2 = grad_w2.to(intermediate_cache2.dtype)

        return grad_intermediate_cache2, grad_w2, grad_topk_weights, None


# ============================================================================
# Import Triton Implementation
# ============================================================================

from slime.backends.fsdp_utils.kernels.fused_experts import DownProjFunction as DownProjFunctionTriton
from slime.backends.fsdp_utils.kernels.fused_experts import GateUpProjFunction as GateUpProjFunctionTriton

# ============================================================================
# Test Fixtures and Utilities
# ============================================================================


@pytest.fixture
def setup_moe_params():
    """Setup MOE parameters for testing."""
    torch.manual_seed(42)

    # Small parameters for easier debugging
    num_tokens = 64
    hidden_size = 128
    intermediate_size = 256
    num_experts = 4
    topk = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Create input tensors with random values for better testing
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create expert weights
    w1 = torch.randn(num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype)

    # Create router outputs
    topk_weights = torch.rand(num_tokens, topk, device=device, dtype=dtype)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # normalize

    # Random expert selection
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)], dim=0).to(
        torch.int32
    )

    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "device": device,
        "dtype": dtype,
    }


# ============================================================================
# Test Cases
# ============================================================================


class TestGateUpProjBackward:
    """Test GateUpProjFunction backward pass comparison."""

    def test_forward_consistency(self, setup_moe_params):
        """Test that Triton and Python implementations produce same forward output."""
        params = setup_moe_params

        # Python implementation
        out_python = GateUpProjFunctionPython.apply(
            params["hidden_states"].clone(),
            params["w1"].clone(),
            params["topk_weights"].clone(),
            params["topk_ids"].clone(),
        )

        # Triton implementation
        out_triton = GateUpProjFunctionTriton.apply(
            params["hidden_states"].clone(),
            params["w1"].clone(),
            params["topk_weights"].clone(),
            params["topk_ids"].clone(),
        )

        # Check outputs are close
        torch.testing.assert_close(out_python, out_triton, rtol=1, atol=1)
        print("✓ GateUpProjFunction forward test passed")

    def test_backward_consistency(self, setup_moe_params):
        """Test that Triton and Python implementations produce same gradients."""
        params = setup_moe_params

        # Prepare inputs with requires_grad
        hidden_states_python = params["hidden_states"].clone().requires_grad_(True)
        w1_python = params["w1"].clone().requires_grad_(True)
        topk_weights_python = params["topk_weights"].clone().requires_grad_(True)
        topk_ids_python = params["topk_ids"].clone()

        hidden_states_triton = params["hidden_states"].clone().requires_grad_(True)
        w1_triton = params["w1"].clone().requires_grad_(True)
        topk_weights_triton = params["topk_weights"].clone().requires_grad_(True)
        topk_ids_triton = params["topk_ids"].clone()

        # Python implementation
        out_python = GateUpProjFunctionPython.apply(
            hidden_states_python,
            w1_python,
            topk_weights_python,
            topk_ids_python,
        )

        # Triton implementation
        out_triton = GateUpProjFunctionTriton.apply(
            hidden_states_triton,
            w1_triton,
            topk_weights_triton,
            topk_ids_triton,
        )

        # Create gradient for backward
        grad_output = torch.randn_like(out_python)

        # Backward pass
        out_python.backward(grad_output.clone())
        out_triton.backward(grad_output.clone())

        # Check hidden_states gradients
        print("\n" + "=" * 80)
        print("GateUpProjFunction Backward - hidden_states gradients:")
        print("=" * 80)
        if hidden_states_python.grad is not None and hidden_states_triton.grad is not None:
            diff = hidden_states_python.grad - hidden_states_triton.grad
            max_diff = torch.max(torch.abs(diff))
            print(f"Max absolute difference: {max_diff:.6f}")
            torch.testing.assert_close(hidden_states_python.grad, hidden_states_triton.grad, rtol=1, atol=1)
            print("✓ hidden_states gradient matches")
        print("=" * 80 + "\n")

        # Check w1 gradients
        print("\n" + "=" * 80)
        print("GateUpProjFunction Backward - w1 gradients:")
        print("=" * 80)
        if w1_python.grad is not None and w1_triton.grad is not None:
            diff = w1_python.grad - w1_triton.grad
            max_diff = torch.max(torch.abs(diff))
            print(f"Max absolute difference: {max_diff:.6f}")
            torch.testing.assert_close(w1_python.grad, w1_triton.grad, rtol=1, atol=1)
            print("✓ w1 gradient matches")
        print("=" * 80 + "\n")

        print("✓ GateUpProjFunction backward test passed")


class TestDownProjBackward:
    """Test DownProjFunction backward pass comparison."""

    def test_forward_consistency(self, setup_moe_params):
        """Test that Triton and Python implementations produce same forward output."""
        params = setup_moe_params

        # Create intermediate input (after SiluAndMul)
        num_tokens = params["hidden_states"].shape[0]
        topk = params["topk_ids"].shape[1]
        intermediate_size = params["w2"].shape[2]
        intermediate_cache2 = torch.randn(
            num_tokens * topk, intermediate_size, device=params["device"], dtype=params["dtype"]
        )

        # Python implementation
        out_python = DownProjFunctionPython.apply(
            intermediate_cache2.clone(),
            params["w2"].clone(),
            params["topk_weights"].clone(),
            params["topk_ids"].clone(),
        )

        # Triton implementation
        out_triton = DownProjFunctionTriton.apply(
            intermediate_cache2.clone(),
            params["w2"].clone(),
            params["topk_weights"].clone(),
            params["topk_ids"].clone(),
        )

        # Check outputs are close
        torch.testing.assert_close(out_python, out_triton, rtol=1, atol=1)
        print("✓ DownProjFunction forward test passed")

    def test_backward_consistency(self, setup_moe_params):
        """Test that Triton and Python implementations produce same gradients."""
        params = setup_moe_params

        # Create intermediate input
        num_tokens = params["hidden_states"].shape[0]
        topk = params["topk_ids"].shape[1]
        intermediate_size = params["w2"].shape[2]

        intermediate_cache2_base = torch.randn(
            num_tokens * topk, intermediate_size, device=params["device"], dtype=params["dtype"]
        )

        intermediate_cache2_python = intermediate_cache2_base.clone().requires_grad_(True)
        intermediate_cache2_triton = intermediate_cache2_base.clone().requires_grad_(True)

        w2_python = params["w2"].clone().requires_grad_(True)
        w2_triton = params["w2"].clone().requires_grad_(True)

        topk_weights_python = params["topk_weights"].clone().requires_grad_(True)
        topk_weights_triton = params["topk_weights"].clone().requires_grad_(True)

        # Python implementation
        out_python = DownProjFunctionPython.apply(
            intermediate_cache2_python,
            w2_python,
            topk_weights_python,
            params["topk_ids"],
        )

        # Triton implementation
        out_triton = DownProjFunctionTriton.apply(
            intermediate_cache2_triton,
            w2_triton,
            topk_weights_triton,
            params["topk_ids"],
        )

        # Create gradient for backward
        grad_output = torch.randn_like(out_python)

        # Backward pass
        out_python.backward(grad_output.clone())
        out_triton.backward(grad_output.clone())

        # Check intermediate_cache2 gradients
        print("\n" + "=" * 80)
        print("DownProjFunction Backward - intermediate_cache2 gradients:")
        print("=" * 80)
        if intermediate_cache2_python.grad is not None and intermediate_cache2_triton.grad is not None:
            diff = intermediate_cache2_python.grad - intermediate_cache2_triton.grad
            max_diff = torch.max(torch.abs(diff))
            print(f"Max absolute difference: {max_diff:.6f}")
            torch.testing.assert_close(
                intermediate_cache2_python.grad, intermediate_cache2_triton.grad, rtol=1, atol=1
            )
            print("✓ intermediate_cache2 gradient matches")
        print("=" * 80 + "\n")

        # Check topk_weights gradients
        print("\n" + "=" * 80)
        print("DownProjFunction Backward - topk_weights gradients:")
        print("=" * 80)
        if topk_weights_python.grad is not None and topk_weights_triton.grad is not None:
            diff = topk_weights_python.grad - topk_weights_triton.grad
            max_diff = torch.max(torch.abs(diff))
            print(f"Max absolute difference: {max_diff:.6f}")
            torch.testing.assert_close(topk_weights_python.grad, topk_weights_triton.grad, rtol=1, atol=1)
            print("✓ topk_weights gradient matches")
        print("=" * 80 + "\n")

        # Check w2 gradients
        print("\n" + "=" * 80)
        print("DownProjFunction Backward - w2 gradients:")
        print("=" * 80)
        if w2_python.grad is not None and w2_triton.grad is not None:
            diff = w2_python.grad - w2_triton.grad
            max_diff = torch.max(torch.abs(diff))
            print(f"Max absolute difference: {max_diff:.6f}")
            torch.testing.assert_close(w2_python.grad, w2_triton.grad, rtol=1, atol=1)
            print("✓ w2 gradient matches")
        print("=" * 80 + "\n")

        print("✓ DownProjFunction backward test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Running Fused Experts Backward Tests")
    print("Testing: Triton Implementation vs Python Reference")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping tests")
        return

    # Setup parameters
    torch.manual_seed(42)
    params_dict = {}

    # Small parameters for testing
    num_tokens = 64
    hidden_size = 128
    intermediate_size = 256
    num_experts = 4
    topk = 2

    device = "cuda"
    dtype = torch.bfloat16

    # Create input tensors
    params_dict["hidden_states"] = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    params_dict["w1"] = torch.randn(num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype)
    params_dict["w2"] = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype)
    params_dict["topk_weights"] = torch.rand(num_tokens, topk, device=device, dtype=dtype)
    params_dict["topk_weights"] = params_dict["topk_weights"] / params_dict["topk_weights"].sum(dim=-1, keepdim=True)
    params_dict["topk_ids"] = torch.stack(
        [torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)], dim=0
    ).to(torch.int32)
    params_dict["device"] = device
    params_dict["dtype"] = dtype

    print("\n" + "=" * 80)
    print("Testing GateUpProjFunction Backward")
    print("=" * 80)
    test_gate_up = TestGateUpProjBackward()
    test_gate_up.test_forward_consistency(params_dict)
    test_gate_up.test_backward_consistency(params_dict)

    print("\n" + "=" * 80)
    print("Testing DownProjFunction Backward")
    print("=" * 80)
    test_down = TestDownProjBackward()
    test_down.test_forward_consistency(params_dict)
    test_down.test_backward_consistency(params_dict)

    print("\n" + "=" * 80)
    print("All Backward Tests Passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
