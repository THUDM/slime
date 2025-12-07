import torch
import pytest
import time
from slime.backends.fsdp_utils.kernels.fused_experts import (
    GateUpProjFunction,
    SiluAndMulFunction,
    DownProjFunction,
    MoeSumReduceFunction,
)
from slime.backends.fsdp_utils.kernels.fused_experts_cuda import (
    GateUpProjFunction as GateUpProjFunctionCuda,
    SiluAndMulFunction as SiluAndMulFunctionCuda,
    DownProjFunction as DownProjFunctionCuda,
    MoeSumReduceFunction as MoeSumReduceFunctionCuda,
)


def fused_experts_impl_python(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    """Original Python implementation for comparison."""
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.bfloat16]

    intermediate_cache1 = GateUpProjFunction.apply(
        hidden_states,
        w1,
        topk_weights,
        topk_ids,
    )
    intermediate_cache2 = SiluAndMulFunction.apply(intermediate_cache1)
    intermediate_cache3 = DownProjFunction.apply(
        intermediate_cache2,
        w2,
        topk_weights,
        topk_ids,
    )
    output_hidden_states = MoeSumReduceFunction.apply(
        intermediate_cache3,
        hidden_states.shape,
    )
    return output_hidden_states

def fused_experts_impl_cuda(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    """CUDA implementation using Triton kernels."""
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.bfloat16]

    intermediate_cache1 = GateUpProjFunctionCuda.apply(
        hidden_states,
        w1,
        topk_weights,
        topk_ids,
    )
    intermediate_cache2 = SiluAndMulFunctionCuda.apply(intermediate_cache1)
    intermediate_cache3 = DownProjFunctionCuda.apply(
        intermediate_cache2,
        w2,
        topk_weights,
        topk_ids,
    )
    output_hidden_states = MoeSumReduceFunctionCuda.apply(
        intermediate_cache3,
        hidden_states.shape,
    )
    return output_hidden_states


@pytest.fixture
def setup_moe_params():
    """Setup MOE parameters for testing."""
    torch.manual_seed(42)
    
    # Parameters
    # batch_size = 4
    # seq_len = 4
    # hidden_size = 8
    # intermediate_size = 16
    # num_experts = 4
    # topk = 2

    num_tokens = 1024
    hidden_size = 2048
    intermediate_size = 4096
    num_experts = 8
    topk = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # Create input tensors
    hidden_states = torch.ones(num_tokens, hidden_size, device=device, dtype=dtype)
    
    # Create expert weights
    # w1: gate_proj + up_proj combined, shape (num_experts, intermediate_size * 2, hidden_size)
    w1 = torch.ones(num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype)
    
    # w2: down_proj, shape (num_experts, hidden_size, intermediate_size)
    w2 = torch.ones(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype)
    
    # Create router outputs
    topk_weights = torch.ones(num_tokens, topk, device=device, dtype=dtype)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # normalize

    # topk_ids = torch.arange(topk, device=device, dtype=torch.int32).unsqueeze(0).expand(num_tokens, -1)

    topk_ids = torch.stack([
        torch.randperm(num_experts, device=device)[:topk]
        for _ in range(num_tokens)
    ], dim=0).to(torch.int32)

    print(f"topk_ids: {topk_ids}")
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "device": device,
        "dtype": dtype,
    }


class TestGateUpProjFunction:
    """Test GateUpProjFunction forward and backward."""
    
    def test_forward_consistency(self, setup_moe_params):
        """Test that CUDA and Python implementations produce same forward output."""
        params = setup_moe_params
        
        # Python implementation
        out_python = GateUpProjFunction.apply(
            params["hidden_states"].clone(),
            params["w1"].clone(),
            params["topk_weights"].clone(),
            params["topk_ids"].clone(),
        )
        
        # CUDA implementation
        out_cuda = GateUpProjFunctionCuda.apply(
            params["hidden_states"].clone(),
            params["w1"].clone(),
            params["topk_weights"].clone(),
            params["topk_ids"].clone(),
        )
        
        # Check outputs are close
        torch.testing.assert_close(out_python, out_cuda, rtol=2e-2, atol=1e-2)
        print("✓ GateUpProjFunction forward test passed")
    
    def test_backward_consistency(self, setup_moe_params):
        """Test that CUDA and Python implementations produce same gradients."""
        params = setup_moe_params
        
        # Prepare inputs with requires_grad
        hidden_states_python = params["hidden_states"].clone().requires_grad_(True)
        w1_python = params["w1"].clone().requires_grad_(True)
        topk_weights_python = params["topk_weights"].clone().requires_grad_(True)
        topk_ids_python = params["topk_ids"].clone()  # topk_ids is int32, cannot require grad

        hidden_states_cuda = params["hidden_states"].clone().requires_grad_(True)
        w1_cuda = params["w1"].clone().requires_grad_(True)
        topk_weights_cuda = params["topk_weights"].clone().requires_grad_(True)
        topk_ids_cuda = params["topk_ids"].clone()  # topk_ids is int32, cannot require grad
        
        # Python implementation
        out_python = GateUpProjFunction.apply(
            hidden_states_python,
            w1_python,
            topk_weights_python,
            topk_ids_python,
        )
        
        # CUDA implementation
        out_cuda = GateUpProjFunctionCuda.apply(
            hidden_states_cuda,
            w1_cuda,
            topk_weights_cuda,
            topk_ids_cuda,
        )
        
        # Create gradient for backward
        grad_output = torch.ones_like(out_python)

        # Backward pass
        out_python.backward(grad_output)
        out_cuda.backward(grad_output)

        # Print gradients for debugging
        print("\n" + "="*80)
        print("GateUpProjFunction Backward - hidden_states gradients:")
        print("="*80)
        print(f"hidden_states_python.grad:\n{hidden_states_python.grad}")
        print(f"\nhidden_states_cuda.grad:\n{hidden_states_cuda.grad}")
        print(f"\nDifference:\n{hidden_states_python.grad - hidden_states_cuda.grad}")
        print(f"Max absolute difference: {torch.max(torch.abs(hidden_states_python.grad - hidden_states_cuda.grad))}")
        print("="*80 + "\n")

        # Check gradients are close
        torch.testing.assert_close(
            hidden_states_python.grad,
            hidden_states_cuda.grad,
            rtol=2e-2,
            atol=1e-2
        )

        # Print gradients for debugging
        print("\n" + "=" * 80)
        print("GateUpProjFunction Backward - hidden_states gradients:")
        print("=" * 80)
        print(f"w1_python.grad:\n{w1_python.grad}\n{w1_python.grad.shape}")
        print(f"\nw1_cuda.grad:\n{w1_cuda.grad}\n{w1_cuda.grad.shape}")
        print(f"\nDifference:\n{w1_python.grad - w1_cuda.grad}")
        print(f"Max absolute difference: {torch.max(torch.abs(w1_python.grad - w1_cuda.grad))}")
        print("=" * 80 + "\n")

        torch.testing.assert_close(
            w1_python.grad,
            w1_cuda.grad,
            rtol=2e-2,
            atol=1e-2
        )

        print("✓ GateUpProjFunction backward test passed")


class TestDownProjFunction:
    """Test DownProjFunction forward and backward."""

    def test_forward_consistency(self, setup_moe_params):
        """Test that CUDA and Python implementations produce same forward output."""
        params = setup_moe_params

        # Create intermediate input (after SiluAndMul)
        num_tokens = params["hidden_states"].shape[0]
        topk = params["topk_ids"].shape[1]
        intermediate_size = params["w2"].shape[2]
        intermediate_cache2 = torch.ones(
            num_tokens * topk,
            intermediate_size,
            device=params["device"],
            dtype=params["dtype"]
        )

        # Python implementation
        out_python = DownProjFunction.apply(
            intermediate_cache2.clone(),
            params["w2"].clone(),
            params["topk_weights"].clone(),
            params["topk_ids"].clone(),
        )

        # CUDA implementation
        out_cuda = DownProjFunctionCuda.apply(
            intermediate_cache2.clone(),
            params["w2"].clone(),
            params["topk_weights"].clone(),
            params["topk_ids"].clone(),
        )

        # Check outputs are close
        torch.testing.assert_close(out_python, out_cuda, rtol=2e-2, atol=1e-2)
        print("✓ DownProjFunction forward test passed")

    def test_backward_consistency(self, setup_moe_params):
        """Test that CUDA and Python implementations produce same gradients."""
        params = setup_moe_params

        # Create intermediate input
        num_tokens = params["hidden_states"].shape[0]
        topk = params["topk_ids"].shape[1]
        intermediate_size = params["w2"].shape[2]

        intermediate_cache2_python = torch.ones(
            num_tokens * topk,
            intermediate_size,
            device=params["device"],
            dtype=params["dtype"]
        ).requires_grad_(True)

        intermediate_cache2_cuda = intermediate_cache2_python.clone().detach().requires_grad_(True)

        w2_python = params["w2"].clone().requires_grad_(True)
        w2_cuda = params["w2"].clone().requires_grad_(True)

        topk_weights_python = params["topk_weights"].clone().requires_grad_(True)
        topk_weights_cuda = params["topk_weights"].clone().requires_grad_(True)

        # Warmup
        for _ in range(5):
            _ = DownProjFunction.apply(
                intermediate_cache2_python,
                w2_python,
                topk_weights_python,
                params["topk_ids"],
            )
            _ = DownProjFunctionCuda.apply(
                intermediate_cache2_cuda,
                w2_cuda,
                topk_weights_cuda,
                params["topk_ids"],
            )

        # Performance test for Python implementation
        num_runs = 100
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            out_python = DownProjFunction.apply(
                intermediate_cache2_python,
                w2_python,
                topk_weights_python,
                params["topk_ids"],
            )
        torch.cuda.synchronize()
        python_time = (time.time() - start_time) / num_runs

        # Performance test for CUDA implementation
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            out_cuda = DownProjFunctionCuda.apply(
                intermediate_cache2_cuda,
                w2_cuda,
                topk_weights_cuda,
                params["topk_ids"],
            )
        torch.cuda.synchronize()
        cuda_time = (time.time() - start_time) / num_runs

        # Print performance results
        print("\n" + "="*80)
        print("DownProjFunction Performance Comparison:")
        print("="*80)
        print(f"Python implementation average time: {python_time*1000:.4f} ms")
        print(f"CUDA implementation average time: {cuda_time*1000:.4f} ms")
        print(f"Speedup: {python_time/cuda_time:.2f}x")
        print("="*80 + "\n")

        # Create gradient for backward
        grad_output = torch.ones_like(out_python)

        # Backward pass
        out_python.backward(grad_output)
        out_cuda.backward(grad_output)

        # Print gradients for debugging
        print("\n" + "="*80)
        print("DownProjFunction Backward - intermediate_cache2 gradients:")
        print("="*80)
        print(f"intermediate_cache2_python.grad:\n{intermediate_cache2_python.grad}")
        print(f"\nintermediate_cache2_cuda.grad:\n{intermediate_cache2_cuda.grad}")
        print(f"\nDifference:\n{intermediate_cache2_python.grad - intermediate_cache2_cuda.grad}")
        print(f"Max absolute difference: {torch.max(torch.abs(intermediate_cache2_python.grad - intermediate_cache2_cuda.grad))}")
        print("="*80 + "\n")

        # Check gradients are close
        torch.testing.assert_close(
            intermediate_cache2_python.grad,
            intermediate_cache2_cuda.grad,
            rtol=2e-2,
            atol=1e-2
        )

        # Print topk_weights gradients for debugging
        print("\n" + "=" * 80)
        print("DownProjFunction Backward - topk_weights gradients:")
        print("=" * 80)
        print(f"topk_weights_python.grad:\n{topk_weights_python.grad}")
        print(f"\ntopk_weights_cuda.grad:\n{topk_weights_cuda.grad}")
        print(f"\nDifference:\n{topk_weights_python.grad - topk_weights_cuda.grad}")
        print(f"Max absolute difference: {torch.max(torch.abs(topk_weights_python.grad - topk_weights_cuda.grad))}")
        print("=" * 80 + "\n")

        torch.testing.assert_close(
            topk_weights_python.grad,
            topk_weights_cuda.grad,
            rtol=2e-2,
            atol=1e-2
        )

        # Print w2 gradients for debugging
        print("\n" + "="*80)
        print("DownProjFunction Backward - w2 gradients:")
        print("="*80)
        print(f"w2_python.grad:\n{w2_python.grad}")
        print(f"\nw2_cuda.grad:\n{w2_cuda.grad}")
        print(f"\nDifference:\n{w2_python.grad - w2_cuda.grad}")
        print(f"Max absolute difference: {torch.max(torch.abs(w2_python.grad - w2_cuda.grad))}")
        print("="*80 + "\n")

        torch.testing.assert_close(
            w2_python.grad,
            w2_cuda.grad,
            rtol=2e-2,
            atol=1e-2
        )


        print("✓ DownProjFunction backward test passed")

def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("Running Fused Experts Tests")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping tests")
        return
    
    # Setup
    torch.manual_seed(42)
    params_dict = {
        "hidden_states": None,
        "w1": None,
        "w2": None,
        "topk_weights": None,
        "topk_ids": None,
        "device": "cuda",
        "dtype": torch.bfloat16,
    }
    
    # Parameters
    batch_size = 1
    seq_len = 4
    hidden_size = 4
    intermediate_size = 8
    num_experts = 4
    topk = 2
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Create input tensors
    num_tokens = batch_size * seq_len
    params_dict["hidden_states"] = torch.ones(num_tokens, hidden_size, device=device, dtype=dtype)
    params_dict["w1"] = torch.ones(num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype)
    params_dict["w2"] = torch.ones(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype)
    params_dict["topk_weights"] = torch.ones(num_tokens, topk, device=device, dtype=dtype)
    params_dict["topk_weights"] = params_dict["topk_weights"] / params_dict["topk_weights"].sum(dim=-1, keepdim=True)
    params_dict["topk_ids"] = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)
    
    print("\n" + "="*80)
    print("Testing GateUpProjFunction")
    print("="*80)
    test_gate_up = TestGateUpProjFunction()
    test_gate_up.test_forward_consistency(params_dict)
    test_gate_up.test_backward_consistency(params_dict)
    
    print("\n" + "="*80)
    print("Testing DownProjFunction")
    print("="*80)
    test_down = TestDownProjFunction()
    test_down.test_forward_consistency(params_dict)
    test_down.test_backward_consistency(params_dict)
    
    # print("\n" + "="*80)
    # print("Testing SiluAndMul")
    # print("="*80)
    # test_silu_and_mul()
    #
    # print("\n" + "="*80)
    # print("Testing End-to-End Implementation")
    # print("="*80)
    # test_e2e = TestFusedExpertsEndToEnd()
    # test_e2e.test_forward_consistency(params_dict)
    # test_e2e.test_backward_consistency(params_dict)
    # test_e2e.test_different_shapes()
    #
    print("\n" + "="*80)
    print("All Tests Passed! ✓")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
