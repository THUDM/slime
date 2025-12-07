"""
Test script to verify the correctness of backward implementations
for GateUpProjFunction, SiluAndMulFunction, and DownProjFunction.

Uses torch.autograd.gradcheck to compare custom backward with numerical gradients.
"""

import torch
from torch.autograd import gradcheck
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slime.backends.fsdp_utils.kernels.fused_experts import (
    GateUpProjFunction,
    SiluAndMulFunction,
    DownProjFunction,
)

# Check CUDA availability
if not torch.cuda.is_available():
    print("="*60)
    print("⚠️  CUDA is not available!")
    print("="*60)
    print("\nThese tests require CUDA to run because the fused_experts")
    print("kernels (moe_align_block_size, silu_and_mul, etc.) only")
    print("support CUDA backend.")
    print("\nPlease run these tests on a machine with CUDA support.")
    print("="*60)
    sys.exit(0)

DEVICE = "cuda"

def test_gate_up_proj_backward():
    """Test GateUpProjFunction backward pass using gradcheck."""
    print("\n" + "="*60)
    print("Testing GateUpProjFunction.backward()")
    print("="*60)

    # Use small tensor sizes for numerical gradient computation
    # Use float32 for compatibility with fused MoE kernels
    batch_size = 2
    hidden_dim = 64
    num_experts = 4
    intermediate_size = 128
    topk = 2

    # Prepare test inputs on CUDA device
    hidden_states = torch.randn(
        batch_size, hidden_dim,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    w1 = torch.randn(
        num_experts, intermediate_size, hidden_dim,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    topk_weights = torch.randn(
        batch_size, topk,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    topk_ids = torch.randint(
        0, num_experts, (batch_size, topk),
        dtype=torch.long,
        device=DEVICE
    )

    print(f"Input shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  w1: {w1.shape}")
    print(f"  topk_weights: {topk_weights.shape}")
    print(f"  topk_ids: {topk_ids.shape}")

    # Run gradcheck
    try:
        test_passed = gradcheck(
            GateUpProjFunction.apply,
            (hidden_states, w1, topk_weights, topk_ids),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True
        )
        print(f"\n✅ GateUpProjFunction gradcheck: PASSED")
        return True
    except Exception as e:
        print(f"\n❌ GateUpProjFunction gradcheck: FAILED")
        print(f"Error: {str(e)}")
        return False

def test_silu_and_mul_backward():
    """Test SiluAndMulFunction backward pass using gradcheck."""
    print("\n" + "="*60)
    print("Testing SiluAndMulFunction.backward()")
    print("="*60)

    # SiluAndMul expects input of shape (num_tokens, N)
    # where N is intermediate_size * 2
    batch_size = 4
    intermediate_size = 128

    intermediate_cache1 = torch.randn(
        batch_size, intermediate_size * 2,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )

    print(f"Input shapes:")
    print(f"  intermediate_cache1: {intermediate_cache1.shape}")

    # Run gradcheck
    try:
        test_passed = gradcheck(
            SiluAndMulFunction.apply,
            (intermediate_cache1,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True
        )
        print(f"\n✅ SiluAndMulFunction gradcheck: PASSED")
        return True
    except Exception as e:
        print(f"\n❌ SiluAndMulFunction gradcheck: FAILED")
        print(f"Error: {str(e)}")
        return False

def test_down_proj_backward():
    """Test DownProjFunction backward pass using gradcheck."""
    print("\n" + "="*60)
    print("Testing DownProjFunction.backward()")
    print("="*60)

    # Use small tensor sizes for numerical gradient computation
    batch_size = 2
    hidden_dim = 64
    num_experts = 4
    intermediate_size = 128
    topk = 2

    # Prepare test inputs on CUDA device
    # intermediate_cache2 shape: (batch_size * topk, intermediate_size)
    intermediate_cache2 = torch.randn(
        batch_size * topk, intermediate_size,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_size,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    topk_weights = torch.randn(
        batch_size, topk,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    topk_ids = torch.randint(
        0, num_experts, (batch_size, topk),
        dtype=torch.long,
        device=DEVICE
    )

    print(f"Input shapes:")
    print(f"  intermediate_cache2: {intermediate_cache2.shape}")
    print(f"  w2: {w2.shape}")
    print(f"  topk_weights: {topk_weights.shape}")
    print(f"  topk_ids: {topk_ids.shape}")

    # Run gradcheck
    try:
        test_passed = gradcheck(
            DownProjFunction.apply,
            (intermediate_cache2, w2, topk_weights, topk_ids),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True
        )
        print(f"\n✅ DownProjFunction gradcheck: PASSED")
        return True
    except Exception as e:
        print(f"\n❌ DownProjFunction gradcheck: FAILED")
        print(f"Error: {str(e)}")
        return False

def test_full_moe_pipeline():
    """Test the full MoE pipeline to ensure gradients flow correctly."""
    print("\n" + "="*60)
    print("Testing Full MoE Pipeline (GateUpProj -> SiluAndMul -> DownProj)")
    print("="*60)

    batch_size = 2
    hidden_dim = 64
    num_experts = 4
    intermediate_size = 128
    topk = 2

    # Prepare inputs on CUDA device
    hidden_states = torch.randn(
        batch_size, hidden_dim,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_dim,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_size,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    topk_weights = torch.randn(
        batch_size, topk,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True
    )
    topk_ids = torch.randint(
        0, num_experts, (batch_size, topk),
        dtype=torch.long,
        device=DEVICE
    )

    print(f"Input shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  w1: {w1.shape}")
    print(f"  w2: {w2.shape}")

    try:
        # Forward pass
        intermediate_cache1 = GateUpProjFunction.apply(
            hidden_states, w1, topk_weights, topk_ids
        )
        print(f"\nAfter GateUpProj: {intermediate_cache1.shape}")

        intermediate_cache2 = SiluAndMulFunction.apply(intermediate_cache1)
        print(f"After SiluAndMul: {intermediate_cache2.shape}")

        intermediate_cache3 = DownProjFunction.apply(
            intermediate_cache2, w2, topk_weights, topk_ids
        )
        print(f"After DownProj: {intermediate_cache3.shape}")

        # Compute loss and backward
        loss = intermediate_cache3.sum()
        loss.backward()

        # Check if gradients are computed
        assert hidden_states.grad is not None, "hidden_states.grad is None"
        assert w1.grad is not None, "w1.grad is None"
        assert w2.grad is not None, "w2.grad is None"
        assert topk_weights.grad is not None, "topk_weights.grad is None"

        print(f"\n✅ Full pipeline test: PASSED")
        print(f"  hidden_states.grad shape: {hidden_states.grad.shape}")
        print(f"  w1.grad shape: {w1.grad.shape}")
        print(f"  w2.grad shape: {w2.grad.shape}")
        print(f"  topk_weights.grad shape: {topk_weights.grad.shape}")

        # Check for NaN or Inf in gradients
        if torch.isnan(hidden_states.grad).any() or torch.isinf(hidden_states.grad).any():
            print(f"⚠️  Warning: hidden_states.grad contains NaN or Inf")
        if torch.isnan(w1.grad).any() or torch.isinf(w1.grad).any():
            print(f"⚠️  Warning: w1.grad contains NaN or Inf")
        if torch.isnan(w2.grad).any() or torch.isinf(w2.grad).any():
            print(f"⚠️  Warning: w2.grad contains NaN or Inf")
        if torch.isnan(topk_weights.grad).any() or torch.isinf(topk_weights.grad).any():
            print(f"⚠️  Warning: topk_weights.grad contains NaN or Inf")

        return True
    except Exception as e:
        print(f"\n❌ Full pipeline test: FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Fused Experts Backward Implementation Tests")
    print("="*60)
    print(f"\nDevice: {DEVICE}")
    print("Note: Using float32 for compatibility with fused MoE kernels")
    print("Tests may take a few minutes due to numerical gradient computation...\n")

    results = []

    # Test individual components
    results.append(("GateUpProjFunction", test_gate_up_proj_backward()))
    results.append(("SiluAndMulFunction", test_silu_and_mul_backward()))
    results.append(("DownProjFunction", test_down_proj_backward()))

    # Test full pipeline
    results.append(("Full MoE Pipeline", test_full_moe_pipeline()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:30s}: {status}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests PASSED!")
    else:
        print("⚠️  Some tests FAILED. Please review the backward implementations.")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())