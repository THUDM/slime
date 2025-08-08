import torch
from slime.backends.megatron_utils.megatron_to_hf import quantize_param_torch, per_block_fp8_quant_triton

import time


def benchmark_performance():

    weight_shapes = [(18432, 7168), (2048, 1536)]
    block_size = (128, 128)

    print("=" * 60)
    print("FP8 Quantization Performance Test")
    print("=" * 60)

    for weight_shape in weight_shapes:
        print(f"Shape: {weight_shape}, Block size: {block_size}")
        print("-" * 60)

        # PyTorch benchmark
        weight = torch.randn(weight_shape, dtype=torch.float16, device="cuda")
        name = "test.weight"

        for _ in range(5):
            quantize_param_torch(name, weight, block_size)

        start = time.perf_counter()
        for _ in range(1024):
            quantize_param_torch(name, weight, block_size)
        end = time.perf_counter()
        pytorch_time = (end - start) * 1000 / 1024
        print(f"PyTorch: {pytorch_time:.2f} ms")

        # Triton benchmark
        for _ in range(5):
            per_block_fp8_quant_triton(name, weight, block_size)

        start = time.perf_counter()
        for _ in range(1024):
            per_block_fp8_quant_triton(name, weight, block_size)
        end = time.perf_counter()
        triton_time = (end - start) * 1000 / 1024
        print(f"Triton:  {triton_time:.2f} ms")

        if triton_time > 0:
            speedup = pytorch_time / triton_time
            print(f"Speedup: {speedup:.2f}x")
        print("-" * 60)
    print("=" * 60)


def check_accuracy():

    weight_shapes = [(1024, 512)]
    block_size = (128, 128)

    print("\n" + "=" * 60)
    print("Accuracy Check")
    print("=" * 60)

    device = torch.device("cuda")

    for weight_shape in weight_shapes:
        print(f"Testing shape: {weight_shape}")
        print("-" * 40)
        weight = torch.randn(weight_shape, dtype=torch.float16, device=device)
        name = "test.weight"

        # PyTorch result
        pytorch_result = quantize_param_torch(name, weight, block_size)
        pytorch_qweight = pytorch_result[0][1]
        pytorch_scale = pytorch_result[1][1]

        # Triton result
        triton_result = per_block_fp8_quant_triton(name, weight, block_size)
        triton_qweight = triton_result[0][1]
        triton_scale = triton_result[1][1]

        # Compare results
        qweight_diff = torch.abs(pytorch_qweight.to(torch.float32) - triton_qweight.to(torch.float32))
        scale_diff = torch.abs(pytorch_scale - triton_scale)

        print(f"QWeight diff - mean: {qweight_diff.mean():.6f}, max: {qweight_diff.max():.6f}")
        print(f"Scale diff - mean: {scale_diff.mean():.6f}, max: {scale_diff.max():.6f}")

        print(f"PyTorch scale range: [{pytorch_scale.min():.6f}, {pytorch_scale.max():.6f}]")
        print(f"Triton scale range: [{triton_scale.min():.6f}, {triton_scale.max():.6f}]")

        if torch.allclose(
            pytorch_qweight.to(torch.float32), triton_qweight.to(torch.float32), rtol=1e-3, atol=1e-3
        ) and torch.allclose(pytorch_scale, triton_scale, rtol=1e-3, atol=1e-5):
            print("✅ All implementations match")
        else:
            print("❌ Implementations differ")

        print("-" * 40)

    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)
    benchmark_performance()
    check_accuracy()
