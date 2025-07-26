import torch
import triton
import triton.language as tl
from slime.backends.megatron_utils.megatron_to_hf import quantize_param

import time


@triton.jit
def per_block_fp8_quant_kernel(
    weight_ptr,
    qweight_ptr,
    scale_ptr,
    n_tiles_n,
    n_tiles_k,
    n,
    k,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_n = offs_n < n
    mask_k = offs_k < k
    mask = mask_n[:, None] & mask_k[None, :]

    weight_block = tl.load(weight_ptr + offs_n[:, None] * k + offs_k[None, :], mask=mask, other=0.0)

    block_max = tl.max(tl.abs(weight_block))

    scale = tl.where(block_max == 0.0, 1.0, block_max / FP8_MIN)

    tl.store(scale_ptr + pid_n * n_tiles_k + pid_k, scale)

    qweight = tl.where(mask, tl.clamp(weight_block / scale, FP8_MIN, FP8_MAX), 0.0)
    tl.store(qweight_ptr + offs_n[:, None] * k + offs_k[None, :], qweight, mask=mask)


def per_block_fp8_quant_triton(name, weight, weight_block_size):
    """Triton-based quantize_param function with same interface as PyTorch version"""
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"

    n, k = weight.shape
    block_n, block_k = weight_block_size
    n_tiles_n = (n + block_n - 1) // block_n
    n_tiles_k = (k + block_k - 1) // block_k

    # Allocate output tensors
    qweight = torch.zeros_like(weight, dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.zeros((n_tiles_n, n_tiles_k), dtype=torch.float32, device=weight.device)

    # Launch Triton kernel
    grid = (n_tiles_n, n_tiles_k)
    FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
    FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

    per_block_fp8_quant_kernel[grid](
        weight, qweight, scales, n_tiles_n, n_tiles_k, n, k, block_n, block_k, FP8_MIN, FP8_MAX
    )

    # Return in same format as PyTorch version
    scale_name = name.replace(".weight", ".weight_scale_inv")
    return [(name, qweight), (scale_name, scales.squeeze())]


def benchmark_performance():

    weight_shape = (18432, 7168)
    block_size = (128, 128)

    print("=" * 60)
    print("FP8 Quantization Performance Test")
    print("=" * 60)
    print(f"Shape: {weight_shape}, Block size: {block_size}")
    print("-" * 60)

    # PyTorch benchmark
    weight = torch.randn(weight_shape, dtype=torch.float16, device="cuda")
    name = "test.weight"

    for _ in range(5):
        quantize_param(name, weight, block_size)

    start = time.perf_counter()
    for _ in range(512):
        quantize_param(name, weight, block_size)
    end = time.perf_counter()
    pytorch_time = (end - start) * 1000 / 512
    print(f"PyTorch: {pytorch_time:.2f} ms")

    # Triton benchmark
    for _ in range(5):
        per_block_fp8_quant_triton(name, weight, block_size)

    start = time.perf_counter()
    for _ in range(512):
        per_block_fp8_quant_triton(name, weight, block_size)
    end = time.perf_counter()
    triton_time = (end - start) * 1000 / 512
    print(f"Triton:  {triton_time:.2f} ms")

    if triton_time > 0:
        speedup = pytorch_time / triton_time
        print(f"Speedup: {speedup:.2f}x")
    print("=" * 60)


def check_accuracy():

    weight_shape = (512, 512)
    block_size = (128, 128)

    print("\n" + "=" * 60)
    print("Accuracy Check")
    print("=" * 60)

    device = torch.device("cuda")
    weight = torch.randn(weight_shape, dtype=torch.float16, device=device)
    name = "test.weight"

    # PyTorch result
    pytorch_result = quantize_param(name, weight, block_size)
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
        pytorch_qweight.to(torch.float32), triton_qweight.to(torch.float32), rtol=1e-3, atol=1e-5
    ) and torch.allclose(pytorch_scale, triton_scale, rtol=1e-3, atol=1e-5):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")

    print("=" * 60)


if __name__ == "__main__":
    benchmark_performance()
    check_accuracy()
