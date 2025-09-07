import torch
import torch.nn.functional as F
import transformer_engine_torch as tex
import triton
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

from slime.utils.fp8_kernel import blockwise_cast_to_fp8_triton

device = "cuda"
dtype = torch.bfloat16
fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def per_block_cast_to_fp8_slime(weight, weight_block_size=[128, 128]):
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

    # per block quant
    block_n, block_k = weight_block_size[0], weight_block_size[1]

    shape_0, shape_1 = weight.shape

    n_tiles = ceil_div(shape_0, block_n)
    k_tiles = ceil_div(shape_1, block_k)

    q_weight = F.pad(
        weight,
        (0, k_tiles * block_k - shape_1, 0, n_tiles * block_n - shape_0),
        mode="constant",
        value=0.0,
    )

    qweight = q_weight.reshape(n_tiles, block_n, k_tiles, block_k)
    block_max = torch.max(torch.abs(qweight), dim=1, keepdim=True)[0]
    block_max = torch.max(block_max, dim=3, keepdim=True)[0]

    scale = block_max.to(torch.float32) / FP8_MAX
    qweight = (
        (qweight / scale)
        .clamp(min=FP8_MIN, max=FP8_MAX)
        .reshape((n_tiles * block_n, k_tiles * block_k))
        .to(torch.float8_e4m3fn)
    )
    qweight = qweight[:shape_0, :shape_1]
    scale = scale.squeeze()
    return qweight, scale


def te_per_token_group_quant_8bit(weight: torch.Tensor, quantizer, weight_block_size=[128, 128]):
    block_n, block_k = weight_block_size[0], weight_block_size[1]
    shape_0, shape_1 = weight.shape
    n_tiles = ceil_div(shape_0, block_n)
    k_tiles = ceil_div(shape_1, block_k)
    param = quantizer(weight)
    return param._rowwise_data, param._rowwise_scale_inv[:n_tiles, :k_tiles]


ref_lib = "pytorch"  # pytorch or te
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib, "triton"],  # Label name for the lines
        line_names=[ref_lib, "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="GB/s",  # Label name for the y-axis
        plot_name="quant-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)


@triton.testing.perf_report(configs)
def benchmark(M, N, provider):
    x = torch.randn((M, N), device=device, dtype=dtype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "pytorch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: per_block_cast_to_fp8_slime(x), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: blockwise_cast_to_fp8_triton(x), quantiles=quantiles)
    if provider == "te":
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            force_pow_2_scales=False,
            block_scaling_dim=2,
        )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: te_per_token_group_quant_8bit(x, quantizer), quantiles=quantiles
        )
    gbps = lambda ms: x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def benchmark_percise():
    for M in (7168, 2112, 1536, 24576, 512, 32768, 16384, 4096, 2048):
        for N in (2048, 4096, 8192):
            x_ref = torch.rand(M, N, dtype=dtype, device=device)
            x_triton, x_s_triton = blockwise_cast_to_fp8_triton(x_ref)
            x_slime, x_s_slime = per_block_cast_to_fp8_slime(x_ref)
            torch.testing.assert_close(x_triton.to(torch.float32), x_slime.to(torch.float32), rtol=1e-3, atol=1e-5)
            torch.testing.assert_close(x_s_triton, x_s_slime, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    benchmark_percise()
    benchmark.run(show_plots=True, print_data=True, save_path=f"./plot/{ref_lib}")
