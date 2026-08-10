#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace device::glm5_router_gemm {

constexpr int kNumExperts = 256;
constexpr int kHiddenDim = 6144;
constexpr int kBlockSize = 128;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockSize / kWarpSize;
constexpr int kValuesPerThread = 8;

template <int kNumTokens>
__global__ __launch_bounds__(kBlockSize, 1) void kernel(
    float* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight) {
  const int expert = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  constexpr int kValuesPerIteration = kValuesPerThread * kBlockSize;
  constexpr int kIterations = kHiddenDim / kValuesPerIteration;

  float accumulators[kNumTokens] = {};
  __shared__ float warp_sums[kNumTokens][kNumWarps];
  const __nv_bfloat16* weight_row = weight + expert * kHiddenDim;

#pragma unroll
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const int k_base = iteration * kValuesPerIteration + tid * kValuesPerThread;
    const uint4 weight_vector =
        *reinterpret_cast<const uint4*>(weight_row + k_base);
    const __nv_bfloat16* weight_values =
        reinterpret_cast<const __nv_bfloat16*>(&weight_vector);

#pragma unroll
    for (int token = 0; token < kNumTokens; ++token) {
      const uint4 input_vector = *reinterpret_cast<const uint4*>(
          input + token * kHiddenDim + k_base);
      const __nv_bfloat16* input_values =
          reinterpret_cast<const __nv_bfloat16*>(&input_vector);
#pragma unroll
      for (int value = 0; value < kValuesPerThread; ++value) {
        accumulators[token] += __bfloat162float(input_values[value]) *
                               __bfloat162float(weight_values[value]);
      }
    }
  }

#pragma unroll
  for (int token = 0; token < kNumTokens; ++token) {
    float sum = accumulators[token];
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
      warp_sums[token][warp] = sum;
    }
  }
  __syncthreads();

  if (tid == 0) {
#pragma unroll
    for (int token = 0; token < kNumTokens; ++token) {
      float sum = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < kNumWarps; ++source_warp) {
        sum += warp_sums[token][source_warp];
      }
      output[token * kNumExperts + expert] = sum;
    }
  }
}

template <int kNumTokens>
void launch(
    float* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    cudaStream_t stream) {
  kernel<kNumTokens><<<kNumExperts, kBlockSize, 0, stream>>>(
      output, input, weight);
}

}  // namespace device::glm5_router_gemm

namespace {

void glm5_router_gemm(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight) {
  using namespace host;

  RuntimeCheck(input.device().device_type == kDLCUDA, "input must be CUDA");
  RuntimeCheck(input.is_contiguous(), "input must be contiguous");
  RuntimeCheck(
      input.dtype().code == kDLBfloat && input.dtype().bits == 16,
      "input must be bfloat16");
  RuntimeCheck(input.ndim() == 2, "input must be 2D");
  RuntimeCheck(input.size(1) == 6144, "input hidden dimension must be 6144");

  RuntimeCheck(weight.device().device_type == kDLCUDA, "weight must be CUDA");
  RuntimeCheck(weight.is_contiguous(), "weight must be contiguous");
  RuntimeCheck(
      weight.dtype().code == kDLBfloat && weight.dtype().bits == 16,
      "weight must be bfloat16");
  RuntimeCheck(
      weight.ndim() == 2 && weight.size(0) == 256 && weight.size(1) == 6144,
      "weight must have shape [256, 6144]");

  RuntimeCheck(output.device().device_type == kDLCUDA, "output must be CUDA");
  RuntimeCheck(output.is_contiguous(), "output must be contiguous");
  RuntimeCheck(
      output.dtype().code == kDLFloat && output.dtype().bits == 32,
      "output must be float32");
  RuntimeCheck(
      output.ndim() == 2 && output.size(0) == input.size(0) &&
          output.size(1) == 256,
      "output must have shape [num_tokens, 256]");

  const int num_tokens = static_cast<int>(input.size(0));
  RuntimeCheck(
      num_tokens >= 1 && num_tokens <= 16,
      "num_tokens must be in [1, 16]");
  const cudaStream_t stream = LaunchKernel::resolve_device(input.device());
  auto* output_ptr = static_cast<float*>(output.data_ptr());
  const auto* input_ptr =
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  const auto* weight_ptr =
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());

#define LAUNCH_GLM5_ROUTER_GEMM(M)                                       \
  case M:                                                               \
    device::glm5_router_gemm::launch<M>(                                \
        output_ptr, input_ptr, weight_ptr, stream);                      \
    break
  switch (num_tokens) {
    LAUNCH_GLM5_ROUTER_GEMM(1);
    LAUNCH_GLM5_ROUTER_GEMM(2);
    LAUNCH_GLM5_ROUTER_GEMM(3);
    LAUNCH_GLM5_ROUTER_GEMM(4);
    LAUNCH_GLM5_ROUTER_GEMM(5);
    LAUNCH_GLM5_ROUTER_GEMM(6);
    LAUNCH_GLM5_ROUTER_GEMM(7);
    LAUNCH_GLM5_ROUTER_GEMM(8);
    LAUNCH_GLM5_ROUTER_GEMM(9);
    LAUNCH_GLM5_ROUTER_GEMM(10);
    LAUNCH_GLM5_ROUTER_GEMM(11);
    LAUNCH_GLM5_ROUTER_GEMM(12);
    LAUNCH_GLM5_ROUTER_GEMM(13);
    LAUNCH_GLM5_ROUTER_GEMM(14);
    LAUNCH_GLM5_ROUTER_GEMM(15);
    LAUNCH_GLM5_ROUTER_GEMM(16);
  }
#undef LAUNCH_GLM5_ROUTER_GEMM
}

}  // namespace
