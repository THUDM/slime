#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define FINAL_MASK 0xFFFFFFFF

__device__ __host__ __forceinline__
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__
float warpReduceMax(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}


__device__ __forceinline__
float warpReduceMin(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fminf(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

// almost all int4 use blocksize = [1, 32]
template<typename scalar_t>
__global__
void int4_quant_1x32_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    scalar_t* out_scale,
    scalar_t* out_zero,
    const int M, const int N,
    const int stride_xm, const int stride_xn,
    const int stride_om, const int stride_on,
    const int stride_osm, const int stride_osn,
    const int stride_ozm, const int stride_ozn,
    bool sym
) {
    constexpr int WARPS_PER_BLOCK = 8;
    const int needed_warps = ceil_div(N, 32);
    
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 0x1F;
    constexpr float SYM_CONS = 1.0f / 7.0f;
    constexpr float ASYM_CONS = 1.0f / 15.0f;
    
    const int row = blockIdx.x;

    for (int item = warp_id; item < needed_warps; item += WARPS_PER_BLOCK) {
        const int col = item * 32 + lane_id;
        float val = 0.0f;

        if (col < N) {
            val = static_cast<float>(x[row * stride_xm + col * stride_xn]);
        }
        
        float scale = 0.0f;
        float zero = 0.0f;
        
        if (sym) {
            float abs_val = fabsf(val);
            
            float block_max = warpReduceMax(abs_val);
            
            scale = fmaxf(block_max * SYM_CONS, 1e-5f);

            val = rintf(val / scale);
        } else {
            float block_min = warpReduceMin(val);
            float block_max = warpReduceMax(val);

            scale = fmaxf((block_max - block_min) * ASYM_CONS, 1e-5f);
            zero = fminf(fmaxf(-rintf(block_min / scale), 0.0f), 15.0f);
            
            val = rintf(val / scale) + zero;
        }
        
        if (col < N) {
            out[row * stride_om + col * stride_on] = static_cast<scalar_t>(val);
            out_scale[row * stride_osm + item * stride_osn] = static_cast<scalar_t>(scale);
            if(!sym) {
                out_zero[row * stride_ozm + item * stride_ozn] = static_cast<scalar_t>(zero);
            }
        }
    }
}

// for some transpose case, blocksize = [32, 1]
template<typename scalar_t>
__global__
void int4_quant_32x1_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    scalar_t* out_scale,
    scalar_t* out_zero,
    const int M, const int N,
    const int stride_xm, const int stride_xn,
    const int stride_om, const int stride_on,
    const int stride_osm, const int stride_osn,
    const int stride_ozm, const int stride_ozn,
    bool sym
) {
    constexpr int WARPS_PER_BLOCK = 8;
    const int start_row = blockIdx.x * 32;
    const int end_row = min((blockIdx.x + 1) * 32, M);
    
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 0x1F;
    constexpr float SYM_CONS = 1.0f / 7.0f;
    constexpr float ASYM_CONS = 1.0f / 15.0f;
    
    for (int item = warp_id; item < N; item += WARPS_PER_BLOCK) {
        const int col = item;
        const int row = start_row + lane_id;

        float val = 0.0f;

        if (row < end_row) {
            val = static_cast<float>(x[row * stride_xm + col * stride_xn]);
        }
    
        float scale = 0.0f;
        float zero = 0.0f;
        
        if (sym) {
            float abs_val = fabsf(val);
            
            float block_max = warpReduceMax(abs_val);
            
            scale = fmaxf(block_max * SYM_CONS, 1e-5f);

            val = rintf(val / scale);
        } else {
            float block_min = warpReduceMin(val);
            float block_max = warpReduceMax(val);

            scale = fmaxf((block_max - block_min) * ASYM_CONS, 1e-5f);
            zero = fminf(fmaxf(-rintf(block_min / scale), 0.0f), 15.0f);
            
            val = rintf(val / scale) + zero;
        }
        
        if (row < end_row) {
            out[row * stride_om + col * stride_on] = static_cast<scalar_t>(val);
            out_scale[blockIdx.x * stride_osm + item * stride_osn] = static_cast<scalar_t>(scale);
            if (!sym) {
                out_zero[blockIdx.x * stride_ozm + item * stride_ozn] = static_cast<scalar_t>(zero);
            }
        }
    }
}

template<typename scalar_t>
__global__ void int4_quant_common_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    scalar_t* out_scale,
    scalar_t* out_zero,
    const int M, const int N,
    const int stride_xm, const int stride_xn,
    const int stride_om, const int stride_on,
    const int stride_osm, const int stride_osn,
    const int stride_ozm, const int stride_ozn,
    const int BLOCK_M, const int BLOCK_N,
    bool sym
) {
    const int start_row = blockIdx.x * BLOCK_M;
    const int WARPS_PER_BLOCK = blockDim.x >> 5;
    
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;
    constexpr float SYM_CONS = 1.0f / 7.0f;
    constexpr float ASYM_CONS = 1.0f / 15.0f;
    constexpr int WARP_SIZE = 32;
    
    const int needed_warps = ceil_div(N, BLOCK_N);
    const int iters = ceil_div(BLOCK_M * BLOCK_N, 32);
    int warp_rows = 1;

    if (BLOCK_N <= WARP_SIZE) {
        warp_rows = WARP_SIZE / BLOCK_N;
    }
    
    for (int item = warp_id; item < needed_warps; item += WARPS_PER_BLOCK) {
        float local_max = -INFINITY;
        float local_min = INFINITY;

        float val = 0.0f;
        float scale, zero = 0.0f;

        const int row_off = lane_id / BLOCK_N;
        const int col_off = lane_id % BLOCK_N;
        int row, col = 0;
        
        for (int i = 0; i < iters; ++i) {
            if (BLOCK_N <= WARP_SIZE) {
                row = start_row + i * warp_rows + row_off;
                col = item * BLOCK_N + col_off;
            } else {
                row = start_row;
                col = item * BLOCK_N + i * WARP_SIZE + col_off;
            }

            if (row < M && col < N) {
                val = static_cast<float>(x[row * stride_xm + col * stride_xn]);
            } else {
                val = 0.0f;
            }

            if (sym) {
                local_max = fmaxf(local_max, fabsf(val));
            } else {
                local_max = fmaxf(local_max, val);
                local_min = fminf(local_min, val);
            }
        }

        if (sym) {
            float block_max = warpReduceMax(local_max);
            scale = fmaxf(block_max * SYM_CONS, 1e-5f);
        } else {
            float block_max = warpReduceMax(local_max);
            float block_min = warpReduceMin(local_min);
            scale = fmaxf((block_max - block_min) * ASYM_CONS, 1e-5f);
            zero = fminf(fmaxf(-rintf(block_min / scale), 0.0f), 15.0f);
        }

        for (int i = 0; i < iters; ++i) {
            if (BLOCK_N <= WARP_SIZE) {
                row = start_row + i * warp_rows + row_off;
                col = item * BLOCK_N + col_off;
            } else {
                row = start_row;
                col = item * BLOCK_N + i * WARP_SIZE + col_off;
            }

            if (row < M && col < N) {
                float val = static_cast<float>(x[row * stride_xm + col * stride_xn]);
                if (sym) {
                    val = rintf(val / scale);
                } else {
                    val = rintf(val / scale) + zero;
                }
                
                out[row * stride_om + col * stride_on] = static_cast<scalar_t>(val);
                out_scale[blockIdx.x * stride_osm + item * stride_osn] = static_cast<scalar_t>(scale);
                if (!sym) {
                    out_zero[blockIdx.x * stride_ozm + item * stride_ozn] = static_cast<scalar_t>(zero);
                }
            }
        }
    }
}

// dispatch
template<typename scalar_t>
void launch_int4_quant_kernel(
    const scalar_t* x,
    scalar_t* out,
    scalar_t* out_scale,
    scalar_t* out_zero,
    int M, int N,
    const int stride_xm, const int stride_xn,
    const int stride_om, const int stride_on,
    const int stride_osm, const int stride_osn,
    const int stride_ozm, const int stride_ozn,
    int block_m, int block_n,
    bool sym,
    cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 256

    if (block_m == 1 && block_n == 32) {
        dim3 grid(M);
        dim3 block(THREADS_PER_BLOCK);
        
        int4_quant_1x32_kernel<scalar_t><<<grid, block, 0, stream>>>(
            x, out, out_scale, out_zero, M, N,
            stride_xm, stride_xn,
            stride_om, stride_on,
            stride_osm, stride_osn,
            stride_ozm, stride_ozn,
            sym
        );
    } else if (block_m == 32 && block_n == 1) {
        dim3 grid(ceil_div(M, block_m));
        dim3 block(THREADS_PER_BLOCK);

        int4_quant_32x1_kernel<scalar_t><<<grid, block, 0, stream>>>(
            x, out, out_scale, out_zero, M, N,
            stride_xm, stride_xn,
            stride_om, stride_on,
            stride_osm, stride_osn,
            stride_ozm, stride_ozn,
            sym
        );
    } else {
        dim3 grid(ceil_div(M, block_m));
        dim3 block(THREADS_PER_BLOCK);
        int4_quant_common_kernel<scalar_t><<<grid, block, 0, stream>>>(
            x, out, out_scale, out_zero, M, N,
            stride_xm, stride_xn,
            stride_om, stride_on,
            stride_osm, stride_osn,
            stride_ozm, stride_ozn,
            block_m, block_n,
            sym
        );
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fake_int4_quant_cuda(
    torch::Tensor& x,
    std::vector<int64_t>& block_size,
    bool sym
) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    
    int M = x.size(0);
    int N = x.size(1);
    int block_m = block_size[0];
    int block_n = block_size[1];

    TORCH_CHECK(block_m > 0 && block_n > 0, "Block sizes must be positive, got block_m=", block_m, ", block_n=", block_n);
    TORCH_CHECK((block_m * block_n) % 32 == 0,
        "block_m * block_n (", block_m * block_n, ") must be divisible by 32. "
        "But got a ", block_m, "x", block_n, " block.");

    auto out = torch::empty_like(x);
    auto out_scale = torch::empty({ceil_div(M, block_m), ceil_div(N, block_n)}, x.options());
    auto out_zero = torch::empty_like(out_scale);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16,
        x.scalar_type(), "int4_quant_cuda", [&] {
        launch_int4_quant_kernel<scalar_t>(
            x.const_data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            out_scale.data_ptr<scalar_t>(),
            out_zero.data_ptr<scalar_t>(),
            M, N,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            out_scale.stride(0), out_scale.stride(1),
            out_zero.stride(0), out_zero.stride(1),
            block_m, block_n,
            sym,
            stream
        );
    });
    
    return std::make_tuple(out, out_scale, out_zero);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fake_int4_quant_cuda", &fake_int4_quant_cuda, "fake INT4 quantization cuda");
}
