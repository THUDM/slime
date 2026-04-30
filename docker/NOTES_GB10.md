# slime on NVIDIA DGX Spark (GB10) — Porting Notes

Target platform:
- Chip: NVIDIA GB10 (Grace + consumer Blackwell, SM 12.1 / sm_121a)
- Arch: aarch64
- OS: Ubuntu 24.04, NVIDIA driver 580.142 (CUDA 13.x forward-compat)
- Unified memory: 128 GB (CPU+GPU shared)

## Why a new Dockerfile is needed

slime's published Docker images (`slimerl/slime:*`) are x86_64-only. The arm64 base
it derives from (`slimerl/sglang:v0.5.9`) ships with CUDA 12.9 and stock PyTorch
2.9.1+cu129 — neither of which knows about the `sm_121a` target used by GB10:
the Triton PTX pipeline crashes with `ptxas fatal: Value 'sm_121a' is not defined`.

The `ENABLE_CUDA_13=1` branch in the upstream Dockerfile is aimed at GB200/GB300
(sm_100a) and still uses an x86-only router wheel and amd64 base, so it does not
directly apply to GB10.

This port rebases slime on `nvcr.io/nvidia/vllm:26.03-py3` (arm64), which ships:
- CUDA 13.2 (ptxas understands sm_121a ✅)
- PyTorch 2.11.0a0 compiled with `compute_120` (PTX forward-compat to sm_121 ✅)
- Triton 3.6.0 (verified: Triton kernel JITs on GB10 ✅)
- flash-attn 2.7.4.post1 preinstalled

### Base image pinning (for reproducibility)

Pull from NGC and verify digest:

```bash
docker pull nvcr.io/nvidia/vllm:26.03-py3
docker inspect nvcr.io/nvidia/vllm:26.03-py3 --format '{{range .RepoDigests}}{{.}}{{end}}'
# Expected digest:
#   nvcr.io/nvidia/vllm@sha256:13e327dad79e6e417f6687fec2ba76b0386d597082ec0ee003c1e964ec6ad0e7
```

All downstream steps pin this digest. Product page:
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=26.03-py3

## Known blockers and their resolutions

| # | Blocker | Root cause | Resolution |
|---|---------|-----------|------------|
| 1 | `ptxas fatal: sm_121a` | CUDA 12.9 ptxas predates sm_121a (added in CUDA 13.0) | Rebase on NGC CUDA 13.2 image |
| 2 | `libnvrtc.so.12` missing from sgl_kernel wheel | Published sgl_kernel wheel is cu12x | Use cu130 wheel OR build from source |
| 3 | `sgl_kernel/sm100/...abi3.so: undefined symbol _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib` | Wheel built against stock libtorch, NGC libtorch has different C++ ABI | Build sgl_kernel from source against NGC torch |
| 4 | sgl_kernel arm64 wheels only build sm_100 variant | sgl-project CI doesn't target GB10 | Build sm_121 variant from source |
| 5 | sgl-kernel source build: cicc OOM on sm_90+100+103+110+120+121 concurrent compile | Cutlass FP8 templates × 7 arches × 12 parallel → >128GB | `SGL_KERNEL_GB10_ONLY=ON` CMake option (see `docker/patch/gb10/sgl-kernel-arch.patch`), drops to sm_120a+121a |
| 6 | TE 2.10 build: `CUDNN::cudnn_engines_precompiled` target not found | NGC vLLM image ships only runtime-compiled cuDNN libs | Install `nvidia-cudnn-cu13==9.20.0.48` pypi wheel (has the precompiled engines), symlink into `/usr/lib/aarch64-linux-gnu/` |
| 7 | TE 2.10 build: `nvtx3/nvToolsExt.h: No such file` | CUDA 13 dropped bundled NVTX headers; `nvidia-nvtx-cu13` pypi wheel is an empty 0.0.1 placeholder | Clone `NVIDIA/NVTX` github, copy `c/include/nvtx3/*` to `/usr/local/cuda/include/nvtx3/` |
| 8 | TE 2.10 build: `ptx.cuh` static_assert "Compile for smXXXf instead of smXXX" | Blackwell family-specific TMA features require CUDA 13's new `f` (family-specific) arch suffix, not plain integer | Set `NVTE_CUDA_ARCHS="120f;121f"` — but CMake ≤3.31 rejects `f` suffix |
| 9 | CMake 3.31 rejects `CMAKE_CUDA_ARCHITECTURES=120f;121f` | `f` suffix for CUDA 13 Blackwell family is only supported in CMake ≥4.0 | Upgrade to `cmake==4.3.1` via pip (must override NGC `/etc/pip/constraint.txt` with `PIP_CONSTRAINT=`) and set `CMAKE_POLICY_VERSION_MINIMUM=3.5` for old bundled deps |
| 10 | TE 2.10 build: `cuda_profiler_api.h: No such file` | CUDA 13 removed the public header for `cudaProfilerStart/Stop`; the symbols still exist in `libcudart.so.13`. TE's 3 `fused_softmax` TUs `#include` the header but don't call the APIs | Install a 20-line shim header at `/usr/local/cuda/include/cuda_profiler_api.h` declaring the two functions extern. Stored as `docker/patch/gb10/cuda_profiler_api.h` |
| 11 | slime `train.py --help`: `'tuple' object has no attribute 'strip'` | Typo in `slime/utils/arguments.py:1073`: `help=("string",)` (trailing comma → tuple) instead of `help=("string")` | Remove trailing comma — simple one-line slime fix, upstream-able |
| 12 | `sglang_router` x86_64-only wheel from `zhuzilin/sgl-router` fork | slime Dockerfile pins `zhuzilin/sgl-router` release (no arm64 builds); slime's `'slime' in version` assertion is only in `wandb_utils.py` (non-critical path) | Install upstream `sglang-router==0.3.2` from PyPI (has arm64 wheel). Accept wandb path fallback |
| 13 | `antlr4-python3-runtime==4.13.2` → `Could not deserialize ATN with version 3` | Omegaconf's bundled grammar was generated with antlr 4.9 serialized format; runtime 4.13 only reads format v4 | Pin `antlr4-python3-runtime==4.9.3` |
| 14 | `megatron.training` not importable after `pip install -e Megatron-LM` | Megatron-LM's setup.py only packages `megatron-core`; `megatron.training`, `megatron.rl`, `megatron.legacy` are sibling dirs meant to be on `PYTHONPATH` | `export PYTHONPATH=/root/src/Megatron-LM:$PYTHONPATH` (slime docs confirm this) |
| 15 | `libz3.so` missing for tilelang | tilelang uses Z3 SMT solver for autoscheduling; NGC vllm base doesn't include libz3 | `apt-get update && apt-get install -y libz3-dev` (libz3-4 package alias needs update first) |

## Build journal

### sgl-kernel OOM during build → arch whitelist patch

First build attempt on GB10 with stock sgl-kernel v0.5.9 CMake: cicc
(NVIDIA's CUDA frontend compiler) was OOM-killed while compiling
`csrc/gemm/fp8_gemm_kernel.cu`, `fp8_blockwise_gemm_kernel.cu`, and
`nvfp4_scaled_mm_kernels.cu`.

Root cause: for `CUDA_VERSION >= 13.0 && aarch64`, sgl-kernel unconditionally
emits seven gencodes per TU — `sm_90, sm_90a, sm_100a, sm_103a, sm_110a,
sm_120a, sm_121a`. Cutlass FP8/NVFP4 gemm template instantiation uses
~10–15 GB of RAM per TU per arch. Combined with 12 parallel nvcc jobs, peak
memory exceeded the 128 GB unified memory limit.

Only `sm_121a` (and `sm_120a` as PTX fallback) actually runs on GB10. Hopper
(sm_90a) and datacenter Blackwell (sm_100a/sm_103a) binaries are dead weight.

Fix: `docker/patch/gb10/sgl-kernel-arch.patch` adds a CMake option
`SGL_KERNEL_GB10_ONLY`. When set, the other gencodes and FA3 (sm_90a-only)
are skipped. Default OFF preserves upstream behavior. See the patch python
script `docker/patch/gb10/patch_sgl_kernel.py` for a programmatic apply.

### sgl-kernel build success (M1)

With `SGL_KERNEL_GB10_ONLY=ON`, `CMAKE_BUILD_PARALLEL_LEVEL=8`, and
`SGL_KERNEL_COMPILE_THREADS=2`, the build completed in 24 minutes with RAM
never exceeding ~25 GB. Produced `sgl_kernel-0.3.21-cp310-abi3-linux_aarch64.whl`
(74 MB), ABI-compatible with NGC libtorch 2.11.0a0 (verified via clean import
and functional rmsnorm kernel on GB10).

## Work items

- [x] M1: Build sgl_kernel from source for sm_121 + NGC torch ABI
- [x] M2: TransformerEngine 2.10 (sm_121f), apex, Megatron-LM all built + verified
- [x] M3: slime installed with megatron.patch applied; `train.py --help` prints 3714-line arg list
- [x] M4: End-to-end RL loop (Qwen2.5-0.5B + dapo-math-17k + GRPO colocated on 1 GB10 GPU) runs clean in 2m10s

Reproducible image: `slime-gb10:m4-success` (36.3 GB, committed post-smoke)
- Step 0 train metrics logged; full rollout → reward → policy-update → weight-sync cycle confirmed.
