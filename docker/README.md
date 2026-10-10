# Docker images and releases

Stable and nightly images use the same Dockerfile. Each build records its
SGLang, Megatron and dependency revisions below.

- Stable images carry an explicit Slime release version tag, for example
  `v0.4.0-cu129` or `v0.4.0-cu130`.
- Nightly builds use `nightly-dev-<date><suffix>-cu129` / `-cu130` tags and
  update the rolling `latest-cu129` and `latest-cu130` tags. The unqualified
  `latest` tracks the CUDA 12 nightly build.

Current build configuration:

- sglang v0.5.15.post1 (0b3bb0cbe31873994c9f989fddfe2f87ca839fdd), Megatron Core v0.19.2 (4b4acac9a1d28ea6829c8d4f566d75698a21249d)
- Build version: `nightly-dev-20261010a` in `docker/version.txt` (Slime `0.4.0`).

Previous build configurations:

- sglang v0.5.15.post1 (0b3bb0cbe31873994c9f989fddfe2f87ca839fdd), megatron dev 1dcf0dafa884ad52ffb243625717a3471643e087
- sglang v0.5.13 (28b095c01005d4a3a2a5b637b7d028b07fba31b2), megatron dev 1dcf0dafa884ad52ffb243625717a3471643e087
- sglang v0.5.12.post1 (5a15cde858ea09b77116212a39356f2fc51b8584), megatron dev 1dcf0dafa884ad52ffb243625717a3471643e087
- sglang v0.5.10.post1 (7c35342c10e201899e22fe2972d40e60da19ff3e), megatron dev 1dcf0dafa884ad52ffb243625717a3471643e087
- sglang v0.5.9 (bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2), megatron dev 3714d81d418c9f1bca4594fc35f9e8289f652862
- sglang v0.5.7 nightly-dev-20260107-dce8b060 (dce8b0606c06d3a191a24c7b8cbe8e238ab316c9), megatron dev 3714d81d418c9f1bca4594fc35f9e8289f652862
- sglang v0.5.6 nightly-dev-20251208-5e2cda61 (5e2cda6158e670e64b926a9985d65826c537ac82), megatron v0.14.0 (23e00ed0963c35382dfe8a5a94fb3cda4d21e133)
- sglang v0.5.5.post1 (303cc957e62384044dfa8e52d7d8af8abe12f0ac), megatron v0.14.0 (23e00ed0963c35382dfe8a5a94fb3cda4d21e133)
- sglang v0.5.0rc0-cu126 (8ecf6b9d2480c3f600826c7d8fef6a16ed603c3f), megatron 48406695c4efcf1026a7ed70bb390793918dd97b

The commands to build and publish:

```bash
cd docker
just release          # Build and publish CUDA 12 first, then CUDA 13.
just release-cu13     # Build and publish only CUDA 13 (Blackwell).
```

`just release` publishes `<version>-cu129`, `latest-cu129`, and `latest`
before building and publishing `<version>-cu130` and `latest-cu130`.
The version comes from `docker/version.txt`.

`slimerl/slime:latest` tracks the CUDA 12 build. The tag suffixes (`-cu129` /
`-cu130`) match the SGLang base image. `docker/Dockerfile` branches on the base
image's CUDA version; it defaults to the cu129 SGLang base, while the cu130 base
is selected via build args (see `docker/justfile`).

To build a single image directly from the repository root without publishing:

```bash
# CUDA 12
docker build -f docker/Dockerfile . \
  --build-arg SGLANG_IMAGE_TAG=v0.5.15.post1-cu129 \
  -t slimerl/slime:latest-cu129

# CUDA 13 (Blackwell)
docker build -f docker/Dockerfile . \
  --build-arg DEEPEP_CUDA_ARCH_LIST='10.0 10.3' \
  --build-arg SGLANG_IMAGE_TAG=v0.5.15.post1-cu130 \
  -t slimerl/slime:latest-cu130
```

The following components are pinned in the image:

- Megatron-LM / NVIDIA Megatron Core `0.19.2`
  (`4b4acac9a1d28ea6829c8d4f566d75698a21249d`), plus
  `docker/patch/<version>/megatron.patch` and `megatron-sglang-aligned.patch`.
- Transformer Engine `2.18.0`: the official CUDA core wheel matching the base
  image, with PyTorch bindings compiled locally. Optional NCCL EP bindings are
  disabled; Slime uses DeepEP.
- NCCL runtime `2.30.7`, installed after dependency resolution.
- Flash Linear Attention `0.5.2`.
- FlashAttention 2 `2.8.3` and FlashAttention 3 `3.0.0` wheels on CUDA 12.
  CUDA 13 keeps the base image's FlashAttention 4 kernels.
- FlashQLA `821fd9d37ede18fdc2a4e707fefe3770bfc32e58` with TileLang `0.1.9`
  on CUDA 12. CUDA 13 uses the default FLA backend and keeps SGLang's TileLang
  `0.1.11`.
- torch_memory_saver `4d525cf378fdbfe7eb044909b8e66e112a508624` from
  `zhuzilin/torch_memory_saver`, rebuilt with CUDA hooks.
- Apex `10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4`, rebuilt with C++ and CUDA
  extensions for fused weight gradients and GLM RoPE.
- sgl-router `0.3.2`, using the `zhuzilin/sgl-router` fork wheel from release
  `v0.3.2-9daabcd`.
- DeepGEMM `b38a77cd193cf38f670caae192310521d24343be` from the
  `zhuzilin/DeepGEMM` batch-invariant branch, rebuilt as an SGLang-compatible wheel.
- DeepEP `6845ffd9d59126ec0030c13e0e155935a61e5b5a` from the
  `zhuzilin/DeepEP` `align_fp8_quantization` branch (GLM-5 low-latency alignment).

For a non-default GPU architecture list, pass
`--build-arg DEEPEP_CUDA_ARCH_LIST='<torch arch list>'`.
