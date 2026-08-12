# Docker release rule

We will publish 2 kinds of docker images:
1. stable version, which based on official sglang release. We will store the patch on those versions.
2. latest version, which aligns to `lmsysorg/sglang:latest`.

current stable version is:
- sglang v0.5.17 (29481685462732237d80d86076d6563e1f658102), megatron dev 1dcf0dafa884ad52ffb243625717a3471643e087

history versions:
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
just release-primary   # publishes both CUDA 12 and CUDA 13 variants below
just release-cu12      # cu129 base: `latest`, `latest-cu129`, `<version>-cu129`
just release-cu13      # cu130 base: `latest-cu130`, `<version>-cu130`
```

`slimerl/slime:latest` tracks the CUDA 12 build. The tag suffixes (`-cu129` /
`-cu130`) match the SGLang base image. `docker/Dockerfile` branches on the base
image's CUDA version; it defaults to the cu129 SGLang base, and the cu130 base
is selected via build args (see `docker/justfile`).

To build a single image directly without publishing:

```bash
# CUDA 12
docker build -f docker/Dockerfile . \
  --build-arg SGLANG_IMAGE_TAG=v0.5.17-cu129 \
  -t slimerl/slime:latest-cu129

# CUDA 13 (Blackwell)
docker build -f docker/Dockerfile . \
  --build-arg DEEPEP_CUDA_ARCH_LIST='10.0 10.3' \
  --build-arg SGLANG_IMAGE_TAG=v0.5.17-cu130 \
  -t slimerl/slime:latest-cu130
```

The following components are pinned and rebuilt in the image:

- Megatron-LM `1dcf0dafa884ad52ffb243625717a3471643e087`, plus
  `docker/patch/<version>/megatron.patch`.
- DeepGEMM `7ad54cbd80ebe24ad36cb5cd3729f37beed420bb` from the
  `zhuzilin/DeepGEMM` fork. It adds batch-invariant FP8 kernels directly to
  the 0.5.17 `v0.1.5` source baseline.
- DeepEP `c5c2b0cfa767b1afeb4d237c4ed7e5ff062a550d` from the
  `zhuzilin/DeepEP` fork. It merges the 0.5.17 baseline and provides aligned
  low-latency FP8 modes for both SGLang 0.5.15.post1 and 0.5.17.

For a non-default GPU architecture list, pass
`--build-arg DEEPEP_CUDA_ARCH_LIST='<torch arch list>'`.
