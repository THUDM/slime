# Docker images and releases

Stable and nightly images use the same Dockerfile. Each build records its
SGLang, Megatron and dependency revisions below.

- Stable images carry an explicit Slime release version, such as `v0.4.0-cu130`.
- Nightly builds use `nightly-dev-<date><suffix>-cu130`. The release command
  updates both `latest-cu130` and `latest` to the CUDA 13 build.
- Existing CUDA 12 images and their versioned patch directories remain available
  for older environments. New builds use CUDA 13.

Current build configuration:

- sglang v0.5.21 (e00930c5489053f26d86b179cee0d087f846acbb), Megatron Core v0.19.2 (4b4acac9a1d28ea6829c8d4f566d75698a21249d)
- Base: `lmsysorg/sglang:v0.5.21`, CUDA 13.0.3, PyTorch 2.13.0.
- Build version: `nightly-dev-20261016a` in `docker/version.txt` (Slime `0.4.0`).

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

Build locally without publishing:

```bash
cd docker
just build
```

Or from the repository root:

```bash
docker build -f docker/Dockerfile . -t slimerl/slime:nightly-dev-20261016a-cu130
```

The local checkout is installed into the image. The existing
`--build-arg SLIME_COMMIT=<revision>` option selects a remote Slime revision.
`SGLANG_IMAGE` and `SGLANG_IMAGE_TAG` can select a custom base; its source must
match `PATCH_VERSION` unless `ENABLE_SGLANG_PATCH=0` is set.

Existing CUDA graph, NSA and Mamba option names remain accepted in Slime's CLI
and per-server YAML overrides. Legacy weight-update endpoints remain available
alongside SGLang's explicit begin/end update sessions, including bucket updates,
`post_process_weights`, and `disable_draft_model` for target-only tensor updates.

Publishing is a separate action: `cd docker && just release` builds and pushes
`<version>-cu130`, `latest-cu130`, and `latest`. `just release-cu13` remains an
alias. The version comes from `docker/version.txt`.

## CUDA 13 on older drivers

The image includes `cuda-compat-13-0` (580.178.04). At container startup, Slime
prepends its user-space driver libraries to `LD_LIBRARY_PATH` on driver
branches 535, 550, 565, 570 and 575. Driver 580 and newer use the host libraries.
This does not install a kernel driver or change the host.

NVIDIA lists CUDA 13.0 forward compatibility for these older driver branches
on supported data-center GPUs (including Hopper). See the
[forward compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).
The `CUDA Version` printed by `nvidia-smi` describes the host driver; it does
not need to change to 13.0 when compatibility is active.

Run commands through the image's default entrypoint:

```bash
docker run --rm --gpus all --ipc=host slimerl/slime:nightly-dev-20261016a-cu130 \
  python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.ones(1, device="cuda"))'
```

`SLIME_CUDA_COMPAT=auto` is the default; `1` forces the compatibility libraries
and `0` disables them. `SLIME_CUDA_COMPAT_PATH` overrides their location.
`docker exec` and a replaced `--entrypoint` bypass startup environment changes;
source `/opt/slime/cuda-compat.sh` before launching Python in that case.
`build_conda.sh` extracts the same package into the environment and enables it
on activation. Hardware-dependent features still need validation on the target
host: run the PD/Mooncake test with the intended driver and GPUs.

## Pinned components

The following components are pinned in the image:

- Megatron-LM / NVIDIA Megatron Core `0.19.2`
  (`4b4acac9a1d28ea6829c8d4f566d75698a21249d`), plus
  `docker/patch/<version>/megatron.patch` and `megatron-sglang-aligned.patch`.
- Transformer Engine `2.18.0`: the official CUDA core wheel matching the base
  image, with PyTorch bindings compiled locally. Optional NCCL EP bindings are
  disabled; Slime uses DeepEP.
- NCCL runtime `2.30.7`, installed after dependency resolution.
- NumPy `2.3.5` and SciPy `1.18.1`, matching the base image and CuPy's CUDA 13 wheel.
- Mooncake transfer engine for CUDA 13 `0.3.11.post1`. This replaces the base's
  `0.3.13`, which returned success with corrupt RDMA payloads after KV memory
  was unloaded and remapped in local testing. The memory re-registration
  regression and GLM colocated PD tests cover this dependency pin.
- Flash Linear Attention `0.5.2`.
- FlashAttention 4 from the SGLang base, FlashAttention 3 `3.0.0` from the
  PyTorch CUDA 13 index for Hopper MLA/context-parallel training, and TileLang `0.1.12`.
  CUDA 13 uses the FLA GDN backend; FlashQLA's older TileLang pin is omitted.
- torch_memory_saver `ac3aaced57ce1f710b9a2ad95151ea152098df66` from
  `zhuzilin/torch_memory_saver`, rebuilt with CUDA hooks and a registered-host
  memory fallback for failed pinned CPU backup allocations.
- Apex `10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4`, rebuilt with C++ and CUDA
  extensions for fused weight gradients and GLM RoPE.
- sgl-router `0.3.2`, using the `zhuzilin/sgl-router` fork wheel from release
  `v0.3.2-9daabcd`.
- SGLang DeepGEMM `0.2.0` (`8965e4a5cded4041c8dbbe8d35b0be31fd8da90b`) from
  `zhuzilin/DeepGEMM`, retaining the batch-invariant APIs from the Slime fork.
- SGLang kernel `0.4.7`, rebuilt after applying the native-kernel patches.
- DeepEP `6845ffd9d59126ec0030c13e0e155935a61e5b5a` from the
  `zhuzilin/DeepEP` `align_fp8_quantization` branch (GLM-5 low-latency alignment).
  This v1 fork replaces the upstream `sgl-deep-ep` wheel; DeepEP v2 is not included.
  SGLang 0.5.21 defaults to `--moe-a2a-backend none`; `deepep` selects v1,
  while v2 requires the separate `deepep_v2` backend. Slime's GLM alignment uses
  v1 on both the SGLang and Megatron sides.

Native extensions default to Hopper/Blackwell (`9.0 10.0 10.3`).
For a non-default DeepEP GPU architecture list, pass
`--build-arg DEEPEP_CUDA_ARCH_LIST='<torch arch list>'`.
