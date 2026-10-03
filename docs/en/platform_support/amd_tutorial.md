# AMD

## Introduction

If you are running slime on AMD Instinct GPUs, this tutorial explains how to set up the development environment with Docker, use the required ROCm dependencies, and run an example experiment. The current ROCm Docker image supports AMD Instinct MI300X, MI325X, MI350X, and MI355X GPUs.

## Docker

Download a prebuilt ROCm image from
[amddevhub/slime](https://hub.docker.com/r/amddevhub/slime) on Docker Hub:

```bash
docker pull amddevhub/slime:<tag>  # Choose a suitable tag for your setup.
```

Alternatively, build the image from the slime repository root with
[Dockerfile.rocm](https://github.com/THUDM/slime/blob/main/docker/Dockerfile.rocm).

```bash
GPU_ARCH=gfx950  # gfx942: MI300X/MI325X; gfx950: MI350X/MI355X.
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm \
  --build-arg GPU_ARCH="${GPU_ARCH}" -t "slime:rocm720-${GPU_ARCH}" .
```

## Quick Start

### Environment Setup

Set `IMAGE` to a compatible image downloaded from Docker Hub or to the local
tag created by the build command above, then start the container. The image
already includes slime, SGLang, and Megatron-LM.

```bash
IMAGE=YOUR_IMAGE_TAG
CONTAINER_NAME=YOUR_CONTAINER_NAME

docker run -d \
  --name "${CONTAINER_NAME}" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt=seccomp=unconfined \
  --shm-size=128g \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  -p 8265:8265 \
  "${IMAGE}" sleep infinity

docker exec -it "${CONTAINER_NAME}" /bin/bash
```

To update slime to the latest version, run:

```bash
cd /root/slime
git pull
pip install -e . --no-deps
```

Download the model and data:

```bash
# hf checkpoint
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# train data
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# eval data
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

### Checkpoint Format Conversion

Convert the Hugging Face checkpoint to Megatron's `torch_dist` format. On
ROCm, the [converter](https://github.com/THUDM/slime/blob/main/tools/convert_hf_to_torch_dist.py)
requires `--use-cpu-initialization` to construct the model parameters on the
CPU.

```bash
cd /root/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --use-cpu-initialization \
  --hf-checkpoint /root/Qwen3-4B \
  --save /root/Qwen3-4B_torch_dist
```

### Run Training

Execute the [AMD Qwen3-4B training script](https://github.com/THUDM/slime/blob/main/scripts/run-qwen3-4B-amd.sh):

```bash
cd /root/slime
bash scripts/run-qwen3-4B-amd.sh
```

The launcher sets `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`, uses
`HIP_VISIBLE_DEVICES` to select GPUs, and mirrors that selection to
`CUDA_VISIBLE_DEVICES` for slime's device-ID mapping. `CUDA_VISIBLE_DEVICES`
is only a compatibility input here.

## Troubleshooting

### Severe slowdown with GPU tensor offload

Slime's [GPU tensor offload mechanism](https://thudm.github.io/slime/zh/blogs/release_v0.1.0.html#offload-gpu-tensor)
uses [`torch_memory_saver`](https://github.com/fzyzcjy/torch_memory_saver) and
HIP virtual memory management (VMM). On Linux kernels without
`CONFIG_DMABUF_MOVE_NOTIFY`, amdgpu
[excludes VRAM from the allowed placement](https://github.com/ROCm/amdgpu/blob/rocm-7.2.0/drivers/gpu/drm/amd/amdgpu/amdgpu_dma_buf.c#L294-L333).
This forces the physical memory for the VMM allocations to reside in GTT
(system memory) rather than VRAM, causing a severe slowdown. Training may
appear to hang or eventually time out because it is progressing extremely
slowly.

AMD's
[IOMMU guide](https://instinct.docs.amd.com/projects/amdgpu-docs/en/docs-30.30.0/conceptual/iommu.html)
documents `CONFIG_PCI_P2PDMA`, `CONFIG_DMABUF_MOVE_NOTIFY`, and
`CONFIG_HSA_AMD_P2P` as the Linux kernel options used to enable peer-to-peer
DMA. For an amdgpu DKMS installation, the
[build script](https://github.com/ROCm/amdgpu/blob/rocm-7.2.0/drivers/gpu/drm/amd/dkms/dkms-config.sh#L108-L115)
is intended to derive `CONFIG_HSA_AMD_P2P` from the first two options instead
of requiring it to be configured separately.

#### Solutions

For system-side solutions—installing a preconfigured distribution kernel,
rebuilding the Linux kernel, or patching amdgpu DKMS—follow the
[MORI known issues guide](https://github.com/ROCm/mori/blob/main/.claude/skills/known-issues/SKILL.md#issue-1--vmm-peer-traffic-falls-off-xgmi-onto-pcie--host-memory).

If the system configuration cannot be changed, a slime-side workaround is to
trigger peer-access initialization before GPU tensor offload creates any VMM
allocations.
In ROCm 7.2.0, ROCclr
[enables peer access when opening a HIP IPC buffer](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.0/projects/clr/hipamd/src/hip_memory.cpp#L3389-L3390)
and applies it to
[existing allocations](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.0/projects/clr/rocclr/device/device.cpp#L962-L970).
Warming up first prevents ROCclr from enabling peer access for the VMM
allocations created by `initialize_model_and_optimizer()`, thereby avoiding the
GTT fallback described above. Add the following warmup at
[this point in `MegatronTrainRayActor.init()`](https://github.com/THUDM/slime/blob/v0.3.2/slime/backends/megatron_utils/actor.py#L92-L94),
after the distributed process group has been initialized and immediately before
`initialize_model_and_optimizer()`:

```python
dist.all_reduce(torch.zeros(1, device="cuda"))
torch.cuda.synchronize()
```

### Multi-process RCCL latency issue in slime with amdgpu 6.16.6

When running slime with amdgpu 6.16.6, its one-process-per-GPU execution may
trigger abnormally high RCCL collective latency.

This behavior may have been introduced by
[`8fbaf22`](https://github.com/ROCm/amdgpu/commit/8fbaf22c7acb) (amdgpu 6.16.6)
and fixed by
[`85aa9d6`](https://github.com/ROCm/amdgpu/commit/85aa9d615ec9) (amdgpu 6.16.13).

Upgrade to amdgpu 6.16.13 or a newer ROCm-supported driver, then reboot the
host.
