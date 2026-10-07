# AMD

## 简介

如果您希望在 AMD Instinct GPU 上运行 slime，本教程将介绍如何使用 Docker
配置开发环境、使用所需的 ROCm 依赖，并运行示例实验。当前 ROCm Docker
镜像支持 AMD Instinct MI300X、MI325X、MI350X 和 MI355X GPU。

## Docker

从 Docker Hub 上的
[amddevhub/slime](https://hub.docker.com/r/amddevhub/slime) 下载预构建的 ROCm
镜像：

```bash
docker pull amddevhub/slime:<tag>  # Choose a suitable tag for your setup.
```

或者在 slime 仓库根目录使用
[Dockerfile.rocm](https://github.com/THUDM/slime/blob/main/docker/Dockerfile.rocm)
自行构建镜像。

```bash
GPU_ARCH=gfx950  # gfx942: MI300X/MI325X; gfx950: MI350X/MI355X.
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm \
  --build-arg GPU_ARCH="${GPU_ARCH}" -t "slime:rocm720-${GPU_ARCH}" .
```

## 快速开始

### 环境配置

将 `IMAGE` 设置为从 Docker Hub 下载的镜像，或者上述构建命令生成的本地
tag，然后启动容器。镜像中已经包含 slime、SGLang 和 Megatron-LM。

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

如需将 slime 更新到最新版本，请运行：

```bash
cd /root/slime
git pull
pip install -e . --no-deps
```

下载模型和数据：

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

### 权重格式转换

将 Hugging Face 权重转换为 Megatron 的 `torch_dist` 格式。在 ROCm 上，
[转换脚本](https://github.com/THUDM/slime/blob/main/tools/convert_hf_to_torch_dist.py)
需要使用 `--use-cpu-initialization` 在 CPU 上构造模型参数。

```bash
cd /root/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --use-cpu-initialization \
  --hf-checkpoint /root/Qwen3-4B \
  --save /root/Qwen3-4B_torch_dist
```

### 启动训练

运行 [AMD Qwen3-4B 训练脚本](https://github.com/THUDM/slime/blob/main/scripts/run-qwen3-4B-amd.sh)：

```bash
cd /root/slime
bash scripts/run-qwen3-4B-amd.sh
```

启动脚本会设置 `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`，使用
`HIP_VISIBLE_DEVICES` 选择 GPU，并将相同的选择同步到 `CUDA_VISIBLE_DEVICES`，
供 slime 映射设备 ID。此处的 `CUDA_VISIBLE_DEVICES` 仅起兼容作用。

## 故障排查

### 使用 GPU Tensor offload 时性能严重下降

Slime 的 [GPU Tensor offload 机制](https://thudm.github.io/slime/zh/blogs/release_v0.1.0.html#offload-gpu-tensor)
使用 [`torch_memory_saver`](https://github.com/fzyzcjy/torch_memory_saver) 和
HIP 虚拟内存管理（VMM）。如果 Linux 内核没有启用
`CONFIG_DMABUF_MOVE_NOTIFY`，amdgpu 会
[从允许的内存放置位置中排除 VRAM](https://github.com/ROCm/amdgpu/blob/rocm-7.2.0/drivers/gpu/drm/amd/amdgpu/amdgpu_dma_buf.c#L294-L333)。
这会迫使 VMM allocation 使用的物理内存驻留在 GTT（系统内存）而非 VRAM 中，
从而导致性能严重下降。
训练时可能看似挂起或最终超时，但实际上仍在极其缓慢地运行。

AMD 的
[IOMMU 指南](https://instinct.docs.amd.com/projects/amdgpu-docs/en/docs-30.30.0/conceptual/iommu.html)
将 `CONFIG_PCI_P2PDMA`、`CONFIG_DMABUF_MOVE_NOTIFY` 和
`CONFIG_HSA_AMD_P2P` 列为在 Linux 内核中启用 P2P DMA 所需要的配置项。
对于 amdgpu DKMS 安装，[构建脚本](https://github.com/ROCm/amdgpu/blob/rocm-7.2.0/drivers/gpu/drm/amd/dkms/dkms-config.sh#L108-L115)
旨在根据前两个配置项生成 `CONFIG_HSA_AMD_P2P`，无需单独配置该选项。

#### 解决方案

系统侧的解决方案包括安装预配置的发行版内核、重新构建 Linux 内核，或者修补
amdgpu DKMS。具体操作请参考
[MORI known issues guide](https://github.com/ROCm/mori/blob/main/.claude/skills/known-issues/SKILL.md#issue-1--vmm-peer-traffic-falls-off-xgmi-onto-pcie--host-memory)。

如果无法更改系统配置，可以使用 slime 侧的临时规避方案：在 GPU Tensor
offload 创建任何 VMM allocation 之前触发 peer access 初始化。在 ROCm 7.2.0 中，
ROCclr 会在[打开 HIP IPC buffer 时启用 peer access](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.0/projects/clr/hipamd/src/hip_memory.cpp#L3389-L3390)，
并将其应用于[已经存在的 allocations](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.0/projects/clr/rocclr/device/device.cpp#L962-L970)。
提前预热可以避免 ROCclr 为 `initialize_model_and_optimizer()` 创建的 VMM
allocations 启用 peer access，从而避免触发上述 GTT 回退。请在
[`MegatronTrainRayActor.init()` 中的此处](https://github.com/THUDM/slime/blob/v0.3.2/slime/backends/megatron_utils/actor.py#L92-L94)
加入以下预热代码，放在分布式进程组初始化完成之后、
`initialize_model_and_optimizer()` 之前：

```python
dist.all_reduce(torch.zeros(1, device="cuda"))
torch.cuda.synchronize()
```

### slime 在 amdgpu 6.16.6 下的多进程 RCCL 延迟异常问题

使用 amdgpu 6.16.6 运行 slime 时，每张 GPU 对应一个进程的执行方式可能会触发 RCCL collective 延迟异常偏高的问题。

该现象可能由 [`8fbaf22`](https://github.com/ROCm/amdgpu/commit/8fbaf22c7acb)
（amdgpu 6.16.6）引入，并由
[`85aa9d6`](https://github.com/ROCm/amdgpu/commit/85aa9d615ec9)
（amdgpu 6.16.13）修复。

请升级到 amdgpu 6.16.13 或更新的 ROCm 支持的驱动版本，然后重启主机。
