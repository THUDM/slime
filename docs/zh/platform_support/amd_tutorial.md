# AMD ROCm

本教程介绍如何使用 ROCm 镜像运行 slime。

支持的 GPU 架构：

- MI300X / MI325X（gfx942）
- MI350X / MI355X（gfx950）

本教程以 gfx942 架构为例进行讲解，gfx950 的使用流程与之相同。如果使用 gfx950，只需在构建镜像时将 `--build-arg GPU_TARGET` 设置为 `gfx950`，并将后续命令中镜像标签的后缀 `_gfx942` 相应替换为 `_gfx950`。

## 环境准备

构建镜像（需在 slime 仓库根目录下执行）：

```bash
docker build --build-arg GPU_TARGET=gfx942 -t amdagi/slime:rocm7.15_torch2.12_py3.12_gfx942 -f docker/Dockerfile.rocm .
```

或者直接拉取预构建镜像：

```bash
docker pull amdagi/slime:rocm7.15_torch2.12_py3.12_gfx942
```

启动容器：

```bash
docker run --rm -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  amdagi/slime:rocm7.15_torch2.12_py3.12_gfx942 /bin/bash
```

## 模型与数据

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

## 权重转换

在执行权重转换或训练之前，先运行以下命令：

```bash
cat > /root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py <<'EOF'
import importlib
import sys

sys.modules[__name__] = importlib.import_module("nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async")
EOF
```

否则权重转换和训练都会报错：

```text
ModuleNotFoundError: No module named 'megatron.core.dist_checkpointing.strategies.filesystem_async'
```

在 ROCm 环境下，通用的权重转换脚本需要添加 `--use-cpu-initialization` 参数，以便在 CPU 上初始化并保存模型权重。

```bash
cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/workspace/Megatron-LM-amd_version python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion \
  --use-cpu-initialization \
  --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
  --save /root/Qwen3-4B-Instruct-2507_torch_dist
```

请将 `PYTHONPATH` 替换为镜像中 Megatron-LM 的实际安装路径。

## 训练

```bash
cd /root/slime
SLIME_DIR=/root MODEL_DIR=/root DATA_DIR=/root \
  bash scripts/run-qwen3-4B-amd.sh
```

该脚本通过 `--no-gradient-accumulation-fusion` 禁用了梯度累积融合。Ray 的 GPU 可见性由 `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES` 共同控制，修改启动脚本时请保留这些设置。
