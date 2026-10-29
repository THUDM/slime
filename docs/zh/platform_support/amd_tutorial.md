# AMD ROCm

本教程使用面向 MI300/MI325 的 ROCm 镜像。

## 环境准备

构建镜像并启动容器：

```bash
cd docker
docker build -f Dockerfile.rocm -t rlsys/slime:latest .
docker run --rm -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  rlsys/slime:latest /bin/bash
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

共享的转换脚本在 ROCm 环境下要求 `--use-cpu-initialization`，以便在 CPU 上初始化和保存模型权重。

```bash
cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
PYTHONPATH=/workspace/Megatron-LM-amd_version python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion \
  --use-cpu-initialization \
  --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
  --save /root/Qwen3-4B-Instruct-2507_torch_dist
```

请将 `PYTHONPATH` 替换为镜像中实际的 Megatron-LM 安装路径。

## 训练

```bash
cd /root/slime
SLIME_DIR=/root MODEL_DIR=/root DATA_DIR=/root \
  bash scripts/run-qwen3-4B-amd.sh
```

该配方通过 `--no-gradient-accumulation-fusion` 禁用梯度累积融合。Ray 的 GPU 可见性由 `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES` 控制；调整启动脚本时请保留这些设置。
