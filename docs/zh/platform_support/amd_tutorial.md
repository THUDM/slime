# AMD ROCm

本教程使用面向 MI300/MI325 的 ROCm 镜像。完整的启动脚本说明见[英文教程](https://thudm.github.io/slime/platform_support/amd_tutorial.html)，可用镜像见 [rlsys/slime](https://hub.docker.com/r/rlsys/slime/tags)。遇到平台相关问题时，可联系 [Yusheng Su](https://yushengsu-thu.github.io/)。

## 环境准备

拉取镜像并启动容器：

```bash
docker pull rlsys/slime:latest
docker run --rm -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  rlsys/slime:latest /bin/bash
```

容器中安装当前 slime checkout：

```bash
git clone https://github.com/THUDM/slime.git /root/slime
cd /root/slime
pip install -e . --no-deps
```

也可以使用仓库中的 `docker/Dockerfile.rocm` 构建镜像。

## 模型与数据

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

## 权重转换

共享的转换脚本在 ROCm 环境下要求 `--use-cpu-initialization`，以便在 CPU 上初始化和保存模型权重。

```bash
cd /root/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/workspace/Megatron-LM-amd_version python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion \
  --use-cpu-initialization \
  --hf-checkpoint /root/Qwen3-4B \
  --save /root/Qwen3-4B_torch_dist
```

请将 `PYTHONPATH` 替换为镜像中实际的 Megatron-LM 安装路径。

## 训练

```bash
cd /root/slime
SLIME_DIR=/root MODEL_DIR=/root DATA_DIR=/root \
  bash scripts/run-qwen3-4B-amd.sh
```

该配方通过 `--no-gradient-accumulation-fusion` 禁用梯度累积融合。Ray 的 GPU 可见性由 `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES` 控制；调整启动脚本时请保留这些设置。
