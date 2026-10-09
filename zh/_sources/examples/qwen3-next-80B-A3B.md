# Qwen3-Next-80B-A3B

## 环境与数据

按照[快速使用指南](../get_started/quick_start.md)选择 Docker 镜像，并下载 DAPO 训练集和 AIME 评估集。不使用 Docker 时，可通过 `build_conda.sh` 搭建 CUDA 12.9 环境。

预置环境包含用于 Gated Delta Net 的 FLA。CUDA 12 环境还提供可选的 [FlashQLA 后端](../developer_guide/install_flashqla.md)，修改启动脚本时可通过 `--qwen-gdn-backend flashqla` 选择。CUDA 13 环境使用 FLA。

将模型下载到与数据集相同的基础目录：

```bash
export BASE_FOLDER=/root
hf download Qwen/Qwen3-Next-80B-A3B-Thinking \
  --local-dir "${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking"
```

## 权重转换

```bash
cd /root/slime
source scripts/models/qwen3-next-80B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking" \
  --save "${BASE_FOLDER}/Qwen3-Next-80B-A3B-Thinking_torch_dist"
```

模型使用自定义的 Megatron 层配置。GDN 和注意力层的集成方式见[支持 Megatron 之外的模型架构](../advanced/arch-support-beyond-megatron.md)。

## 执行训练

单机八卡：

```bash
cd /root/slime
export BASE_FOLDER=/root
export MASTER_ADDR=127.0.0.1
ACTOR_NUM_NODES=1 CP_SIZE=1 bash scripts/run-qwen3-next-80B-A3B.sh
```

四机、每机八卡时，请确保所有节点使用相同的模型和数据路径，并准备 hostfile：

```bash
cd /root/slime
export BASE_FOLDER=/root
export MASTER_ADDR=your_master_addr
export HOSTFILE=/path/to/hostfile
bash scripts/run-qwen3-next-80B-A3B.sh
```

启动脚本默认使用 `ACTOR_NUM_NODES=4`、`ACTOR_NUM_GPUS_PER_NODE=8` 和 `CP_SIZE=4`。修改节点数或并行方式时，请显式设置这些值。H100/H200 配方是参考配置；适配 Blackwell 时，需要验证所选 kernel 和并行配置。

如果梯度累积占用过多显存，可在启动脚本中用 `--grad-reduce-in-bf16` 替换 `--accumulate-allreduce-grads-in-fp32`。
