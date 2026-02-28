# 8×H100 训练 GLM-4.7-Flash

## 环境准备

搭建环境、数据与 ckpt 转换均与 Qwen3-4B 模型相同，可以参考 [示例：Qwen3-4B](qwen3-4B.md)，将文中 Qwen3-4B 的部分转换为 GLM-4.7-Flash 即可。

### 下载模型

```bash
hf download THUDM/GLM-4.7-Flash --local-dir /root/GLM-4.7-Flash
```

### 转换 Checkpoint

可以用如下方法把 Hugging Face checkpoint 转化为 torch_dist 格式：

```bash
cd /root/slime
pip install -e . --no-deps
source scripts/models/glm4.7-30B-A3B.sh
PYTHONPATH=/root/Megatron-LM/ torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/GLM-4.7-Flash/ \
   --save /root/GLM-4.7-Flash_torch_dist/
```

## 执行训练

执行训练：

```bash
cd /root/slime
bash scripts/run-glm4.7-30B-A3B-8gpus.sh
```

### 参数简介

这里我们简单介绍一下脚本 [run-glm4.7-30B-A3B-8gpus.sh](https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-30B-A3B-8gpus.sh) 中的关键部分。

#### MoE 配置

GLM-4.7-Flash 是一个 MoE（混合专家）模型，包含 64 个路由专家（top-4 激活）和 1 个共享专家。共 47 层：1 层 dense 层 + 46 层 MoE 层。

1. 为了支持在 8×H100 环境中运行 GLM-4.7-Flash，我们需要开启 Megatron 的 CPU Adam 以节省显存：

   ```bash
   OPTIMIZER_ARGS=(
      ...
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
   ```

2. 开启 Megatron 支持的 MoE 优化，单机 8×H100 配置为 TP=1, EP=8：

   ```bash
   PERF_ARGS=(
      --tensor-model-parallel-size 1
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --expert-model-parallel-size 8
      --expert-tensor-parallel-size 1
      ...
   )
   ```

3. 开启 SGLang 支持的 MoE 优化，使用 DP attention：

   ```bash
   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 8
      --sglang-mem-fraction-static 0.7
      --sglang-enable-dp-attention
      --sglang-dp-size 8
      --sglang-enable-dp-lm-head
      --sglang-moe-dense-tp-size 1
      ...
   )
   ```

#### MTP 投机解码（推理加速）

GLM-4.7-Flash 包含 1 层 MTP（Multi-Token Prediction）层，可用于推理时的投机解码来加速 rollout 生成。要启用此功能，在 `SGLANG_ARGS` 中添加以下配置：

```bash
SGLANG_ARGS=(
   ...
   # MTP 投机解码 (EAGLE)
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3
)
```

这会让 SGLang 使用模型的 MTP 层作为 EAGLE 风格投机解码的 draft 模型。MTP 层预测多个未来 token，SGLang 并行验证它们，从而加速生成。

> ⚠️ **注意**：投机解码会占用额外的 GPU 显存。如果遇到 OOM 问题，可以尝试降低 `--sglang-mem-fraction-static` 或关闭投机解码。

#### MTP 训练

slime 也支持将 MTP 层与主模型联合训练，适用于已实现 MTP 权重转换的模型（如 MiMo、GLM-4.5）。启用时，相关参数如下：

```bash
# 在模型配置中添加 MTP 层数
MODEL_ARGS+=(--mtp-num-layers 1)

# 启用 MTP 训练
SPEC_ARGS=(
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

- `--mtp-num-layers 1`：告知 Megatron 从 checkpoint 中加载 MTP 层。
- `--enable-mtp-training`：启用 MTP 层的梯度计算。不设置此标志时，MTP 层会被加载但冻结。
- `--mtp-loss-scaling-factor 0.2`：MTP loss 相对于主策略 loss 的权重，默认为 0.2。

> ⚠️ **注意**：GLM-4.7-Flash 的 MTP 训练目前尚不支持，因为 deepseek_v3 的 checkpoint bridge 尚未实现 MTP 权重转换（上游 mbridge 中标注为 `# TODO: mtp`）。但推理时的投机解码仍然可用 — SGLang 会内部处理 MTP 层。
>
> 对于完整支持 MTP 训练的模型（如 MiMo），可参考 `scripts/run-mimo-7B-rl-eagle.sh`。

### 多机支持

对于多机训练（例如 2×8 H100），使用多机脚本：

```bash
cd /root/slime
export BASE_DIR=/shared/path  # 所有节点都可以访问的路径
bash scripts/run-glm4.7-30B-A3B.sh
```

对于多机环境，需要进行如下修改：

- 将训练模型、数据放在所有机器都可以访问到的路径上；
- 设置各台机器都可以访问到的 `MASTER_ADDR`；
- 去掉 CPU Adam 相关的配置，因为使用了 distributed optimizer，多机环境下 optimizer 的显存占比会明显下降。
- 调整并行度：例如 TP=4, PP=2, EP=8, CP=2。

当总卡数并不能被 expert 总数（64）乘除时，可以使用 `--sglang-ep-num-redundant-experts` 来增加冗余的 expert。例如对于 24 卡的场景：

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 24
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 24
   --sglang-enable-dp-attention
   --sglang-dp-size 3
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
   --sglang-ep-num-redundant-experts 16
)
```
