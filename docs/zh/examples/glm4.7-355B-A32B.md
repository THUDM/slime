# 64xH100 训练 GLM-4.7

## 环境准备

搭建环境与下载数据的方法与 Qwen3-4B 模型相同，可以参考 [示例：Qwen3-4B](qwen3-4B.md)，将文中 Qwen3-4B 的部分替换为 GLM-4.7 即可。

### 前置条件

GLM-4.7 使用 slime 标准 Docker 环境即可。多机启动前，请确保所有机器都能访问同一个 `$BASE_DIR` 路径，并在启动 Ray worker 前先取消代理：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

### 下载模型

```bash
hf download zai-org/GLM-4.7 --local-dir $BASE_DIR/GLM-4.7-355B-A32B
```

### 转换 Checkpoint

可以用如下方法把 Hugging Face checkpoint 转换为 torch_dist 格式（2 机 x 8 卡）：

```bash
cd /root/slime
pip install -e . --no-deps
source scripts/models/glm4.5-355B-A32B.sh
PYTHONPATH=/root/Megatron-LM/ torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=2 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/GLM-4.7-355B-A32B/ \
   --save $BASE_DIR/GLM-4.7-355B-A32B_torch_dist/
```

其中 `MASTER_ADDR` 是 node0 的 IP，`NODE_RANK` 表示当前机器的编号，配置方式与普通多机 `torchrun` 一致。

## 执行训练

从 node0 执行训练脚本：

```bash
cd /root/slime
export BASE_DIR=/shared/path  # 所有节点都能访问的共享路径
bash scripts/run-glm4.7-355B-A32B.sh
```

### 参数简介

这里我们简单介绍一下 [run-glm4.7-355B-A32B.sh](https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-355B-A32B.sh) 中的关键部分。

#### MoE 配置

GLM-4.7 是一个 MoE（混合专家）模型，包含 160 个路由专家（top-8 激活）和共享专家。模型共 92 层：3 层 dense + 89 层 MoE。

1. 为了支持在 64xH100 环境中运行 GLM-4.7，我们开启 Megatron 的 CPU Adam 来节省显存：

   ```bash
   OPTIMIZER_ARGS=(
      ...
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
   ```

2. 在 Megatron 中开启 MoE 优化。当前 64xH100 示例使用 TP=8、PP=4、CP=2、EP=16：

   ```bash
   PERF_ARGS=(
      --tensor-model-parallel-size 8
      --sequence-parallel
      --pipeline-model-parallel-size 4
      --context-parallel-size 2
      --expert-model-parallel-size 16
      --expert-tensor-parallel-size 1
      ...
      --use-dynamic-batch-size
      --max-tokens-per-gpu 16384
   )
   ```

3. 在 SGLang 中开启带 DP attention 的 MoE 优化：

   ```bash
   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 32
      --sglang-mem-fraction-static 0.7
      --sglang-enable-dp-attention
      --sglang-dp-size 4
      --sglang-ep-size 32
      --sglang-enable-dp-lm-head
      --sglang-moe-dense-tp-size 1
      ...
   )
   ```

#### MTP 投机解码（推理加速）

GLM-4.7 包含 MTP（Multi-Token Prediction）层，可以在推理阶段用于投机解码，加速 rollout 生成。启用方法是在 `SGLANG_ARGS` 中加入：

```bash
SGLANG_ARGS=(
   ...
   # MTP 投机解码 (EAGLE)
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
)
```

这样 SGLang 就会使用模型自带的 MTP 层作为 EAGLE 风格投机解码的 draft model。

> ⚠️ **注意**：投机解码会额外占用 GPU 显存。如果遇到 OOM，可以尝试降低 `--sglang-mem-fraction-static` 或暂时关闭投机解码。

#### MTP 训练

slime 也支持在 GLM-4.7 上将 MTP 层与主模型联合训练。启用时，相关参数如下：

```bash
# 在模型配置中添加 MTP 层数
MODEL_ARGS+=(--mtp-num-layers 1)

# 启用 MTP 训练
MTP_ARGS=(
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

- `--mtp-num-layers 1`：告知 Megatron 从 checkpoint 中加载 MTP 层。
- `--enable-mtp-training`：启用 MTP 层的梯度计算；不设置时 MTP 层会被加载但保持冻结。
- `--mtp-loss-scaling-factor 0.2`：MTP loss 相对主策略 loss 的权重，默认值为 0.2。

> **注意**：GLM-4.7 的 MTP 训练依赖 `GLM4MoEBridge`（位于 `slime_plugins/mbridge/glm4moe.py`）在 HuggingFace 与 Megatron 格式之间正确映射普通层和 MTP 层权重。

#### 多机支持

这个示例本身就是多机训练配置。启动前请确认：

- 模型权重和数据集放在所有节点都能访问到的路径；
- `MASTER_ADDR` 设置为所有节点都能访问到的地址；
- 在启动 Ray worker 前先取消代理；
- 提供一个 `HOSTFILE` 列出 worker IP（每行一个），并在启动前 `export HOSTFILE=/path/to/hostfile`；
- 并行度需要成套调整。默认示例使用 TP=8、PP=4、EP=16、CP=2，rollout 侧则使用 32 张卡 / engine + SGLang DP attention。

如果 rollout GPU 数与 expert 数（160）之间不能整除，可以通过 `--sglang-ep-num-redundant-experts` 增加冗余 expert。

## FP8 Rollout

开源版 GLM-4.7 的 FP8 checkpoint 使用的是 per-channel 量化，目前无法在 SGLang 中直接启用 DeepEP。可以利用 slime 自带工具将其转换为 128x128 的 per-block FP8 checkpoint：

```bash
cd /root/slime
python tools/convert_hf_to_fp8.py \
    --model-dir $BASE_DIR/GLM-4.7-355B-A32B/ \
    --save-dir $BASE_DIR/GLM-4.7-355B-A32B-FP8/ \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

随后把 `--hf-checkpoint` 改成 `$BASE_DIR/GLM-4.7-355B-A32B-FP8/` 即可开启 FP8 rollout。

一个可参考的 FP8 `SGLANG_ARGS` 配置如下：

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-dp-size 32
   --sglang-ep-size 32
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)

   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4

   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode auto
)
```
