# 64xH100 训练 GLM-4.5

这里是使用 64xH100 进行 GLM-4.5 355B RL 训练的示例。

## 环境准备

搭建环境与下载数据的方法可以参考 [示例：Qwen3-4B](./qwen3-4B.md)。

首先需要在多机均可访问到的地址（下记为 `$BASE_DIR`）上下载 GLM-4.5：

```bash
huggingface-cli download zai-org/GLM-4.5 --local-dir $BASE_DIR/GLM-4.5-355B-A32B
```

通过如下方式通过 2 机 16 卡将 huggingface checkpoint 转换为 torch dist 格式：

```bash
cd slime/
source scripts/models/glm4.5-355B-A32B.sh
PYTHONPATH=/root/Megatron-LM/ torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=2 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/GLM-4.5-355B-A32B/ \
   --save $BASE_DIR/GLM-4.5-355B-A32B_torch_dist/
```

其中 `MASTER_ADDR` 为 node0 的 ip，`NODE_RANK` 表示这是第几台机器，这两者就像是在多机 `torchrun` 的时候进行的配置。

## 执行训练

在 node0 运行：

```bash
cd slime/
bash scripts/run-glm4.5-355B-A32B.sh
```

在其他 node 需要通过如下的指令加入 ray 集群：

```bash
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats"
```

或者如果你能获取到所有节点的 ip 列表，例如有一个 mpi hostfie（每一行为 `ip slot=8`），那么可以在 `scripts/run-glm4.5-355B-A32B.sh` 中的 `ray start --head` 指令之后加入如下的指令，从而只需要从 node0 执行训练：

```bash
for WORKER_IP in $(awk '{print $1}' $BASE_DIR/mpi_hostfile); do
  if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then
    continue
  fi
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
done
wait
```

### 参数简介

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4.5-355B-A32B.sh"
```

从 [scripts/models/glm4.5-355B-A32B.sh](../../../scripts/models/glm4.5-355B-A32B.sh) 读取模型的 config。这些 config 都是 megatron 的参数。在使用 megatron 进行训练的时候，megatron 无法从 ckpt 中读取模型 config，需要我们自行配置。我们在 [scripts/models](../../../scripts/models/) 中提供了一些样例。

#### PERF_ARGS

一堆 megatron 的并行参数，只有 `--use-dynamic-batch-size` 与 `--max-tokens-per-gpu` 是 slime 添加的。

megatron 的部分，我们配置了 tp8、pp4、cp2、ep16。

`max_tokens_per_gpu` 是指每张卡最多跑多少 token，在开启 `use_dynamic_batch_size` 之后，会尽可能将一个 batch 内部长短不一的数据拼到 `max_tokens_per_gpu`，从而组成动态的 micro batch size，如果有一条数据长度超过了 `max_tokens_per_gpu`，则自成一条，不会对数据进行截断。在开启 context parallel (CP) 时，会让 CP 张卡去上的数据去共享总长为 `CP * max_tokens_per_gpu` 的 token。

在开启 dynamic_batch_size，会忽略传统的 `micro_batch_size`。

⚠️  slime 总是会通过 data packing 的方法训练模型，并且严格保证 per sample loss 或 per token loss，也就是开启 dynamic batch size 不会对 loss 计算有影响，推荐开启。

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 2
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)
```

#### GRPO_ARGS

目前 slime 这是一些 grpo 相关的参数：

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

如果希望训练时不加载 reference model，需要去掉 `--use-kl-loss` 并设置 `--kl-coef 0.00`（默认值为 0）。

#### OPTIMIZER_ARGS

我们通过了如下几个参数配置了 CPU Adam，用来节省显存。

```bash
OPTIMIZER_ARGS=(
   ...

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)
```

#### SGLANG_ARGS

sglang 所需的参数，这里 `--rollout-num-gpus-per-engine` 基本对应 sglang 的 `tp_size`，除此之外的 sglang 参数均通过添加 `--sglang-` 的前缀来传给 slime。

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-dp-size 4
)
```

#### MISC_ARGS

一些额外的 megatron 配置。注意这里配置了 megatron 的 deepep。

```bash
MISC_ARGS=(
   ...

   # use deepep for megatron
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)
```

## 用 fp8 进行数据生成

开源版本的 GLM-4.5 fp8 ckpt 使用的是 per-channel 量化，目前无法在 sglang 中使用 deepep。我们可以利用 slime 提供的工具转换一个 128x128 per-block 量化的 checkpoint：

```bash
cd slime/
python tools/convert_hf_to_fp8.py \
    --model-dir $BASE_DIR/GLM-4.5-355B-A32B/ \
    --save-dir $BASE_DIR/GLM-4.5-355B-A32B-FP8/ \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

之后将 `--hf-checkpoint` 设置为 `$BASE_DIR/GLM-4.5-355B-A32B-FP8/` 就可以在训练中使用 fp8 进行 rollout 了。

一个样例的 fp8 `SGLANG_ARGS` 为：

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
   --sglang-disable-radix-cache

   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode auto
)
```
