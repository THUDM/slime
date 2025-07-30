# 示例：128xH100 训练 DeepSeek R1

[English](../../en/models/deepseek-r1.md)

这里是使用 128xH100 进行 DeepSeek R1 RL 训练的示例。

我们会使用 bf16 进行训练，128x128 blockwise quant 的 fp8 格式进行推理，模型最大回复长度为 32k，并训练中会使用 dynamic sampling 对数据进行筛选。

在并行上，sglang 方面我们会启用 ep64，开启 dp attention 与 deepep；megatron 部分我们采用 tp8、pp4、ep32、cp4。

⚠️  为了节省 GPU 显存，我们会使用 CPU Adam，每个 node（8xH100）会占用 1.4~1.5B 内存。如果单机的内存不够，可以通过增加 GPU，扩大并行的方式解决。

## 环境准备

搭建环境与下载数据的方法可以参考 [示例：Qwen3-4B](./qwen3-4B.md)。

准备 DeepSeek R1 的 ckpt 则需要使用 [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)。

首先在多机均可访问到的地址（下记为 `$BASE_DIR`）上下载 DeepSeek-R1：

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir $BASE_DIR/DeepSeek-R1
```

DeepSeek-R1 的 huggingface ckpt 为 block-quant 的 fp8 格式，为了转换一个 Megatron 可以加载的 torch dist 格式，需要加载先利用 Pai-Megatron-Patch 转化为 bf16 的 huggingface ckpt：

```bash
cd $BASE_DIR/
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git

export MP_PP0_LAYERS=5
cd Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/deepseek
python fp8_cast_bf16.py --input-fp8-hf-path $BASE_DIR/DeepSeek-R1 --output-bf16-hf-path $BASE_DIR/DeepSeek-R1-bf16/
```

之后我们需要使用 Pai-Megatron-Patch 的多机转换脚本将 bf16 版本的 DeepSeek-R1 转换为 torch dist 格式。具体为在 4 台机器上分别执行：

```bash
cd $BASE_DIR/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
WORLD_SIZE=4 \
RANK=$RANK \
PYTHONPATH=/root/Megatron-LM \
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size 1 --pipeline-model-parallel-size 8 --expert-tensor-parallel-size 1 --expert-model-parallel-size 4 --decoder-first-pipeline-num-layers 7 --decoder-last-pipeline-num-layers 6" \
bash scripts/deepseek_v3/run_32xH20.sh \
   A37B \
   $BASE_DIR/DeepSeek-R1-bf16/ \
   $BASE_DIR/DeepSeek-R1_torch_dist/ \
   false \
   true \
   bf16
```

其中 `MASTER_ADDR` 为 node0 的 ip，`MASTER_PORT` 为某个端口，这两者就像是在进行多机 `torchrun` 的时候进行的配置。`RANK` 表示这是第几台机器。

## 执行训练

在 node0 运行：

```bash
cd slime/
bash scripts/run-deepseek-r1.sh
```

在其他 node 需要通过如下的指令加入 ray 集群：

```bash
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats"
```

或者如果你能获取到所有节点的 ip 列表，例如有一个 mpi hostfie（每一行为 `ip slot=8`），那么可以在 `scripts/run-deepseek-r1.sh` 中的 `ray start --head` 指令之后加入如下的指令，从而只需要从 node0 执行训练：

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
source "${SCRIPT_DIR}/models/deepseek-v3.sh"
```

从 [scripts/models/deepseek-v3.sh](../../../scripts/models/deepseek-v3.sh) 读取模型的 config。这些 config 都是 megatron 的参数。在使用 megatron 进行训练的时候，megatron 无法从 ckpt 中读取模型 config，需要我们自行配置。我们在 [scripts/models](../../../scripts/models/) 中提供了一些样例。


#### CKPT_ARGS

```bash
CKPT_ARGS=(
   # sglang 需要的 hf ckpt，我们也会从这里读 tokenizer
   --hf-checkpoint $BASE_DIR/DeepSeek-R1/
   #--hf-checkpoint $BASE_DIR/DeepSeek-R1-bf16/
   --ref-load $BASE_DIR/DeepSeek-R1_torch_dist/
   # actor 的 load dir，如果是空的，会从 `ref_load` 里面读
   --load $BASE_DIR/DeepSeek-R1_slime/
   --save $BASE_DIR/DeepSeek-R1_slime/
   --save-interval 20
)
```

slime 会根据 `hf_checkpoint` 中的量化配置从而在训练中进行在线量化。例如当前的例子中，我们使用的是 DeepSeek R1 的 fp8 ckpt，那么在进行参数更新的时候，我们会首先将参数进行 blockwise quant，再传至 sglang。

#### ROLLOUT_ARGS

```bash
ROLLOUT_ARGS=(
   # prompt 数据集，每行是个 json
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   # 如果 prompt 的 `input_key` 中是 openai message，
   # 会进行 tokenizer.apply_chat_template(...)
   --apply-chat-template
   # 是否 shuffle 数据
   --rollout-shuffle

   # reward model 类型，
   # slime 提供了很多类型以及用于自定义的 --custom-rm-path
   --rm-type deepscaler

   # 一共要训练多少 rollout
   --num-rollout 3000
   # 一个 rollout 有多少 prompt
   --rollout-batch-size 128
   # 每个 prompt 采多少回复
   # 一个 rollout 会有 rollout_batch_size * n_samples_per_prompt 条
   --n-samples-per-prompt 8
   # rollout sampling param
   --rollout-max-response-len 32768
   --rollout-temperature 0.8

   # 用双倍的 batch size 进行采样，并筛掉 reward 的方差为 0 的 sample
   --over-sampling-batch-size 256
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std


   # 一次 rollout 对应几个训练步
   --num-steps-per-rollout 4
   # 是否在训练时 balance data，可能对速度有好处
   --balance-data
)
```

#### PERF_ARGS

一堆 megatron 的并行参数，只有 `--use-dynamic-batch-size` 与 `--max-tokens-per-gpu` 是 slime 添加的。

megatron 的部分，我们配置了 tp8、pp4、cp4、ep32，由于 DeepSeek-R1 有 61 层，不能被 4 整除，所以我们专门配置最后一个 pp stage 为 13 层。

`max_tokens_per_gpu` 是指每张卡最多跑多少 token，在开启 `use_dynamic_batch_size` 之后，会尽可能将一个 batch 内部长短不一的数据拼到 `max_tokens_per_gpu`，从而组成动态的 micro batch size，如果有一条数据长度超过了 `max_tokens_per_gpu`，则自成一条，不会对数据进行截断。在开启 context parallel (CP) 时，会让 CP 张卡去上的数据去共享总长为 `CP * max_tokens_per_gpu` 的 token。

在开启 dynamic_batch_size，会忽略传统的 `micro_batch_size`。

⚠️  slime 总是会通过 data packing 的方法训练模型，并且严格保证 per sample loss 或 per token loss，也就是开启 dynamic batch size 不会对 loss 计算有影响，推荐开启。

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 4
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1
   --decoder-last-pipeline-num-layers 13

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

sglang 所需的参数，这里 `--rollout-num-gpus-per-engine` 基本对应 sglang 的 `tp_size`，除此之外的 sglang 参数均通过添加 `--sglang-` 的前缀来传给 slime。为了充分利用 sglang 的大 EP 推理能力，我们加上了 ep64、dp_attention dp8、deepep mode auto 等配置。

最后的 `--sglang-server-concurrency` 是 slime 的特有参数，是为了方式同时发给 sglang server 的并发太大打爆 http server，默认为 512。但是我们现在是 8 机一个 server，为了保证每个 dp rank 能有 128 的并发，我们调整为 1024。

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 64
   --sglang-mem-fraction-static 0.5
   --sglang-enable-ep-moe

   # dp attention
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
   --sglang-disable-radix-cache

   # enable deepep for sglang
   --sglang-enable-deepep-moe
   --sglang-deepep-mode auto

   # make every dp rank has 128 concurrency
   --sglang-server-concurrency 1024
)
```

#### MISC_ARGS

一些额外的 megatron 配置。注意这里配置了 megatron 的 deepep。

```bash
MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   # --attention-backend flash

   # use deepep for megatron
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)
```
