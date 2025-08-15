# 示例：Qwen3-4B

[English](../../en/models/qwen3-4B.md)

## 环境准备

拉取 `zhuzilin/slime:latest` 镜像后，用如下方式初始化镜像环境：

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime/
pip install -e .
```

下载模型与数据：

```bash
# hf checkpoint
huggingface-cli download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B

# train data
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# eval data
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

将 huggingface checkpoint 转换成 megatron 可以加载的 huggingface checkpoint：

```bash
# mcore checkpoint
cd /root/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

## 执行训练

执行训练：

```bash
cd /root/slime
bash scripts/run-qwen3-4B.sh
```

### 参数简介

这里我们简单介绍一下脚本 [run-qwen3-4B.sh](../../../scripts/run-qwen3-4B.sh) 中的各个组成部分：

#### MODEL_ARGS

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
```

从 [scripts/models/qwen3-4B.sh](../../../scripts/models/qwen3-4B.sh) 读取模型的 config。这些 config 都是 megatron 的参数。在使用 megatron 进行训练的时候，megatron 无法从 ckpt 中读取模型 config，需要我们自行配置。我们在 [scripts/models](../../../scripts/models/) 中提供了一些样例。

⚠️  注意检查模型文件中的 `--rotary-base` 等配置是否对应你当前训练模型的配置，因为同一个模型结构的不同模型可能有不同的取值。在这种情况下，你可以在导入模型参数后在脚本里进行覆盖，例如：

```bash
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

MODEL_ARGS += ( --rotary-base 10000 )
```

#### CKPT_ARGS

```bash
CKPT_ARGS=(
   # sglang 需要的 hf ckpt，我们也会从这里读 tokenizer
   --hf-checkpoint /root/Qwen3-4B
   # reference model 的 ckp
   --ref-load /root/Qwen3-4B_torch_dist
   # actor 的 load dir，如果是空的，会从 `ref_load` 里面读
   --load /root/Qwen3-4B_slime/
   --save /root/Qwen3-4B_slime/
   --save-interval 20
)
```

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
   --rollout-batch-size 32
   # 每个 prompt 采多少回复
   # 一个 rollout 会有 rollout_batch_size * n_samples_per_prompt 条
   --n-samples-per-prompt 8
   # rollout sampling param
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   # 一次 rollout 对应几个训练步
   --num-steps-per-rollout 1
   # 是否在训练时 balance data，可能对速度有好处
   --balance-data
)
```

#### EVAL_ARGS

eval 的时候基本上是会继承所有 rollout 的参数，但是我们提供了一些可以 rollout 配置覆盖的参数，从而实现训练和 eval 用不同的采样策略。

```bash
EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)
```

#### PERF_ARGS

一堆 megatron 的并行参数，只有 `--use-dynamic-batch-size` 与 `--max-tokens-per-gpu` 是 slime 添加的。

`max_tokens_per_gpu` 是指每张卡最多跑多少 token，在开启 `use_dynamic_batch_size` 之后，会尽可能将一个 batch 内部长短不一的数据拼到 `max_tokens_per_gpu`，从而组成动态的 micro batch size，如果有一条数据长度超过了 `max_tokens_per_gpu`，则自成一条，不会对数据进行截断。在开启 context parallel (CP) 时，会让 CP 张卡去上的数据去共享总长为 `CP * max_tokens_per_gpu` 的 token。

在开启 dynamic_batch_size，会忽略传统的 `micro_batch_size`。

⚠️  slime 总是会通过 data packing 的方法训练模型，并且严格保证 per sample loss 或 per token loss，也就是开启 dynamic batch size 不会对 loss 计算有影响，推荐开启。

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
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

#### OPTIMIZER_ARGS

```bash
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
```

#### SGLANG_ARGS

sglang 所需的参数，这里 `--rollout-num-gpus-per-engine` 基本对应 sglang 的 `tp_size`，除此之外的 sglang 参数均通过添加 `--sglang-` 的前缀来传给 slime。

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)
```

⚠️  slime 会用 sgl-router 来调度多个 sglang server，在不开启 dp attention 的情况下不支持 `dp_size`。

### dynamic sampling

slime 支持了更复杂的 sampling 方案，例如 [DAPO](https://dapo-sia.github.io/) 中的 dynamic sampling。如果要开启 dynamic sampling，需要配置：

```bash
   --over-sampling-batch-size ${OVER_SAMPLING_BS} \
   --dynamic-sampling-filter-path \
     slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
```

这里 `over_sampling_batch_size` 需要大于 ``rollout_batch_size`，例如配置为：

```bash
   --rollout-batch-size 32 \
   --n-samples-per-prompt 8 \
   --over-sampling-batch-size 64 \
```

那么 sampling 会直接采样 64 条 prompt，每条 prompt 采样 8 次。因为 slime 内部进行的是异步采样，所以我们会先后获得每个 prompt 的 8 条回复。在收到回复时，会用 `dynamic_sampling_filter_path` 对应的函数进行筛选，如果通过，则留下这 8 条数据，否则则丢掉。例子中的函数是判断回答是否全对或全错：

```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.reward for sample in samples]
    return torch.tensor(rewards, dtype=torch.float).std() > 0.0
```

当我们收到了 32 * 8 条数据时，我们会立刻停止采样，而不会等剩余的数据采样完成。如果删除的数据超过了 32 条 prompt（剩余的小于 32 条 prompt），那么我们会再采样 64 条 prompt。

### partial rollout

在进行 dynamic sampling 的过程中，会提前终止（abort）大量请求，我们可以通过配置 `--partial-rollout` 参数来将生成到一半的请求保存至 data buffer，在下一个 rollout 中取出来继续进行数据生成，从而进一步优化性能。

可以通过配置 `--buffer-filter-path` 来自定义如何从 buffer 中取出数据，默认的函数为：

```python
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

即每次取出前 `num_samples` 个 prompt 对应的 `num_samples * n_samples_per_prompt` 条数据。

⚠️  每条 partial rollout sample 的 `sample.metadata` 中存储了第一次进行生成的 rollout id，可以用于数据过滤。

### bf16 训练 fp8 推理

slime 还支持 bf16 训练，fp8 推理。对于 Qwen3-4B 模型，只需要下载如下模型：

```bash
huggingface-cli download Qwen/Qwen3-4B-FP8 --local-dir /root/Qwen3-4B-FP8
```

并将 `--hf-checkpoint` 替换为：

```bash
#--hf-checkpoint /root/Qwen3-4B
--hf-checkpoint /root/Qwen3-4B-FP8
```

即可触发 fp8 训练。目前我们会将 bf16 权重直接 cast 为 fp8，后续会逐渐添加对精度影响更小的量化方案。

⚠️  训练的 megatron checkpoint 还需要是最开始用 bf16 的 huggingface 转换的。

### 训推分离

在原始的脚本中，资源配置如下：

```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ...
```

即开启训推一体（colocate），并且训练部分会使用 1 机 8 卡，推理会和训练共同使用这 8 张卡张卡。

如果想使用训推分离的功能，需要去掉 `--colocate` 并配置上 `--rollout-num-gpus`，例如：

```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 6 \
   ...
```

此时，就会分配 2 张卡给训练，6 张卡给推理。

⚠️  在进行训推分离的时候，每个 sglang server 上的并发度太大，超过了 sglang 默认的 cuda graph 的并发度（默认最大 160），影响推理速度。可以用以下 2 种方式进行调整：

1. 通过 `--sglang-server-concurrency` 限制发给一个 sglang server 的最大并发量，例如：

   ```bash
   --sglang-server-concurrency 160
    ```

2. 使用 `--sglang-cuda-graph-bs`，即 sglang 原生的 `--cuda-graph-bs`, 增大 sglang 初始化的 cuda graph 数量，例如：

   ```bash
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   ```

### 异步训练

当进行训推分离时，你会发现训练和推理的 GPU 总是相互等待着，为了避免这种资源空闲，我们可以开启异步训练。开启的方式即为将启动脚本中的 `train.py` 改变为 `train_async.py`。这样 slime 就会在进行当前 rollout 的训练时进行下一个 rollout 的数据生成了。

`train.py` 和 `train_async.py` 的差别只在于 train loop 的同步逻辑，我们通过 ray 的异步（`.remote`, `ray.get`）实现了这点。

⚠️  在异步训练时，sglang 的性能检测日志与训练日志可能会混到一起，不易区分，可以通过 `--sglang-log-level` 来减少 sglang 的日志。