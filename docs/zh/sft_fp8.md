# 示例：Qwen3-4B-Base + OpenHermes-2.5
> ⚠️ **Note**:
> FP8 sft目前在实验阶段，还在进行精度对齐实验


[English](../en/sft_fp8.md)

## 环境准备

首先需要按照[FP8 Dockerfile](../../docker/Dockerfile.fp8)创建镜像环境。

然后我们仿照 [示例：Qwen3-4B 模型](./models/qwen3-4B.md) 转换 `Qwen3-4B-Base` 模型。

之后，我们处理 sft 数据。这里我们以经典的 [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) 为例，首先把数据处理成适合 slime 加载的格式，可以用如下的脚本进行处理，增加一个符合 openai message 格式的列，并保存在 `/root/openhermes2_5.parquet`。

```python
from datasets import load_dataset

ds = load_dataset("teknium/OpenHermes-2.5")["train"]

def convert(sample):
    conversations = sample["conversations"]

    def convert_role(role):
        if role == "human":
            return "user"
        elif role == "gpt":
            return "assistant"
        elif role == "system":
            return "system"
        else:
            raise ValueError(f"Unknown role: {role}")

    messages = [
        {
            "role": convert_role(turn["from"]),
            "content": turn["value"],
        }
        for turn in conversations
    ]

    return {"messages": messages}

ds = ds.map(convert)
ds.to_parquet("/root/openhermes2_5.parquet")
```

## 执行训练

执行训练：

```bash
cd /root/slime
bash script/run-qwen3-4B-base-sft-fp8.sh
```

### 参数简介

可以将 [run-qwen3-4B-base-sft.sh](../../scripts/run-qwen3-4B-base-sft.sh) 与 [run-qwen3-4B.sh](../../scripts/run-qwen3-4B.sh) 进行对比。会发现除了我们将模型由 instruct 模型换为了 base 模型之外，主要进行了如下的几个调整：

1. 移除了 `SGLANG_ARGS` 和 `GRPO_ARGS`。这是因为 sft 的过程中不需要启动 sglang 或者做 grpo 相关的配置；

2. 将 `ROLLOUT_ARGS` 改名为了 `SFT_ARGS`，并配置为：

   ```bash
   SFT_ARGS=(
      --rollout-function-path slime.rollout.sft_rollout.generate_rollout
      --prompt-data /root/openhermes2_5.parquet
      --input-key messages
      --rollout-shuffle
      --num-epoch 3
      --rollout-batch-size 128
      --global-batch-size 128
   
      --loss-type sft_loss
      --calculate-per-token-loss
      --disable-compute-advantages-and-returns
      --debug-train-only
   )
   ```

   slime 中的 sft 实际上是复用了 slime 的 custom rollout 功能，通过 `--rollout-function-path` 将数据生成部分从使用 sglang 的 RL rollout，切换成了从文件中读取数据的 sft 版本，即 `slime.rollout.sft_rollout.generate_rollout`。

   对于 sft 来说，建议将 `rollout_batch_size` 与 `global_batch_size` 设置成相同的，并不要配置 `n_samples_per_prompt`，这样相当于是读一个 batch 就训一个 batch。

   slime 还支持不同的 loss 类型，我们就是通过 `--loss-type sft_loss` 配置上 sft loss 的。

   至于 `--calculate-per-token-loss`，这是因为 slime 默认是以 GRPO 的 per sample mean 进行计算的，而一般 sft 训练都是按一个 batch 的所有不被 mask 的 token 取平均，所以建议配置上。

   最后 `--disable-compute-advantages-and-returns` 表示 sft 的过程中不需要预先计算 log prob，`--debug-train-only` 表示不需要初始化 sglang。

3. 使用了 `train_async.py` 而不是 `train.py`。这是为了利用异步训练的流程，来实现数据 prefetch。

4. 新增参数说明
   
    ```bash
    PERF_ARGS=(
      --tensor-model-parallel-size 4
      --sequence-parallel
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --expert-model-parallel-size 1
      --expert-tensor-parallel-size 1
    
      #--recompute-granularity full
      #--recompute-method uniform
      #--recompute-num-layers 1
    
      # --micro-batch-size 1
      --use-dynamic-batch-size
      --max-tokens-per-gpu 32768
    )
    ```
   
   - `tp`, `sp`, `pp`：正常设置，这里只设置了tp和sp。
   - `cp`: 可以设置，但是因为比较影响性能，这里选择关掉
   - `recompute-granularity`: recompute有效降低显存使用，但是会增加step时间，这里关掉取最大性能
   - `max-tokens-per-gpu`: 每张卡处理长度。经过一些memory peak bug修复之后，直接拉到32k

   ```bash
   PERCISE_ARGS=(
     --transformer-impl transformer_engine
     --bf16
     --fp8-format e4m3
     --fp8-recipe blockwise
     --fp8-param-gather
   
     #--fp8-recipe mxfp8
     #--reuse-grad-buf-for-mxfp8-param-ag
   
     #--activation-func-fp8-input-store
     #--overlap-grad-reduce
     #--overlap-param-gather
   )
   ```
   
   - `transformer-impl`: 使用te model
   - `fp8-format`, `fp8-recipe` and `fp8-param-gather`: 组合使用，开启Megatron的fp8 training
   - `fp8-recipe mxfp8` and `reuse-grad-buf-for-mxfp8-param-ag`: mxfp8的组合使用，开启nvidia的mxfp8 training，但是只在Blackwell上支持
   - `activation-func-fp8-input-store`: 激活值量化，有效降低使用显存(关闭recompute-granularity的情况下),笔者自己增加上去的实验功能
   - `overlap-grad-reduce`, `overlap-param-gather`: 适配到了slime上，但目前开了发现有反效果，暂时不考虑开启

   ```bash
   WANDB_ARGS=(
     --use-wandb
     --wandb-mode offline
     --wandb-project slime-dev
     --wandb-group qwen3-4B-base-sft
     --wandb-dir ${SAVE_DIR}/fp8
     # --wandb-key ${WANDB_KEY}
   )

   TENSORBOARD_ARGS=(
     --use-pytorch-profiler
     --profile-step-start 16
     --profile-step-end 18
     --tensorboard-dir ${SAVE_DIR}/tensorboard/
     --record-memory-history
   )
   ```
   一些profiler参数，可以根据自己需要进行设置
   - `use-pytorch-profiler`: 开启MegatronTrainRayActor的torch.profiler
   - `profile-step-start`, `profile-step-end`: 第几个rollout_id开始profile和第几个rollout_id结束profile
   - `use-wandb`: 开启wandb