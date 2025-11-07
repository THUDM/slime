# GLM4-9B with 8xH100

## Environment Setup

After pulling the `slimerl/slime:latest` image, initialize the image environment as follows:

```bash
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime/
pip install -e .
```

Download the model and data:

```bash
# hf checkpoint
huggingface-cli download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414

# train data
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# eval data
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

Convert the Hugging Face checkpoint to a Megatron-loadable Hugging Face checkpoint:

```bash
# mcore checkpoint
cd /root/slime
source scripts/models/glm4-9B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist
```

## Run Training

Execute the training:

```bash
cd /root/slime
bash scripts/run-glm4-9B.sh
```

### Parameter Introduction

Here, we will briefly introduce the various components of the [run-glm4-9B.sh](../../../scripts/run-glm4-9B.sh) script:

#### MODEL\_ARGS

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"
```

Reads the model's config from [scripts/models/glm4-9B.sh](../../../scripts/models/glm4-9B.sh). These configs are all Megatron parameters. When training with Megatron, it cannot read the model config from the checkpoint, so we need to configure it ourselves. We provide some examples in [scripts/models](../../../scripts/models/).

⚠️  Ensure that settings such as `--rotary-base` in the model configuration file match the settings of the model you are currently training. This is because different models, even with the same architecture, might use different values. If needed, you can override these parameters in your script after loading the model weights. For instance:

```bash
source "${SCRIPT_DIR}/models/glm4-9B.sh"

MODEL_ARGS += ( --rotary-base 10000 )
```

#### CKPT\_ARGS

```bash
CKPT_ARGS=(
   # HF checkpoint required by sglang; we also read the tokenizer from here
   --hf-checkpoint /root/GLM-Z1-9B-0414
   # Checkpoint for the reference model
   --ref-load /root/GLM-Z1-9B-0414_torch_dist
   # Load directory for the actor; if empty, it will be loaded from `ref_load`
   --load /root/GLM-Z1-9B-0414_slime/
   --save /root/GLM-Z1-9B-0414_slime/
   --save-interval 20
)
```

#### ROLLOUT\_ARGS

```bash
ROLLOUT_ARGS=(
   # Prompt dataset, each line is a JSON object
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   # If the `input_key` in the prompt contains an OpenAI message,
   # tokenizer.apply_chat_template(...) will be executed
   --apply-chat-template
   # Whether to shuffle the data
   --rollout-shuffle

   # Reward model type.
   # slime provides many types and --custom-rm-path for custom models
   --rm-type deepscaler

   # Total number of rollouts to train
   --num-rollout 3000
   # Number of prompts in one rollout
   --rollout-batch-size 32
   # Number of responses to sample per prompt
   # A rollout will have rollout_batch_size * n_samples_per_prompt items
   --n-samples-per-prompt 8
   # Rollout sampling parameters
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   # Number of training steps corresponding to one rollout
   --num-steps-per-rollout 1
   # Whether to balance data during training, which might improve speed
   --balance-data
)
```

#### EVAL\_ARGS

During evaluation, most rollout parameters are inherited, but we provide some parameters that can override the rollout configuration, allowing for different sampling strategies for training and evaluation.

```bash
EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)
```

#### PERF\_ARGS

A set of Megatron's parallelism parameters. Only `--use-dynamic-batch-size` and `--max-tokens-per-gpu` are added by slime.

`max_tokens_per_gpu` specifies the maximum number of tokens each GPU can process. When `use_dynamic_batch_size` is enabled, it will try to pack data of varying lengths within a batch up to `max_tokens_per_gpu`, thus forming a dynamic micro-batch size. If a single data item's length exceeds `max_tokens_per_gpu`, it will form its own batch without being truncated. When context parallelism (CP) is enabled, it allows the CP GPUs to share data with a total length of `CP * max_tokens_per_gpu` tokens.

When `dynamic_batch_size` is enabled, the traditional `micro_batch_size` is ignored.

⚠️  slime always trains the model using data packing and strictly guarantees per-sample or per-token loss. This means enabling dynamic batch size will not affect the loss calculation. It is recommended to enable it.

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)
```

#### GRPO\_ARGS

Here are some GRPO-related parameters:

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

#### OPTIMIZER\_ARGS

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

#### SGLANG\_ARGS

Parameters required by sglang. Here, `--rollout-num-gpus-per-engine` basically corresponds to sglang's `tp_size`. Other sglang parameters are passed to slime by adding the `--sglang-` prefix.

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)
```

⚠️  slime uses `sgl-router` to schedule multiple sglang servers. `dp_size` is not supported when DP attention is disabled.

### Co-located Training and Inference

In the original script, the resource configuration is as follows:

```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ...
```

This enables decoupled training and inference, where the training part will use 1 machine with 4 GPUs, and the inference will use another 4 GPUs.

If you want to use the co-located feature, you need to add `--colocate` and remove `--rollout-num-gpus`:

```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ...
```

In this case, both training and inference will share these 8 GPUs.

⚠️ When using co-located training and inference, Megatron will always occupy some GPU memory. Therefore, you need to adjust `--sglang-mem-fraction-static` to reduce the proportion of memory occupied by sglang.

### Dynamic Sampling

slime supports more complex sampling schemes, such as the dynamic sampling in [DAPO](https://dapo-sia.github.io/). To enable dynamic sampling, you need to configure:

```bash
   --over-sampling-batch-size ${OVER_SAMPLING_BS} \
   --dynamic-sampling-filter-path \
     slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
```

Here, `over_sampling_batch_size` needs to be greater than `rollout_batch_size`. For example:

```bash
   --rollout-batch-size 32 \
   --n-samples-per-prompt 8 \
   --over-sampling-batch-size 64 \
```

The sampling will then directly sample 64 prompts, with 8 samples per prompt. Since slime performs asynchronous sampling internally, we will receive the 8 responses for each prompt sequentially. Upon receiving responses, they will be filtered using the function specified by `dynamic_sampling_filter_path`. If they pass, these 8 data points are kept; otherwise, they are discarded. The function in the example checks if the answers are all correct or all incorrect:

```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.reward for sample in samples]
    return torch.tensor(rewards, dtype=torch.float).std() > 0.0
```

When we have received 32 \* 8 data points, we will immediately stop sampling and will not wait for the remaining data to be sampled. If more than 32 prompts' worth of data is discarded (leaving fewer than 32 prompts' worth), we will then sample another 64 prompts.

### Partial Rollout

During the process of dynamic sampling, a large number of requests are aborted prematurely. We can configure the `--partial-rollout` parameter to save these partially generated requests to a data buffer. In the next rollout, these requests can be retrieved to continue data generation, thereby further optimizing performance.

You can customize how data is retrieved from the buffer by configuring the `--buffer-filter-path`. The default function is:

```python
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

This means that each time, the data corresponding to the first `num_samples` prompts is retrieved, totaling `num_samples * n_samples_per_prompt` items.

⚠️ The `sample.metadata` of each partial rollout sample stores the rollout ID from its initial generation, which can be used for data filtering.
