# Example: Training DeepSeek R1 with 128xH100

[中文版](../../zh/models/deepseek-r1.md)

This is an example of doing DeepSeek R1 RL training using 128xH100 GPUs.

We will use bf16 for training, and an fp8 format with 128x128 blockwise quantization for inference. The maximum response length is 32k, and dynamic sampling will be used to filter data during training.

Regarding parallelism, for sglang we will enable EP64, activate dp attention, and deepep. For the Megatron part, we will use TP8, PP4, EP32, and CP4.

⚠️ To save GPU memory, we will use CPU Adam. Each node (8xH100) will occupy 1.4\~1.5TB of host memory. If a single machine's host memory is insufficient, this can be resolved by adding more GPUs to expand the parallelism.

## Environment Setup

To prepare the DeepSeek R1 checkpoint, you will need to use [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch).

First, download DeepSeek-R1 to a directory accessible by all machines (hereinafter referred to as `$BASE_DIR`):

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir $BASE_DIR/DeepSeek-R1
```

The Hugging Face checkpoint for DeepSeek-R1 is in a block-quantized fp8 format. To convert it into a torch_dist format that Megatron can load, you first need to use Pai-Megatron-Patch to convert it to a bf16 Hugging Face checkpoint:

```bash
cd $BASE_DIR/
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git

export MP_PP0_LAYERS=5
cd Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/deepseek
python fp8_cast_bf16.py --input-fp8-hf-path $BASE_DIR/DeepSeek-R1 --output-bf16-hf-path $BASE_DIR/DeepSeek-R1-bf16/
```

Next, we need to use Pai-Megatron-Patch's multi-node conversion script to convert the bf16 version of DeepSeek-R1 into the torch_dist format. Specifically, execute the following on 4 separate nodes:

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

Here, `MASTER_ADDR` is the IP of node0, and `MASTER_PORT` is a specific port, both configured similarly to a multi-node `torchrun` setup. `RANK` indicates the node's index.

For instructions on setting up the environment and downloading data, please refer to [Example: Qwen3-4B](./qwen3-4B.md).

## Executing the Training

On node0, run:

```bash
cd slime/
bash scripts/run-deepseek-r1.sh
```

On other nodes, you need to join the Ray cluster with the following command:

```bash
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats"
```

Alternatively, if you have a list of all node IPs, for example, an MPI hostfile (where each line is `ip slot=8`), you can add the following commands after the `ray start --head` command in `scripts/run-deepseek-r1.sh`. This allows you to execute the training entirely from node0:

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

### Parameter Introduction

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/deepseek-v3.sh"
```

This reads the model's config from [scripts/models/deepseek-v3.sh](../../../scripts/models/deepseek-v3.sh). These configs are all Megatron parameters. When training with Megatron, it cannot read the model config from the checkpoint, so we need to configure it ourselves. We provide some examples in [scripts/models](../../../scripts/models/).

#### CKPT\_ARGS

```bash
CKPT_ARGS=(
   # HF ckpt required by sglang, we also read the tokenizer from here
   --hf-checkpoint $BASE_DIR/DeepSeek-R1/
   #--hf-checkpoint $BASE_DIR/DeepSeek-R1-bf16/
   --ref-load $BASE_DIR/DeepSeek-R1_torch_dist/
   # Actor's load directory, if empty, it will read from `ref_load`
   --load $BASE_DIR/DeepSeek-R1_slime/
   --save $BASE_DIR/DeepSeek-R1_slime/
   --save-interval 20
)
```

slime will perform online quantization during training based on the quantization configuration in `hf_checkpoint`. For instance, in the current example, we are using the fp8 checkpoint of DeepSeek R1. This means that when updating parameters, we will first perform blockwise quantization on the parameters before passing them to sglang.

#### ROLLOUT\_ARGS

```bash
ROLLOUT_ARGS=(
   # Prompt dataset, each line is a json
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   # If the `input_key` in the prompt contains an OpenAI message,
   # it will perform tokenizer.apply_chat_template(...)
   --apply-chat-template
   # Whether to shuffle the data
   --rollout-shuffle

   # Reward model type,
   # slime provides many types and --custom-rm-path for customization
   --rm-type deepscaler

   # Total number of rollouts to train
   --num-rollout 3000
   # Number of prompts in one rollout
   --rollout-batch-size 128
   # Number of replies to sample for each prompt
   # One rollout will have rollout_batch_size * n_samples_per_prompt items
   --n-samples-per-prompt 8
   # Rollout sampling parameters
   --rollout-max-response-len 32768
   --rollout-temperature 0.8

   # Use double the batch size for sampling, and filter out samples where the reward variance is 0
   --over-sampling-batch-size 256
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   # Number of training steps corresponding to one rollout
   --num-steps-per-rollout 4
   # Whether to balance data during training, which may improve speed
   --balance-data
)
```

#### PERF\_ARGS

A set of Megatron parallelism parameters. Only `--use-dynamic-batch-size` and `--max-tokens-per-gpu` are added by slime.

For the Megatron part, we have configured TP8, PP4, CP4, and EP32. Since DeepSeek-R1 has 61 layers, which is not divisible by 4, we have specifically configured the last pipeline stage to have 13 layers.

`max_tokens_per_gpu` refers to the maximum number of tokens each GPU can process. When `use_dynamic_batch_size` is enabled, it will pack data of varying lengths within a batch as close to `max_tokens_per_gpu`. If a single data item exceeds `max_tokens_per_gpu`, it will form its own batch without truncation. When context parallelism (CP) is enabled, it allows CP GPUs to share a total length of `CP * max_tokens_per_gpu` tokens.

When `dynamic_batch_size` is enabled, the traditional `micro_batch_size` is ignored.

⚠️ slime always trains the model using data packing and strictly guarantees per-sample or per-token loss. This means enabling dynamic batch size will not affect the loss calculation. It is recommended to enable it.

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

#### GRPO\_ARGS

Currently, these are some GRPO-related parameters in slime:

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

If you wish to train without loading the reference model, you need to remove `--use-kl-loss` and set `--kl-coef 0.00` (the default value is 0).

#### OPTIMIZER\_ARGS

We have configured CPU Adam with the following parameters to save GPU memory.

```bash
OPTIMIZER_ARGS=(
   ...

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)
```

#### SGLANG\_ARGS

These are the parameters required by sglang. Here, `--rollout-num-gpus-per-engine` basically corresponds to sglang's `tp_size`. Other sglang parameters are passed to slime by adding a `--sglang-` prefix. To fully leverage sglang's large EP inference capabilities, we have added configurations like ep64, dp\_attention dp8, and deepep mode auto.

The final `--sglang-server-concurrency` is a parameter specific to slime. It is used to prevent the sglang server's concurrent requests from becoming too large and crashing the HTTP server. The default is 512. However, since we now have one server for 8 nodes, we have adjusted it to 1024 to ensure that each dp rank can have a concurrency of 128.

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

   # make every dp rank have 128 concurrency
   --sglang-server-concurrency 1024
)
```

#### MISC\_ARGS

Some additional Megatron configurations. Note that Megatron's deepep is configured here.

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
