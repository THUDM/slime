# DeepSeek R1 with 128xH100


This is an example of doing DeepSeek R1 RL training using 128xH100 GPUs.

We will use bf16 for training, and an fp8 format with 128x128 blockwise quantization for inference. The maximum response length is 32k, and dynamic sampling will be used to filter data during training.

Regarding parallelism, for sglang we will enable EP64, activate dp attention, and deepep. For the Megatron part, we will use TP8, PP4, EP32, and CP4.

⚠️ To save GPU memory, we will use CPU Adam. Each node (8xH100) will occupy 1.4\~1.5TB of host memory. If a single machine's host memory is insufficient, this can be resolved by adding more GPUs to expand the parallelism.

## Environment Setup

For instructions on setting up the environment and downloading data, please refer to [Example: Qwen3-4B](qwen3-4B.md).

To prepare the DeepSeek R1 checkpoint, first you will need to download DeepSeek-R1 to a directory accessible by all machines (hereinafter referred to as `$BASE_DIR`):

```bash
hf download deepseek-ai/DeepSeek-R1 --local-dir $BASE_DIR/DeepSeek-R1
```

The Hugging Face checkpoint for DeepSeek-R1 is in a block-quantized fp8 format. To convert it into a torch_dist format that Megatron can load, you first need to convert it to a bf16 Hugging Face checkpoint:

```bash
cd slime/
python tools/fp8_cast_bf16.py --input-fp8-hf-path $BASE_DIR/DeepSeek-R1 --output-bf16-hf-path $BASE_DIR/DeepSeek-R1-bf16/
```

Next, we need to convert the bf16 version of DeepSeek-R1 into the torch_dist format. Specifically, execute the following on 4 separate nodes:

```bash
cd slime/
source scripts/models/deepseek-v3.sh
PYTHONPATH=/root/Megatron-LM/ torchrun \
   --nproc-per-node 8 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 8 \
   --expert-tensor-parallel-size 1 \
   --expert-model-parallel-size 4 \
   --decoder-first-pipeline-num-layers 7 \
   --decoder-last-pipeline-num-layers 6 \
   --hf-checkpoint $BASE_DIR/DeepSeek-R1-bf16/ \
   --save $BASE_DIR/DeepSeek-R1_torch_dist/
```

Here, `MASTER_ADDR` is the IP of node0, and `NODE_RANK` indicates the node's index, both configured similarly to a multi-node `torchrun` setup.

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

This reads the model's config from [scripts/models/deepseek-v3.sh](https://github.com/THUDM/slime/blob/main/scripts/models/deepseek-v3.sh). These configs are all Megatron parameters. When training with Megatron, it cannot read the model config from the checkpoint, so we need to configure it ourselves. We provide some examples in [scripts/models](https://github.com/THUDM/slime/tree/main/scripts/models/).

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
   --sglang-mem-fraction-static 0.7
   --sglang-enable-ep-moe

   # dp attention
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head

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
   ...

   # use deepep for megatron
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)
```
