# GLM-4.5 with 64xH100


This is an example of doing GLM-4.5 RL training using 64xH100 GPUs.

## Environment Setup

For instructions on setting up the environment and downloading data, please refer to [Example: Qwen3-4B](./qwen3-4B.md).

First, you will need to download GLM-4.5 to a directory accessible by all machines (hereinafter referred to as `$BASE_DIR`):

```bash
huggingface-cli download zai-org/GLM-4.5 --local-dir $BASE_DIR/GLM-4.5-355B-A32B
```

Next, we need to convert the huggingface checkpoint into the torch_dist format with 2 nodes, each with 8 GPUs:

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

Here, `MASTER_ADDR` is the IP of node0, and `NODE_RANK` indicates the node's index, both configured similarly to a multi-node `torchrun` setup.

## Executing the Training

On node0, run:

```bash
cd slime/
bash scripts/run-glm4.5-355B-A32B.sh
```

On other nodes, you need to join the Ray cluster with the following command:

```bash
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats"
```

Alternatively, if you have a list of all node IPs, for example, an MPI hostfile (where each line is `ip slot=8`), you can add the following commands after the `ray start --head` command in `scripts/run-glm4.5-355B-A32B.sh.sh`. This allows you to execute the training entirely from node0:

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
source "${SCRIPT_DIR}/models/glm4.5-355B-A32B.sh"
```

This reads the model's config from [scripts/models/glm4.5-355B-A32B.sh](../../../scripts/models/glm4.5-355B-A32B.sh). These configs are all Megatron parameters. When training with Megatron, it cannot read the model config from the checkpoint, so we need to configure it ourselves. We provide some examples in [scripts/models](../../../scripts/models/).

#### PERF\_ARGS

A set of Megatron parallelism parameters. Only `--use-dynamic-batch-size` and `--max-tokens-per-gpu` are added by slime.

For the Megatron part, we have configured TP8, PP4, CP2, and EP16.

`max_tokens_per_gpu` refers to the maximum number of tokens each GPU can process. When `use_dynamic_batch_size` is enabled, it will pack data of varying lengths within a batch as close to `max_tokens_per_gpu`. If a single data item exceeds `max_tokens_per_gpu`, it will form its own batch without truncation. When context parallelism (CP) is enabled, it allows CP GPUs to share a total length of `CP * max_tokens_per_gpu` tokens.

When `dynamic_batch_size` is enabled, the traditional `micro_batch_size` is ignored.

⚠️ slime always trains the model using data packing and strictly guarantees per-sample or per-token loss. This means enabling dynamic batch size will not affect the loss calculation. It is recommended to enable it.

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

These are the parameters required by sglang. Here, `--rollout-num-gpus-per-engine` basically corresponds to sglang's `tp_size`. Other sglang parameters are passed to slime by adding a `--sglang-` prefix.

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-dp-size 4
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

## FP8 Rollout

The open-source FP8 checkpoint of GLM-4.5 is of per-channel quantization, which could not enable deepep in SGLang. We can use the tool scripts within slime to convert an FP8 checkpoint of 128x128 per-block quant:

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir $BASE_DIR/GLM-4.5-355B-A32B/ \
    --save-dir $BASE_DIR/GLM-4.5-355B-A32B-FP8/ \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

Then, simply change `--hf-checkpoint` to `$BASE_DIR/GLM-4.5-355B-A32B-FP8/` to enable FP8 rollout.

And exemplar `SGLANG_ARGS` for FP8 is:

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
