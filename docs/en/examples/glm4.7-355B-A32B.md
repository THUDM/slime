# GLM-4.7 with 64xH100

## Environment Preparation

The environment setup and dataset download are the same as for the Qwen3-4B model. You can refer to [Example: Qwen3-4B Model](qwen3-4B.md), replacing mentions of Qwen3-4B with GLM-4.7.

### Prerequisites

GLM-4.7 follows the standard slime Docker environment. For multi-node launches, make sure all nodes can access the same `$BASE_DIR` path and unset proxy variables before starting Ray workers:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

### Download Model

```bash
hf download zai-org/GLM-4.7 --local-dir $BASE_DIR/GLM-4.7-355B-A32B
```

### Convert Checkpoint

To convert the Hugging Face checkpoint to torch_dist format, use 2 nodes x 8 GPUs:

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

Here, `MASTER_ADDR` is the IP of node0, and `NODE_RANK` is the node index, configured just like a multi-node `torchrun` job.

## Run Training

Execute the training script from node0:

```bash
cd /root/slime
export BASE_DIR=/shared/path  # accessible by all nodes
bash scripts/run-glm4.7-355B-A32B.sh
```

### Parameter Introduction

Here, we briefly introduce the key parts in the [run-glm4.7-355B-A32B.sh](https://github.com/THUDM/slime/blob/main/scripts/run-glm4.7-355B-A32B.sh) script.

#### MoE Configuration

GLM-4.7 is a Mixture-of-Experts (MoE) model with 160 routed experts (top-8 activation) and shared experts. It has 92 layers: 3 dense layers + 89 MoE layers.

1. To support GLM-4.7 on 64xH100, we enable Megatron's CPU Adam to save GPU memory:

   ```bash
   OPTIMIZER_ARGS=(
      ...
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
   ```

2. Enable MoE optimization in Megatron. For the provided 64xH100 example, we use TP=8, PP=4, CP=2, and EP=16:

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

3. Enable MoE optimization in SGLang with DP attention:

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

#### MTP Speculative Decoding (Inference Acceleration)

GLM-4.7 includes MTP (Multi-Token Prediction) layers that can be used for speculative decoding during inference to speed up rollout generation. To enable this, add the following to `SGLANG_ARGS`:

```bash
SGLANG_ARGS=(
   ...
   # MTP speculative decoding (EAGLE)
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
)
```

This lets SGLang use the model's MTP layer as the draft model for EAGLE-style speculative decoding.

> ⚠️ **Note**: Speculative decoding requires additional GPU memory. If you encounter OOM issues, try reducing `--sglang-mem-fraction-static` or disabling speculative decoding.

#### MTP Training

slime also supports training the MTP layers jointly with the main model for GLM-4.7. When enabled, the relevant arguments are:

```bash
# Add MTP layer count to model config
MODEL_ARGS+=(--mtp-num-layers 1)

# Enable MTP training
MTP_ARGS=(
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)
```

- `--mtp-num-layers 1`: Tells Megatron to load the MTP layer from the checkpoint.
- `--enable-mtp-training`: Enables gradient computation for MTP layers. Without this flag, the MTP layer is loaded but frozen.
- `--mtp-loss-scaling-factor 0.2`: Weight of the MTP loss relative to the main policy loss. Default is 0.2.

> **Note**: MTP training for GLM-4.7 relies on `GLM4MoEBridge` (in `slime_plugins/mbridge/glm4moe.py`) to map regular and MTP weights between HuggingFace and Megatron formats.

#### Multi-Node Support

This example already targets multi-node training. Before launching:

- Place the model checkpoints and datasets on a path accessible by all nodes.
- Set `MASTER_ADDR` to an address reachable by all nodes.
- Unset proxy variables before starting Ray workers.
- Provide a `HOSTFILE` listing worker IPs (one per line) and export `HOSTFILE=/path/to/hostfile` before launching.
- Adjust parallelism coherently. The default example uses TP=8, PP=4, EP=16, CP=2, while rollout uses 32 GPUs per engine with SGLang DP attention.

If your rollout GPU count does not divide the expert count cleanly, you can use `--sglang-ep-num-redundant-experts` to add redundant experts.

## FP8 Rollout

The open-source FP8 checkpoint of GLM-4.7 uses per-channel quantization, which cannot currently enable DeepEP in SGLang. You can convert it to a 128x128 per-block FP8 checkpoint with the tool provided in slime:

```bash
cd /root/slime
python tools/convert_hf_to_fp8.py \
    --model-dir $BASE_DIR/GLM-4.7-355B-A32B/ \
    --save-dir $BASE_DIR/GLM-4.7-355B-A32B-FP8/ \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

Then switch `--hf-checkpoint` to `$BASE_DIR/GLM-4.7-355B-A32B-FP8/` to enable FP8 rollout.

An example FP8 `SGLANG_ARGS` setup is:

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
