#!/bin/bash
# OSWorld SFT training for Qwen3-VL-4B-Instruct.
#
# Dataset: Jarrodbarnes/osworld-reasoning-sft-v1 (339 samples)
# - 273 original ground-truth demos
# - 66 Claude Opus 4.5 reasoning trajectories
#
# Hyperparameters tuned for:
# - Small dataset (339 samples) → conservative learning rate
# - Warmup prior to RL → limited epochs to preserve base capabilities
# - Qwen3-VL methodology → frozen vision encoder
#
# Usage:
#   ./train_sft.sh
#   SLIME_SCRIPT_NUM_GPUS=4 ./train_sft.sh

set -ex

# Configuration
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-4B-Instruct"}
DATASET_NAME=${SLIME_SCRIPT_DATASET_NAME:-"Jarrodbarnes/osworld-reasoning-sft-v1"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-4}
OUTPUT_DIR=${SLIME_SCRIPT_OUTPUT_DIR:-"/ephemeral/osworld-vlm-sft"}

# Load API keys if present
if [ -f "/root/slime/.env" ]; then
    set -a
    source /root/slime/.env
    set +a
fi

# Cleanup
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3

# Ray setup
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" --disable-usage-stats

SLIME_DIR="${SLIME_DIR:-/root/slime}"
cd "$SLIME_DIR"

# Download model and dataset
mkdir -p /root/models /ephemeral
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
    echo "Downloading model: Qwen/${MODEL_NAME}"
    huggingface-cli download "Qwen/${MODEL_NAME}" --local-dir "/root/models/${MODEL_NAME}"
fi

DATASET_LOCAL="/ephemeral/osworld_reasoning_sft"
if [ ! -d "$DATASET_LOCAL" ]; then
    echo "Downloading dataset: ${DATASET_NAME}"
    huggingface-cli download --repo-type dataset "${DATASET_NAME}" --local-dir "$DATASET_LOCAL"
fi

# Checkpoint args
# Note: --load is for resuming from Slime checkpoints, not needed for fresh SFT from HF
CKPT_ARGS=(
    --hf-checkpoint "/root/models/${MODEL_NAME}"
)

# SFT training args
# Dataset: 339 samples
# - batch_size 32 → ~10 steps/epoch
# - 5 epochs → ~50 steps total (conservative warmup)
# - LR 5e-6 with cosine decay → preserve base capabilities
SFT_ARGS=(
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --prompt-data "${DATASET_LOCAL}/osworld_reasoning_sft_v1.jsonl"
    --input-key messages
    --rollout-shuffle

    # Training schedule - conservative for small dataset
    --num-epoch 5
    --rollout-batch-size 32
    --global-batch-size 32

    --loss-type sft_loss
    --loss-mask-type qwen3
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
)

# Multimodal keys - maps "images" field to model inputs
MULTIMODAL_KEYS='{"image": "images"}'

# Optimizer args - conservative for warmup
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-6
    --lr-decay-style cosine
    --min-lr 5e-7
    --lr-warmup-fraction 0.1
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

# FSDP backend - recommended for VLM training
BACKEND_ARGS=(
    --train-backend fsdp
    --actor-num-nodes 1
    --actor-num-gpus-per-node "$NUM_GPUS"
    --gradient-checkpointing
    --freeze-vision-encoder
    --attn-implementation flash_attention_2
)

# Save args
SAVE_ARGS=(
    --save "$OUTPUT_DIR"
    --save-interval 1
)

# Wandb args (optional)
if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project osworld-sft
        --wandb-group "qwen3-vl-4b-reasoning"
        --wandb-key "$WANDB_API_KEY"
    )
else
    WANDB_ARGS=()
fi

# Runtime environment
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"TOKENIZERS_PARALLELISM\": \"false\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

# Print configuration
echo "=========================================="
echo "OSWorld SFT Training Configuration"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME} (339 samples)"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size: 32 | Epochs: 5 | ~50 steps"
echo "LR: 5e-6 (cosine) | Vision encoder: FROZEN"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python3 "$SLIME_DIR/train_async.py" \
    --multimodal-keys "${MULTIMODAL_KEYS}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${BACKEND_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    "${WANDB_ARGS[@]}"

echo "SFT training complete. Checkpoint: $OUTPUT_DIR"
echo "Next step: Run train_grpo.sh with SLIME_SCRIPT_SFT_CKPT=$OUTPUT_DIR"
