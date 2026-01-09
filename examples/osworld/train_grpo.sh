#!/bin/bash
# OSWorld GSPO training with experience replay for Qwen3-VL-4B-Instruct.

set -ex

# Load API keys if present
if [ -f "/root/slime/.env" ]; then
    set -a
    source /root/slime/.env
    set +a
fi

# Configuration
MODEL_NAME="Qwen3-VL-4B-Instruct"
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-4}
CP_SIZE=${SLIME_SCRIPT_CP_SIZE:-2}
TRAIN_GPUS=${SLIME_SCRIPT_TRAIN_GPUS:-2}
INFER_GPUS=${SLIME_SCRIPT_INFER_GPUS:-1}
# Training turns (exported for Ray job) - max 8 based on trajectory data
export OSWORLD_TRAIN_TRUNCATE_TURNS=${SLIME_SCRIPT_TRAIN_TRUNCATE_TURNS:-8}

export OSWORLD_SCREEN_DIFF_THRESHOLD=${SLIME_SCRIPT_SCREEN_DIFF_THRESHOLD:-0.005}

POLICY_CKPT=${SLIME_SCRIPT_SFT_CKPT:-"/ephemeral/osworld-vlm-sft-step25-hf"}
TASK_DATA=${SLIME_SCRIPT_TASK_DATA:-"/ephemeral/osworld_union/osworld_tasks_union.parquet"}
EVAL_DATA=${SLIME_SCRIPT_EVAL_DATA:-"/ephemeral/osworld_tasks/test.parquet"}
OUTPUT_DIR="/ephemeral/osworld-vlm-gspo-4b-replay"
DISABLE_EVAL=${SLIME_SCRIPT_DISABLE_EVAL:-"true"}

export OSWORLD_REPLAY_BUFFER=${SLIME_SCRIPT_REPLAY_BUFFER:-"/ephemeral/osworld_union/osworld_replay_union.jsonl"}
export OSWORLD_REPLAY_THRESHOLD=${SLIME_SCRIPT_REPLAY_THRESHOLD:-"0.5"}
export OSWORLD_REWARD_DEBUG_LIMIT=${SLIME_SCRIPT_REWARD_DEBUG_LIMIT:-"5"}

# Sampling schedule:
# Phase 1 (exploration): SLIME_SCRIPT_ROLLOUT_TEMPERATURE=0.8
# Phase 2 (stability):   SLIME_SCRIPT_ROLLOUT_TEMPERATURE=0.5
ROLLOUT_TEMPERATURE=${SLIME_SCRIPT_ROLLOUT_TEMPERATURE:-0.8}
ROLLOUT_TOP_P=${SLIME_SCRIPT_ROLLOUT_TOP_P:-0.95}

pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus "$NUM_GPUS" --disable-usage-stats

SLIME_DIR="${SLIME_DIR:-/root/slime}"
cd "$SLIME_DIR"

if [ ! -d "$POLICY_CKPT" ]; then
    echo "ERROR: SFT checkpoint not found at $POLICY_CKPT"
    echo "Run examples/osworld/train_sft.sh first"
    exit 1
fi

if [ ! -f "$TASK_DATA" ]; then
    echo "WARNING: Task data not found at $TASK_DATA"
    echo "Make sure OSWorld tasks are available"
fi

CKPT_ARGS=(
    --hf-checkpoint "$POLICY_CKPT"
)

ROLLOUT_ARGS=(
    --prompt-data "$TASK_DATA"
    --input-key prompt
    --metadata-key task_config
    --apply-chat-template
    --loss-mask-type qwen3
    --rollout-shuffle
    --num-rollout 500
    --rollout-batch-size 2
    --n-samples-per-prompt 4
    --rollout-max-response-len 512
    --rollout-max-context-len 16384
    --rollout-temperature "$ROLLOUT_TEMPERATURE"
    --rollout-top-p "$ROLLOUT_TOP_P"
    --global-batch-size 8
)

CUSTOM_ARGS=(
    --rollout-function-path examples.osworld.rollout.generate_rollout
    --custom-generate-function-path examples.osworld.rollout.generate
    --custom-rm-path examples.osworld.reward.async_compute_reward
    --custom-config-path "$SLIME_DIR/examples/osworld/grpo_config.yaml"
)

MULTIMODAL_ARGS=()

if [ "$DISABLE_EVAL" = "true" ]; then
    EVAL_ARGS=()
else
    EVAL_ARGS=(
        --eval-interval 10
        --eval-prompt-data osworld_test "$EVAL_DATA"
        --n-samples-per-eval-prompt 1
        --eval-max-response-len 2048
    )
fi

GSPO_ARGS=(
    --advantage-estimator gspo
    --kl-loss-coef 0.01
    --kl-loss-type low_var_kl
    --entropy-coef 0.0001
    --eps-clip 3.5e-4
)

OPTIMIZER_ARGS=(
    --lr 5e-7
    --lr-decay-style constant
    --clip-grad 0.5
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus "$INFER_GPUS"
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.50
    --sglang-cuda-graph-bs 1 2 4
)

BACKEND_ARGS=(
    --train-backend fsdp
    --actor-num-nodes 1
    --actor-num-gpus-per-node "$TRAIN_GPUS"
    --gradient-checkpointing
    --context-parallel-size "$CP_SIZE"
    --fsdp-cpu-offload
    --fsdp-cpu-backend gloo
)

SAVE_ARGS=(
    --save "$OUTPUT_DIR"
    --save-interval 1
)

if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project osworld-grpo
        --wandb-group "qwen3-vl-4b-replay"
        --wandb-key "$WANDB_API_KEY"
    )
else
    WANDB_ARGS=()
fi

export OSWORLD_SERVER_URL=${OSWORLD_SERVER_URL:-"http://172.17.0.1:8100"}
export OSWORLD_TIMEOUT=${SLIME_SCRIPT_OSWORLD_TIMEOUT:-900}

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"TOKENIZERS_PARALLELISM\": \"false\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CUDA_VISIBLE_DEVICES\": \"0,1,2,3\",
    \"SGLANG_DISABLE_CUDNN_CHECK\": \"1\",
    \"SGLANG_VLM_CACHE_SIZE_MB\": \"3072\",
    \"OSWORLD_REWARD_ALPHA\": \"0.3\",
    \"OSWORLD_REWARD_DEBUG_LIMIT\": \"3\",
    \"OSWORLD_TRAIN_TRUNCATE_TURNS\": \"${OSWORLD_TRAIN_TRUNCATE_TURNS}\",
    \"OSWORLD_SCREEN_DIFF_THRESHOLD\": \"${OSWORLD_SCREEN_DIFF_THRESHOLD}\",
    \"OSWORLD_SERVER_URL\": \"${OSWORLD_SERVER_URL}\",
    \"OSWORLD_TIMEOUT\": \"${OSWORLD_TIMEOUT}\",
    \"OSWORLD_REPLAY_BUFFER\": \"${OSWORLD_REPLAY_BUFFER}\",
    \"OSWORLD_REPLAY_THRESHOLD\": \"${OSWORLD_REPLAY_THRESHOLD}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\"
  }
}"

# Run training
echo "GSPO Training: ${MODEL_NAME} | GPUs: $NUM_GPUS | Output: $OUTPUT_DIR"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python3 "$SLIME_DIR/train.py" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${MULTIMODAL_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GSPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${BACKEND_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    "${WANDB_ARGS[@]}"

echo "GSPO training complete. Checkpoint: $OUTPUT_DIR"
