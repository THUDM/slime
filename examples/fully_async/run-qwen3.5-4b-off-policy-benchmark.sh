#!/bin/bash
#
# Benchmark fully async rollout on 8x H100 with Qwen3.5-4B.
#
# Usage:
#   MODE=one_step_off      bash examples/fully_async/run-qwen3.5-4b-off-policy-benchmark.sh
#   MODE=fully_async       bash examples/fully_async/run-qwen3.5-4b-off-policy-benchmark.sh
#   MODE=window_partial    bash examples/fully_async/run-qwen3.5-4b-off-policy-benchmark.sh
#   MODE=staleness_partial bash examples/fully_async/run-qwen3.5-4b-off-policy-benchmark.sh
#
# Modes:
#   one_step_off      - default rollout, one-step off-policy async baseline, not support partial rollout
#   fully_async       - fully async rollout, no staleness control, not support partial rollout
#   window_partial    - fully async + version-window eviction + partial rollout + mask off-policy
#   staleness_partial - fully async + staleness backpressure + partial rollout + mask off-policy

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-4B.sh"

# --- Paths (adjust to your environment) ---
HF_CHECKPOINT=${HF_CHECKPOINT:-"/root/Qwen3.5-4B"}
REF_LOAD=${REF_LOAD:-"/root/Qwen3.5-4B_torch_dist"}
LOAD_PATH=${LOAD_PATH:-"/root/Qwen3.5-4B_slime_async_${MODE}/"}
SAVE_PATH=${SAVE_PATH:-"/root/Qwen3.5-4B_slime_async_${MODE}/"}
PROMPT_SET=${PROMPT_SET:-"/root/dapo-math-17k/dapo-math-17k.jsonl"}
# EVAL_DATASET=${EVAL_DATASET:-"/root/aime-2024/aime-2024.jsonl"}
MODE=${MODE:-"one_step_off"}
FULLY_ASYNC_VERSION_WINDOW=${FULLY_ASYNC_VERSION_WINDOW:-1}
FULLY_ASYNC_MAX_COMPLETED_SAMPLES=${FULLY_ASYNC_MAX_COMPLETED_SAMPLES:-128}
FULLY_ASYNC_EVICTION_POLICY=${FULLY_ASYNC_EVICTION_POLICY:-"drop_oldest_version"}
echo "=== Running fully async benchmark: mode=${MODE} ==="

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --load ${LOAD_PATH}
   --save ${SAVE_PATH}
   --save-interval 20
   --no-save-optim
   --no-load-optim
)

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 40
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data

   --update-weights-interval 2
)


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

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.9
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-max-running-requests 256
)

WANDB_ARGS=(
#    --use-wandb
#    --wandb-project slime-async-release
#    --wandb-group qwen3.5-4B-async-${MODE}
#    --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# --- Fully async rollout args (shared by all fully_async-based modes) ---
FULLY_ASYNC_ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --fully-async-debug-version-tracking
)

# --- Mode-specific flags ---
MODE_ARGS=()
case "${MODE}" in
    one_step_off)
        ;;
    fully_async)
        MODE_ARGS+=("${FULLY_ASYNC_ROLLOUT_ARGS[@]}")
        ;;
    window_partial)
        MODE_ARGS+=(
            "${FULLY_ASYNC_ROLLOUT_ARGS[@]}"
            --fully-async-buffer-policy window_evict
            --fully-async-version-window "${FULLY_ASYNC_VERSION_WINDOW}"
            --fully-async-max-completed-samples "${FULLY_ASYNC_MAX_COMPLETED_SAMPLES}"
            --fully-async-eviction-policy "${FULLY_ASYNC_EVICTION_POLICY}"
            --partial-rollout
            --mask-offpolicy-in-partial-rollout
        )
        ;;
    staleness_partial)
        MODE_ARGS+=(
            "${FULLY_ASYNC_ROLLOUT_ARGS[@]}"
            --fully-async-buffer-policy legacy_backpressure
            --staleness-threshold 0.5
            --partial-rollout
            --mask-offpolicy-in-partial-rollout
        )
        ;;
    *)
        echo "Unknown MODE: ${MODE}. Use one of: one_step_off, fully_async, window_partial, staleness_partial"
        exit 1
        ;;
esac

# launch ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# 4 GPUs for training, 4 GPUs for rollout (sglang)
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${MODE_ARGS[@]}