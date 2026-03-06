#!/bin/bash

# ==================== UNIFIED Off-policy GRPO Training Script ====================
# Qwen3-4B with 2xGPU - Flexible configuration via environment variables
#
# USAGE:
#   # Default (off-policy with random sampling):
#   bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#
#   # Different configurations:
#   MODE=offpolicy_hybrid bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#   MODE=offpolicy_vanilla bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#   MODE=onpolicy bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#   MODE=http bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#
# MODES:
#   - offpolicy_random (default): staleness=4, random sampling, m2po=0.16, decoupled_policy_loss
#   - offpolicy_hybrid: staleness=4, hybrid sampling (20% lifo + 80% priority), m2po=0.04, train_iters=2
#   - offpolicy_vanilla: staleness=4, random sampling, m2po=0.16, policy_loss (baseline, no correction)
#   - onpolicy: staleness=0, standard on-policy training, m2po=0.0, policy_loss
#   - http: HTTP buffer mode, staleness=4, requires external buffer server
#
# CUSTOMIZATION:
#   Override any parameter with environment variables:
#   MAX_STALENESS=16 BUFFER_SIZE=2048 M2PO_THRESHOLD=0.32 bash run_qwen3_4B_2xgpu_offpolicy_unified.sh

# ==================== CONFIGURATION MODE ====================
MODE=${MODE:-"offpolicy_random"}

# Clean up previous processes
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "/mnt/shared-storage-user/puyuan/code/slime/scripts/models/qwen3-4B.sh"

# ==================== MODE-SPECIFIC DEFAULTS ====================
case "$MODE" in
  offpolicy_random)
    : ${MAX_STALENESS:=4}
    : ${BUFFER_STRATEGY:="random"}
    : ${M2PO_THRESHOLD:=0.16}
    : ${LOSS_TYPE:="decoupled_policy_loss"}
    : ${TRAIN_ITERS_PER_ROLLOUT:=1}
    : ${WANDB_GROUP:="qwen3-4B-2xgpu-offpolicy-random"}
    ;;
  offpolicy_hybrid)
    : ${MAX_STALENESS:=4}
    : ${BUFFER_STRATEGY:="hybrid"}
    : ${HYBRID_LIFO_RATIO:=0.2}
    : ${HYBRID_PRIORITY_RATIO:=0.8}
    : ${M2PO_THRESHOLD:=0.04}
    : ${LOSS_TYPE:="decoupled_policy_loss"}
    : ${TRAIN_ITERS_PER_ROLLOUT:=2}
    : ${UPDATE_POLICY_EVERY_ITER:="true"}
    : ${WANDB_GROUP:="qwen3-4B-2xgpu-offpolicy-hybrid"}
    ;;
  offpolicy_vanilla)
    # BASELINE: Test stale data with vanilla policy_loss (no decoupled correction)
    # Purpose: Measure performance degradation without off-policy correction
    # Comparison: vs offpolicy_random to validate decoupled_policy_loss effectiveness
    : ${MAX_STALENESS:=4}
    : ${BUFFER_STRATEGY:="random"}
    : ${M2PO_THRESHOLD:=0.16}
    : ${LOSS_TYPE:="policy_loss"}
    : ${USE_ROLLOUT_LOGPROBS:="true"}
    : ${TRAIN_ITERS_PER_ROLLOUT:=1}
    : ${WANDB_GROUP:="qwen3-4B-2xgpu-offpolicy-vanilla-baseline"}
    ;;
  onpolicy)
    : ${MAX_STALENESS:=0}
    : ${BUFFER_STRATEGY:="none"}
    : ${M2PO_THRESHOLD:=0.0}
    : ${LOSS_TYPE:="policy_loss"}
    : ${USE_ROLLOUT_LOGPROBS:="true"}
    : ${DISABLE_BUFFER:="true"}
    : ${TRAIN_ITERS_PER_ROLLOUT:=1}
    : ${WANDB_GROUP:="qwen3-4B-2xgpu-onpolicy"}
    ;;
  http)
    : ${MAX_STALENESS:=4}
    : ${BUFFER_MODE:="http"}
    : ${BUFFER_SERVER_URL:="http://localhost:8889"}
    : ${BUFFER_TASK_TYPE:="grpo"}
    : ${M2PO_THRESHOLD:=0.16}
    : ${LOSS_TYPE:="decoupled_policy_loss"}
    : ${TRAIN_ITERS_PER_ROLLOUT:=1}
    : ${WANDB_GROUP:="qwen3-4B-2xgpu-http-buffer"}
    ;;
  *)
    echo "Unknown MODE: $MODE. Valid options: offpolicy_random, offpolicy_hybrid, offpolicy_vanilla, onpolicy, http"
    exit 1
    ;;
esac

# ==================== CUSTOMIZABLE PARAMETERS ====================
: ${BUFFER_SIZE:=1024}
: ${BUFFER_REUSE:=${MAX_STALENESS}}
: ${ROLLOUT_BATCH_SIZE:=64}
: ${GLOBAL_BATCH_SIZE:=256}
: ${IMP_WEIGHT_MIN:=0.5}
: ${IMP_WEIGHT_MAX:=2.0}
: ${BEHAV_IMP_WEIGHT_CAP:=5.0}
: ${EPS_CLIP:=0.2}
: ${EPS_CLIP_HIGH:=0.28}
: ${WANDB_PROJECT:="slime-search-r1-offpolicy-unified"}

echo "========================================"
echo "Running in MODE: $MODE"
echo "MAX_STALENESS: $MAX_STALENESS"
echo "LOSS_TYPE: $LOSS_TYPE"
echo "BUFFER_STRATEGY: ${BUFFER_STRATEGY:-N/A}"
echo "M2PO_THRESHOLD: ${M2PO_THRESHOLD:-N/A}"
echo "TRAIN_ITERS_PER_ROLLOUT: $TRAIN_ITERS_PER_ROLLOUT"
echo "========================================"

# ==================== CHECKPOINT CONFIGURATION ====================
CKPT_ARGS=(
   --hf-checkpoint /mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B/
   --ref-load /mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B_torch_dist/
)

# ==================== ROLLOUT CONFIGURATION ====================
ROLLOUT_ARGS=(
   --prompt-data /mnt/shared-storage-user/puyuan/code/Search-R1/data/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

# Add train_iters_per_rollout if > 1
if [ "$TRAIN_ITERS_PER_ROLLOUT" -gt 1 ]; then
   ROLLOUT_ARGS+=(--train_iters_per_rollout ${TRAIN_ITERS_PER_ROLLOUT})
fi

# Add update_policy_version_every_train_iter if enabled
if [ "${UPDATE_POLICY_EVERY_ITER}" == "true" ]; then
   ROLLOUT_ARGS+=(--update_policy_version_every_train_iter)
fi

# ==================== PERFORMANCE CONFIGURATION ====================
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
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# ==================== OFF-POLICY GRPO CONFIGURATION ====================
OFFPOLICY_GRPO_ARGS=(
   --advantage-estimator grpo
   --max-staleness ${MAX_STALENESS}
   --eps-clip ${EPS_CLIP}
   --eps-clip-high ${EPS_CLIP_HIGH}
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --importance-weight-clip-min ${IMP_WEIGHT_MIN}
   --importance-weight-clip-max ${IMP_WEIGHT_MAX}
   --behav-imp-weight-cap ${BEHAV_IMP_WEIGHT_CAP}
)

# Loss type configuration
if [ "$LOSS_TYPE" == "policy_loss" ]; then
   OFFPOLICY_GRPO_ARGS+=(--loss-type policy_loss)
   if [ "${USE_ROLLOUT_LOGPROBS}" == "true" ]; then
      OFFPOLICY_GRPO_ARGS+=(--use-rollout-logprobs)
   fi
else
   OFFPOLICY_GRPO_ARGS+=(--loss-type decoupled_policy_loss)
fi

# ==================== BUFFER CONFIGURATION ====================
if [ "$MODE" == "http" ]; then
   # HTTP Buffer mode
   BUFFER_SAMPLING_ARGS=(
      --buffer-mode http
      --buffer-server-url ${BUFFER_SERVER_URL}
      --buffer-task-type ${BUFFER_TASK_TYPE}
      --buffer-max-size ${BUFFER_SIZE:-1000}
      --buffer-timeout 30
      --buffer-max-retries 3
   )
   echo "HTTP Buffer enabled. Server URL: ${BUFFER_SERVER_URL}"
   echo "Make sure the buffer server is running: cd slime_plugins/rollout_buffer && python buffer.py"
elif [ "${DISABLE_BUFFER}" == "true" ]; then
   # On-policy mode: no buffer, no M2PO filtering
   BUFFER_SAMPLING_ARGS=()
   echo "On-policy mode: Buffer disabled"
else
   # In-process buffer mode (off-policy)
   BUFFER_SAMPLING_ARGS=(
      --buffer-max-size ${BUFFER_SIZE}
      --buffer-remove-on-sample false
      --buffer-reuse-samples ${BUFFER_REUSE}
   )

   # Add M2PO filtering only if threshold > 0
   if (( $(echo "$M2PO_THRESHOLD > 0" | bc -l) )); then
      BUFFER_SAMPLING_ARGS+=(
         --enable-m2po-filtering
         --m2po-threshold ${M2PO_THRESHOLD}
      )
   fi

   # Buffer sampling strategy
   case "$BUFFER_STRATEGY" in
      random)
         BUFFER_SAMPLING_ARGS+=(--buffer-sampling-strategy random)
         ;;
      lifo)
         BUFFER_SAMPLING_ARGS+=(--buffer-sampling-strategy lifo_staleness)
         ;;
      priority)
         BUFFER_SAMPLING_ARGS+=(
            --buffer-sampling-strategy priority
            --buffer-priority-metric reward
         )
         ;;
      hybrid)
         BUFFER_SAMPLING_ARGS+=(
            --buffer-sampling-strategy hybrid
            --buffer-hybrid-lifo-ratio ${HYBRID_LIFO_RATIO:-0.2}
            --buffer-hybrid-priority-ratio ${HYBRID_PRIORITY_RATIO:-0.8}
            --buffer-priority-metric reward
         )
         ;;
   esac
fi

# ==================== OPTIMIZER CONFIGURATION ====================
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ==================== WANDB CONFIGURATION ====================
export WANDB_MODE="offline"
export WANDB_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${WANDB_PROJECT}
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_KEY}
)

# ==================== SGLANG CONFIGURATION ====================
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)

# ==================== MISC CONFIGURATION ====================
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ==================== FORMAT REWARD CONFIGURATION ====================
FORMAT_REWARD_ARGS=(
   --enable-format-reward
   --structure-format-score 0.2
   --retrieval-score 0.1
   --final-format-score 0.1
)

# ==================== CUSTOM CONFIGURATION ====================
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func
)

# ==================== RAY SETUP ====================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

# ==================== LAUNCH TRAINING ====================
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${OFFPOLICY_GRPO_ARGS[@]} \
   ${BUFFER_SAMPLING_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${FORMAT_REWARD_ARGS[@]} \
   ${CUSTOM_ARGS[@]}

# ==================== USAGE EXAMPLES ====================
#
# 1. Default (off-policy with random sampling):
#    bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#
# 2. Use off-policy hybrid sampling:
#    MODE=offpolicy_hybrid bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#
# 3. Use off-policy vanilla (baseline without correction):
#    MODE=offpolicy_vanilla bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#
# 4. Use on-policy training:
#    MODE=onpolicy bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#
# 5. Use HTTP buffer:
#    MODE=http bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#
# 6. Custom staleness with off-policy random:
#    MAX_STALENESS=8 M2PO_THRESHOLD=0.32 bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#
# 7. Custom buffer strategy:
#    BUFFER_STRATEGY=priority bash run_qwen3_4B_2xgpu_offpolicy_unified.sh
#
# 8. Logging to file:
#    bash run_qwen3_4B_2xgpu_offpolicy_unified.sh 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
#
# ==================== KEY PARAMETERS GUIDE ====================
#
# MAX_STALENESS: Controls how old samples can be used
#   - 0: On-policy (synchronous)
#   - 1-4: Conservative off-policy, more on-policy behavior
#   - 8-16: Aggressive off-policy, more sample reuse
#
# BUFFER_STRATEGY: Sample selection strategy (off-policy only)
#   - random: Uniform sampling from buffer
#   - lifo: Newest samples first (prioritize recent data)
#   - priority: High-reward samples first (prioritize quality)
#   - hybrid: Mix of lifo and priority (balanced approach)
#
# M2PO_THRESHOLD: Minimum advantage filtering threshold
#   - 0.0: No filtering (on-policy or aggressive off-policy)
#   - 0.04: Low threshold, allows most samples (hybrid strategy)
#   - 0.16: Medium threshold, filters low-advantage samples (random strategy)
#   - 0.32+: High threshold, very conservative filtering
#   - Recommended: Scale with staleness (0.04 * staleness)
#
# LOSS_TYPE:
#   - decoupled_policy_loss: Decoupled PPO for off-policy (recommended for staleness > 0)
#   - policy_loss: Standard PPO for on-policy (staleness = 0)
#
# TRAIN_ITERS_PER_ROLLOUT: Training steps per rollout
#   - 1: Update once per rollout (standard)
#   - 2+: Multiple updates per rollout (faster convergence but may be less stable)
#
# ==================== MODE COMPARISON ====================
#
# offpolicy_random:
#   - Best for: Stable baseline, exploratory training
#   - Pros: Simple, predictable, moderate sample reuse
#   - Cons: May not fully utilize high-quality samples
#
# offpolicy_hybrid:
#   - Best for: Sample efficiency, quality-driven training
#   - Pros: Prioritizes high-reward samples, faster convergence
#   - Cons: Requires more tuning, may overfit to high-reward regions
#
# offpolicy_vanilla:
#   - Best for: Baseline comparison, measuring off-policy correction effectiveness
#   - Pros: Tests stale data without correction, isolates decoupled_policy_loss impact
#   - Cons: Expected to perform worse than offpolicy_random, training may be unstable
#
# onpolicy:
#   - Best for: Maximum stability, benchmarking
#   - Pros: Most stable, well-understood behavior
#   - Cons: No sample reuse, lower sample efficiency
#
# http:
#   - Best for: Distributed training, async data generation
#   - Pros: Decoupled rollout/training, horizontal scaling
#   - Cons: Additional latency (~10ms), requires separate server
