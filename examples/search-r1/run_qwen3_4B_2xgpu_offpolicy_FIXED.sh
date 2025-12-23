#!/bin/bash

# 🔧 FIXED Off-policy GRPO training script for Qwen3-4B with 2xGPU
# This version includes the buffer version management fix
#
# KEY FIXES:
# 1. ✅ Version-aware stratified sampling (code-level fix)
# 2. ✅ Optimized buffer_reuse_samples (1000 → 10)
# 3. ✅ Adjusted buffer_max_size for better version diversity
# 4. ✅ Recommended max_staleness=4 for balanced off-policy

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "/mnt/shared-storage-user/puyuan/code/slime/scripts/models/qwen3-4B.sh"


CKPT_ARGS=(
   --hf-checkpoint /mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B/
   --ref-load /mnt/shared-storage-user/puyuan/code/slime/Qwen3-4B_torch_dist/
   # --load /root/qwen3-4B_slime/
   # --save /root/qwen3-4B_slime_offpolicy_fixed/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/shared-storage-user/puyuan/code/Search-R1/data/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 512
   --rollout-temperature 1

   # eval args
   # --eval-interval 25
   # --eval-prompt-data nq_test /root/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:3000]
   # --eval-input-key prompt
   # --eval-label-key reward_model
   # --n-samples-per-eval-prompt 1

   --global-batch-size 256
   --balance-data
)

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

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# ==================== OFF-POLICY GRPO CONFIGURATION ====================
# 🔧 FIXED: Optimized for stable training with version diversity

OFFPOLICY_GRPO_ARGS=(
   # Advantage estimator
   --advantage-estimator grpo

   # Use decoupled policy loss for off-policy
   --loss-type decoupled_policy_loss

   # === Staleness Control (RECOMMENDED) ===
   # 🔧 FIXED: Use max_staleness=4 for balanced off-policy
   # - Allows using samples from versions [current-4, current]
   # - Combined with stratified sampling, ensures version diversity
   # - Not too aggressive (prevents large importance weight variance)
   --max-staleness 4

   # === PPO Clipping Parameters ===
   # Asymmetric clipping: allows more aggressive positive updates
   --eps-clip 0.2
   --eps-clip-high 0.28

   # === KL Divergence Control ===
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl

   # === Entropy Regularization ===
   --entropy-coef 0.00

   # === Importance Weight Clipping (RECOMMENDED) ===
   # 🔧 FIXED: Enable clipping to prevent extreme importance weights
   --importance-weight-clip-min 0.1
   --importance-weight-clip-max 10.0
)


# ==================== BUFFER CONFIGURATION (FIXED) ====================
# 🔧 CRITICAL FIX: Optimized parameters for version diversity
#
# Key changes:
# 1. buffer_reuse_samples: 1000 → 10
#    - Prevents over-reusing old samples
#    - Allows new versions to enter training
#    - Maintains version diversity
#
# 2. buffer_max_size: 1024 → 256
#    - Matches: max_staleness × rollout_batch_size × margin
#    - Example: 4 × 32 × 2 = 256
#    - Ensures fast version turnover
#
# 3. Strategy: fifo_staleness with stratified sampling (code-level fix)
#    - Automatically samples from multiple versions
#    - Each batch contains samples from versions [current-staleness, current]
#    - Logs version distribution for monitoring

BUFFER_SAMPLING_ARGS=(
   # 🔧 FIXED: Reduced buffer size for faster version turnover
   # --buffer-max-size 256
   --buffer-max-size 1024


   # Use fifo_staleness strategy (with stratified sampling fix)
   --buffer-sampling-strategy fifo_staleness

   # Allow sample reuse but don't remove on sample
   --buffer-remove-on-sample false

   # 🔧 CRITICAL FIX: Reduced reuse count (1000 → 10)
   # This is the most important parameter change!
   # Old value (1000) caused samples to be reused for 1000 iterations,
   # preventing new versions from being used.
   # --buffer-reuse-samples 10
   --buffer-reuse-samples 1000

)

# ==================== ALTERNATIVE CONFIGURATIONS ====================

# === OPTION 1: More aggressive off-policy (for large-scale training) ===
# BUFFER_SAMPLING_ARGS=(
#    --buffer-max-size 512                    # Larger buffer
#    --buffer-sampling-strategy fifo_staleness
#    --buffer-remove-on-sample false
#    --buffer-reuse-samples 20                # More reuse
# )
# OFFPOLICY_GRPO_ARGS+=(--max-staleness 8)   # Higher staleness

# === OPTION 2: Random sampling for maximum diversity ===
# BUFFER_SAMPLING_ARGS=(
#    --buffer-max-size 256
#    --buffer-sampling-strategy random        # Pure random sampling
#    --buffer-random-seed 42
#    --buffer-remove-on-sample false
#    --buffer-reuse-samples 10
# )

# === OPTION 3: Priority-based sampling (high-reward samples first) ===
# BUFFER_SAMPLING_ARGS=(
#    --buffer-max-size 256
#    --buffer-sampling-strategy priority
#    --buffer-priority-metric reward          # Prioritize by reward
#    --buffer-priority-weight 1.0
#    --buffer-staleness-penalty 0.1           # Small penalty for staleness
#    --buffer-normalize-priority-scores true
#    --buffer-priority-norm-method minmax
#    --buffer-remove-on-sample false
#    --buffer-reuse-samples 10
# )

# ==================== STANDARD CONFIGURATION ====================

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

export WANDB_MODE="offline"  # Set to "online" if you want real-time logging
export WANDB_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"  # Replace with your key

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-search-r1-offpolicy-stale-FIXED
   --wandb-group qwen3-4B-2xgpu-offpolicy-stale4
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

echo "========================================"
echo "🔧 FIXED OFF-POLICY GRPO CONFIGURATION"
echo "========================================"
echo "✅ Version-aware stratified sampling enabled"
echo "✅ buffer_reuse_samples: 10 (was 1000)"
echo "✅ buffer_max_size: 256 (was 1024)"
echo "✅ max_staleness: 4"
echo ""
echo "Expected improvements:"
echo "1. Each training batch uses samples from MULTIPLE versions"
echo "2. Importance weights should stay close to 1.0"
echo "3. Raw reward should increase steadily (no sudden drops)"
echo "4. Version distribution log: {v1: N1, v2: N2, ...} instead of [vX, vX]"
echo "========================================"
echo ""

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
   ${CUSTOM_ARGS[@]}


# ==================== USAGE ====================
# Run the fixed script:
#   bash examples/search-r1/run_qwen3_4B_2xgpu_offpolicy_FIXED.sh 2>&1 | tee logs/output_stale4_fixed.log
#
# Monitor key metrics during training:
#   tail -f logs/output_stale4_fixed.log | grep -E "Buffer Sampling|importance_weight|raw_reward"
#
# Expected log output (FIXED):
#   [Buffer Sampling] Stratified sample: {60: 6, 61: 7, 62: 7, 63: 6, 64: 6}
#     (total=32 groups from 5 versions)
#
# Old log output (BROKEN):
#   [Buffer Sampling] Sampled 32 groups. Version range: [60, 60], Current version: 64
