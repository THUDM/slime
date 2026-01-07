#!/bin/bash

# 🔧 FIXED Off-policy GRPO training script for Qwen3-4B with 2xGPU
#
# CRITICAL FIXES:
# 1. Changed buffer_reuse_samples from 1000 → 3 (prevent version label corruption)
# 2. Changed buffer_sampling_strategy from random → lifo_staleness (prioritize new samples)
# 3. Reduced buffer_max_size from 1024 → 256 (faster version turnover)
# 4. Increased max_staleness from 1 → 5 (match reuse count)
# 5. Tightened importance weight clipping 0.1-10.0 → 0.5-2.0 (reduce variance)

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
   # Advantage estimator: GRPO
   # IMPORTANT: GRPO advantages = reward (broadcasted to all tokens)
   # Unlike standard PPO, GRPO does NOT include KL penalty in advantages
   # See get_grpo_returns() in slime/utils/ppo_utils.py for implementation
   --advantage-estimator grpo

   # Use decoupled policy loss for off-policy
   --loss-type decoupled_policy_loss

   # === Staleness Control ===
   # 🔧 FIXED: Increased from 1 to 5 to match reuse count
   # - Allows using samples from versions [current-5, current]
   # - Balanced off-policy: not too aggressive, not too conservative
   # - Combined with LIFO sampling, ensures newest data is prioritized
   --max-staleness 5

   # === PPO Clipping Parameters ===
   # Asymmetric clipping: allows more aggressive positive updates
   --eps-clip 0.2
   --eps-clip-high 0.28

   # === KL Divergence Control ===
   # CRITICAL CLARIFICATION: This is NOT double penalization!
   # In GRPO, advantages contain ONLY rewards (no KL subtraction).
   # The KL constraint is enforced separately here via explicit KL loss.
   # This is different from algorithms where KL is subtracted from reward in advantages.
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl

   # === Entropy Regularization ===
   --entropy-coef 0.00

   # === Importance Weight Clipping ===
   # 🔧 FIXED: Tightened from [0.1, 10.0] to [0.5, 2.0]
   # - Dramatically reduces gradient variance
   # - Prevents weight explosion from old samples
   # - More conservative but much more stable
   --importance-weight-clip-min 0.5
   --importance-weight-clip-max 2.0
)


BUFFER_SAMPLING_ARGS=(
   # 🔧 FIXED: Reduced from 1024 to 256 for faster version turnover
   # - Smaller buffer means old samples are evicted faster
   # - Ensures buffer contains mostly recent data
   # - Reduces memory footprint
   --buffer-max-size 256

   # 🔧 CRITICAL FIX: Changed from random to lifo_staleness
   # - LIFO = Last-In-First-Out (newest samples first)
   # - Prioritizes samples where π_behave ≈ π_theta
   # - Minimizes importance weight variance
   # - See slime/utils/buffer_sampling_strategies.py:369
   --buffer-sampling-strategy lifo_staleness

   # Allow sample reuse but don't remove on sample
   --buffer-remove-on-sample false

   # 🔧 CRITICAL FIX: Reduced from 1000 to 3
   # - Prevents version label corruption
   # - Ensures policy_version accurately reflects sample age
   # - Key insight: reuse_count acts as "hidden staleness"
   # - With reuse=3 and staleness=5, effective max age = 5+3×0.5 ≈ 6.5 steps
   --buffer-reuse-samples 3
)


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
   --wandb-project slime-search-r1-offpolicy-stale-0107
   --wandb-group qwen3-4B-2xgpu-offpolicy-FIXED
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
echo "✅ buffer_sampling_strategy: lifo_staleness (was: random)"
echo "✅ buffer_reuse_samples: 3 (was: 1000)"
echo "✅ buffer_max_size: 256 (was: 1024)"
echo "✅ max_staleness: 5 (was: 1)"
echo "✅ importance_weight_clip: [0.5, 2.0] (was: [0.1, 10.0])"
echo ""
echo "Expected improvements:"
echo "1. Effective sample size should stay > 400 (was dropping to ~240)"
echo "2. Grad norm should stay < 5.0 (was exploding to 20+)"
echo "3. KL loss growth should be slower and more stable"
echo "4. Version distribution: newest samples dominate the batch"
echo "5. Training convergence should be smooth without sudden drops"
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
# bash /mnt/shared-storage-user/puyuan/code/slime/examples/search-r1/run_qwen3_4B_2xgpu_offpolicy_FIXED.sh 2>&1 | tee /mnt/shared-storage-user/puyuan/code/slime/examples/search-r1/logs/output_FIXED.log
#
# Monitor key metrics during training:
#   tail -f logs/output_FIXED.log | grep -E "Buffer Sampling|importance_weight|effective_sample_size|grad_norm"
#
# Expected log output:
#   [Buffer Sampling] LIFO sample: {218: 10, 219: 12, 220: 10} (total=32 groups, version_range=[218, 220], newest-first)
#   train/effective_sample_size: 450.5 (should stay > 400)
#   train/grad_norm: 2.3 (should stay < 5.0)
