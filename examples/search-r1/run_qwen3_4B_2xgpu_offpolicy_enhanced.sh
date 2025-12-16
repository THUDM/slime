#!/bin/bash

# Enhanced Off-policy GRPO training script for Qwen3-4B with 2xGPU
# Based on run_qwen3_4B_2xgpu_offpolicy.sh with extensible buffer sampling strategies
#
# NEW FEATURES:
# - Staleness-aware FIFO buffer sampling (default)
# - Priority-based sampling support
# - Configurable sample reuse
# - Multiple sampling strategy options

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
   # --save /root/qwen3-4B_slime_offpolicy_enhanced/
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
   # --rollout-max-response-len 4096
   --rollout-temperature 1

   # eval args
   # --eval-interval 25
   # --eval-prompt-data nq_test /root/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:3000]
   # # --eval-prompt-data nq_test /root/nq_search/test.parquet
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
# This section contains parameters specific to off-policy GRPO with decoupled PPO

OFFPOLICY_GRPO_ARGS=(
   # Advantage estimator (kept as grpo for compatibility)
   --advantage-estimator grpo

   # Use decoupled policy loss instead of standard policy loss
   --loss-type decoupled_policy_loss
   # --loss-type policy_loss

   # === Staleness Control ===
   # Maximum allowed staleness (η in the paper)
   # - η=0: synchronous (on-policy), equivalent to standard GRPO
   # - η=1: allows data from 1 version ago
   # - η=5: allows data from up to 5 versions ago (more aggressive off-policy)
   # Recommended: start with 2-3 for moderate off-policy
   --max-staleness 2

   # === PPO Clipping Parameters ===
   # Asymmetric clipping: allows more aggressive positive updates
   --eps-clip 0.2
   --eps-clip-high 0.28

   # === KL Divergence Control ===
   # Use KL loss to prevent policy from deviating too far from reference
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl

   # === Entropy Regularization ===
   # Set to 0.0 for less exploration (suitable for Search-R1)
   --entropy-coef 0.00

   # === Importance Weight Clipping (optional) ===
   # Clip importance weights π_prox/π_behav to prevent extreme values
   # Uncomment to enable:
   # --importance-weight-clip-min 0.1
   # --importance-weight-clip-max 10.0

   # === Proximal Policy Update ===
   # The proximal policy is automatically set to the model parameters
   # from the previous training step (before gradient update)
   # No additional configuration needed
)


# ==================== ALTERNATIVE CONFIGURATIONS ====================
# Uncomment one of the following to try different sampling strategies

# === CONFIGURATION 2: Sample reuse for higher data efficiency (CURRENT) ===
BUFFER_SAMPLING_ARGS=(
   --buffer-max-size 1024
   --buffer-sampling-strategy fifo_staleness
   --buffer-remove-on-sample false
   --buffer-reuse-samples 1000  # Each sample can be used up to x times
)

# === CONFIGURATION 3: Random sampling for maximum diversity ===
# BUFFER_SAMPLING_ARGS=(
#    --buffer-max-size 500
#    --buffer-sampling-strategy random
#    --buffer-random-seed 42
#    --buffer-remove-on-sample true
# )

# === CONFIGURATION 4: Priority-based with normalized scoring (RECOMMENDED FOR TUNING) ===
# This configuration uses normalized priority scores for easier parameter tuning:
# - Base scores (reward/advantage) normalized to [0, 1]
# - Staleness normalized to [0, 1]
# - priority_weight and staleness_penalty now have interpretable ranges
# - Higher priority_weight → favor high-reward samples
# - Higher staleness_penalty → favor fresh samples
#
# BUFFER_SAMPLING_ARGS=(
#    --buffer-max-size 1024
#    --buffer-sampling-strategy priority
#    --buffer-priority-metric reward
#    --buffer-priority-weight 1.0  # 2x weight on reward
#    --buffer-staleness-penalty 0.1  # 0.5x weight on staleness
#    --buffer-normalize-priority-scores true  # Enable normalization
#    --buffer-priority-norm-method minmax  # Min-max normalization
#    --buffer-remove-on-sample false
#    --buffer-reuse-samples 1000
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
   --wandb-project slime-search-r1-offpolicy-enhanced
   --wandb-group qwen3-4B-2xgpu-offpolicy-eta3-enhanced
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


#  bash examples/search-r1/run_qwen3_4B_2xgpu_offpolicy_enhanced.sh 2>&1 | tee debug_output.log