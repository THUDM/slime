#!/bin/bash

# Off-policy GRPO training with HTTP Buffer for agent tasks
# This example demonstrates HTTP buffer usage for large-scale scenarios
#
# Prerequisites:
#   1. Start the HTTP buffer server first:
#      cd slime_plugins/rollout_buffer && python buffer.py
#   2. Verify server is running at http://localhost:8889

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
   # --save /root/qwen3-4B_slime_offpolicy_http/
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

   # === Staleness Control ===
   # Maximum allowed staleness (η in the paper)
   # - η=0: synchronous (on-policy), equivalent to standard GRPO
   # - η=1: allows data from 1 version ago
   # - η=5: allows data from up to 5 versions ago (more aggressive off-policy)
   # Recommended: start with 2-3 for moderate off-policy
   --max-staleness 3

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

# ==================== HTTP BUFFER CONFIGURATION ====================
# This section demonstrates HTTP buffer usage for large-scale agent tasks
#
# Key differences from in-process buffer:
#   - Decouples data generation from training
#   - Enables horizontal scaling
#   - Supports async trajectory generation
#   - ~10ms latency vs ~0.1ms for in-process

BUFFER_ARGS=(
   # === Enable HTTP Buffer ===
   --buffer-mode http

   # === Buffer Server Configuration ===
   # URL of the HTTP buffer server (must be started separately)
   # Start with: cd slime_plugins/rollout_buffer && python buffer.py
   --buffer-server-url http://localhost:8889

   # === Task Type ===
   # Task type determines which generator to use
   # Options: 'grpo' (standard), 'math', 'tool', or custom types
   # Custom generators: slime_plugins/rollout_buffer/generator/{task_type}_generator.py
   --buffer-task-type grpo

   # === Buffer Size ===
   # Maximum number of sample groups to store in buffer
   # Larger buffer = more diversity but more memory usage
   --buffer-max-size 1000

   # === HTTP Configuration ===
   # Request timeout (seconds) - increase for slow agent tasks
   --buffer-timeout 30

   # Maximum retries for failed HTTP requests
   --buffer-max-retries 3
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
   --wandb-project slime-search-r1-http-buffer
   --wandb-group qwen3-4B-2xgpu-http-buffer-demo
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
echo "HTTP Buffer Configuration:"
echo "  Mode: http"
echo "  Server URL: http://localhost:8889"
echo "  Task Type: grpo"
echo "  Buffer Size: 1000"
echo "========================================"
echo ""
echo "IMPORTANT: Make sure the HTTP buffer server is running!"
echo "If not, start it with:"
echo "  cd slime_plugins/rollout_buffer && python buffer.py"
echo ""
echo "========================================"

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
   ${BUFFER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
