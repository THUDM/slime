#!/bin/bash

# Multi-Teacher On-Policy Distillation (MOPD) — Single Teacher SGLang Mode
# Model: Qwen3.5-35B-A3B (MoE, 256 experts, 8 active)
# Environment: 8× H20 (143GB)
# Layout: 4 GPUs for SGLang rollout, 4 GPUs for Megatron training
# Teacher: Same as student (self-distillation for connectivity validation only)
# Mode: SGLang (teacher runs on external SGLang server, no architecture constraint)
#
# This script is for MOPD E2E connectivity validation only.
# In production, use a DIFFERENT (stronger) model as teacher.
#
# usage: bash examples/multi_teacher_on_policy_distillation/run-qwen3.5-35B-A3B-mopd-sglang.sh

# ============================================================================
# Cleanup: kill existing SGLang / Ray / Python processes
# ============================================================================
pkill -9 sglang
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

# ============================================================================
# 1. Configure and start teacher model server (self-distillation for testing)
# ============================================================================
# For this connectivity test, the teacher is the same model as the student.
# In production, replace with a stronger model (e.g., Qwen3-72B or domain expert).
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
TEACHER_LOG_FILE="/tmp/sglang_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

# Launch teacher on GPU 0-3 (4 GPUs for TP=4, or adjust TP as needed)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
    --model-path /mnt4/data/open_source/Qwen3.5-35B-A3B/ \
    --host 0.0.0.0 \
    --port $TEACHER_PORT \
    --tp 4 \
    --ep-size 4 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.7 \
    > "$TEACHER_LOG_FILE" 2>&1 &

TEACHER_PID=$!
echo "Starting teacher model server (PID: $TEACHER_PID)..."

# Wait for teacher server to be ready
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting for teacher model server to start..."
    tail -n 10 "$TEACHER_LOG_FILE" 2>/dev/null || true
    sleep 10
done
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."

# ============================================================================
# 2. Set environment variables
# ============================================================================

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/mntfn/yanyi/code/slime/scripts/models/qwen3.5-35B-A3B.sh"

# MOPD teachers JSON config
export MOPD_TEACHERS_JSON='[{"name":"self_teacher","domain":"default"}]'

# MOPD teacher URLs
export MOPD_TEACHER_URLS="{\"default\":\"http://$TEACHER_IP:$TEACHER_PORT/generate\"}"

# ============================================================================
# 3. Configure training arguments
# ============================================================================

# IMPORTANT: Before running this script, convert the HF checkpoint to Megatron
# torch_dist format:
#
#   cd /mntfn/yanyi/code/slime
#   source scripts/models/qwen3.5-35B-A3B.sh
#
#   PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
#       ${MODEL_ARGS[@]} \
#       --hf-checkpoint /mnt4/data/open_source/Qwen3.5-35B-A3B \
#       --save /mnt4/data/open_source/Qwen3.5-35B-A3B_torch_dist

CKPT_ARGS=(
   --hf-checkpoint /mnt4/data/open_source/Qwen3.5-35B-A3B/
   --ref-load /mnt4/data/open_source/Qwen3.5-35B-A3B_torch_dist/
   --load /mnt4/data/zhixiaobao/yanyi/Qwen3.5-35B-A3B-mopd-test/
   --save /mnt4/data/zhixiaobao/yanyi/Qwen3.5-35B-A3B-mopd-test/
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data /mntfn/yanyi/dataset/train_text_user_only.jsonl
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 10                 # Small for connectivity test
   --rollout-batch-size 16
   --n-samples-per-prompt 1         # No need for multiple samples in pure distillation
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 16
   --balance-data
)

# For MOPD SGLang mode, we use the MOPD reward_func and post_process_rewards
# The --rm-url is used as the default/fallback URL; per-teacher URLs come from MOPD_TEACHER_URLS env var
RM_ARGS=(
    --custom-rm-path slime.rollout.mopd.reward_func
    --custom-reward-post-process-path slime.rollout.mopd.post_process_rewards
    --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

EVAL_ARGS=(
   # No eval for connectivity test
)

# Qwen3.5-35B-A3B with 4 GPUs for training:
#   TP=2, EP=4 (256 experts / 4 = 64 experts per GPU)
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

# MOPD Configuration (SGLang mode, single teacher)
# In SGLang mode, teacher log-probs are obtained by querying the teacher SGLang server
# during rollout. No teacher model is loaded into Megatron training memory.
MOPD_ARGS=(
   --advantage-estimator grpo

   # MOPD flags — single teacher
   --use-mopd
   # Note: --mopd-teachers is read from $MOPD_TEACHERS_JSON env var (see above)
   # to avoid shell quoting issues with JSON in ray job submit.

   # No --mopd-teacher-loads needed in SGLang mode!
   # Teacher log-probs come from the SGLang server via reward_func.

   # MOPD hyperparameters
   --mopd-alpha 0.0                # Pure distillation, no ORM
   --mopd-eps-low 0.2              # IS weight lower bound
   --mopd-eps-high 5.0             # IS weight upper bound
   --mopd-sampling-logprobs-key rollout_log_probs

   # Standard training flags
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7                       # Conservative LR for stability
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3.5-35B-mopd-sglang
   # --wandb-key ${WANDB_KEY}
)

# SGLang rollout config: 4 GPUs for rollout
SGLANG_ARGS=(
   --rollout-num-gpus 4              # 4 GPUs for SGLang rollout engine
   --rollout-num-gpus-per-engine 4   # 4 GPUs per engine (TP=4 for Qwen3.5-35B-A3B)
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 4                # Match EP=4 for MoE expert parallelism
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   # MoE communication
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)

# ============================================================================
# 4. Launch training
# ============================================================================

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# 8 GPUs total: 4 for SGLang rollout (GPU 0-3, already used by teacher server),
# 4 for Megatron training (GPU 4-7)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON=$(python3 -c "
import json, os
env = {
    'PYTHONPATH': '/root/Megatron-LM/',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'NCCL_NVLS_ENABLE': os.environ.get('HAS_NVLINK', '0'),
    'MOPD_TEACHER_URLS': os.environ.get('MOPD_TEACHER_URLS', ''),
    'MOPD_TEACHERS_JSON': os.environ.get('MOPD_TEACHERS_JSON', '')
}
print(json.dumps({'env_vars': env}))
")

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${MOPD_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

# ============================================================================
# 5. Cleanup
# ============================================================================
kill $TEACHER_PID 2>/dev/null || true
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python