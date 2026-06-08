#!/bin/bash

# usage:
#   bash examples/on_policy_distillation/run-qwen3-8B-qwen3.5-35B-A3B-cross-vocab-opd.sh
#
# This example trains a Qwen3-8B student with a Qwen3.5-35B-A3B SGLang teacher.
# The teacher and student prompts are rendered with their own chat templates, so
# the OPD hooks use cross-vocabulary alignment instead of sending student token IDs
# to the teacher.

set -ex

SLIME_ROOT=${SLIME_ROOT:-/root/slime}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
DATA_PATH=${DATA_PATH:-/root/dapo-math-17k/dapo-math-17k.jsonl}

STUDENT_HF_PATH=${STUDENT_HF_PATH:-/root/Qwen3-8B}
STUDENT_TORCH_DIST_PATH=${STUDENT_TORCH_DIST_PATH:-/root/Qwen3-8B_torch_dist}
STUDENT_SLIME_CKPT_PATH=${STUDENT_SLIME_CKPT_PATH:-/root/Qwen3-8B_slime}
TEACHER_HF_PATH=${TEACHER_HF_PATH:-/root/Qwen3.5-35B-A3B}

TEACHER_IP=${TEACHER_IP:-127.0.0.1}
TEACHER_PORT=${TEACHER_PORT:-13141}
TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES:-6,7}
TEACHER_TP=${TEACHER_TP:-2}
TEACHER_MEM_FRACTION_STATIC=${TEACHER_MEM_FRACTION_STATIC:-0.75}
TEACHER_CHUNKED_PREFILL_SIZE=${TEACHER_CHUNKED_PREFILL_SIZE:-4096}
LOG_FILE=${LOG_FILE:-/tmp/sglang_qwen35_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log}

SLIME_CUDA_VISIBLE_DEVICES=${SLIME_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
SLIME_NUM_GPUS=${SLIME_NUM_GPUS:-6}

CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES} python3 -m sglang.launch_server \
    --model-path ${TEACHER_HF_PATH} \
    --host 0.0.0.0 \
    --port ${TEACHER_PORT} \
    --tp ${TEACHER_TP} \
    --chunked-prefill-size ${TEACHER_CHUNKED_PREFILL_SIZE} \
    --mem-fraction-static ${TEACHER_MEM_FRACTION_STATIC} \
    > "${LOG_FILE}" 2>&1 &

echo "Starting Qwen3.5-35B-A3B teacher model server..."
until curl -sf http://${TEACHER_IP}:${TEACHER_PORT}/health_generate > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "${LOG_FILE}"
    sleep 5
done

curl http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info
echo "Teacher model server is up at ${TEACHER_IP}:${TEACHER_PORT}."
sleep 10

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${STUDENT_HF_PATH}
   --ref-load ${STUDENT_TORCH_DIST_PATH}
   --load ${STUDENT_SLIME_CKPT_PATH}
   --save ${STUDENT_SLIME_CKPT_PATH}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_PATH}
   --input-key prompt
   --apply-chat-template
   --opd-prompt-messages-key opd_messages
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 6
   --n-samples-per-prompt 2
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --global-batch-size 12
   --balance-data
)

RM_ARGS=(
   --custom-rm-path slime.rollout.on_policy_distillation.reward_func_cross_vocab
   --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards_cross_vocab
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
   --teacher-tokenizer-path ${TEACHER_HF_PATH}
   --opd-teacher-timeout 300
   --opd-teacher-retries 2
)

EVAL_ARGS=(
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 8
   # --eval-max-response-len 4096
   # --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-8B-qwen35-35B-A3B-cross-vocab-opd
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --colocate
   --num-gpus-per-node ${SLIME_NUM_GPUS}
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CUDA_VISIBLE_DEVICES=${SLIME_CUDA_VISIBLE_DEVICES} ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${SLIME_NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "'${MEGATRON_ROOT}'/",
        "CUDA_VISIBLE_DEVICES": "'${SLIME_CUDA_VISIBLE_DEVICES}'",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${SLIME_NUM_GPUS} \
   --rollout-num-gpus ${SLIME_NUM_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

pkill -9 sglang
sleep 3
