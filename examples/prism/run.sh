#!/usr/bin/env bash
# Use an existing Ray cluster; resource ownership stays with slime.
set -euo pipefail

TASK_EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_REPO_ROOT=$(cd -- "${TASK_EXAMPLE_DIR}/../.." && pwd)
TASK_MODE=${1:?Expected grpo or mopd}
shift
case "${TASK_MODE}" in
  grpo|mopd) ;;
  *) echo "Expected grpo or mopd, got ${TASK_MODE}" >&2; exit 2 ;;
esac

: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the matching Prism inference artifact}"
: "${PROMPT_DATA:?Set PROMPT_DATA to your localized Prism JSONL dataset}"
: "${OMNI_CONFIG:?Set OMNI_CONFIG to the student pipeline YAML}"
TASK_TRAIN_PYTHON=${TRAIN_PYTHON:-python3}
TASK_OMNI_PYTHON=${OMNI_PYTHON:-${TASK_TRAIN_PYTHON}}
TASK_OUTPUT_DIR=${OUTPUT_DIR:-${PWD}/runs/prism-${TASK_MODE}}
TASK_ACTOR_NODES=${ACTOR_NODES:-1}
TASK_ACTOR_GPUS=${ACTOR_GPUS_PER_NODE:-1}
TASK_ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
TASK_COLOCATE=${COLOCATE:-0}
[[ ${TASK_COLOCATE} == 0 || ${TASK_COLOCATE} == 1 ]] || { echo "COLOCATE must be 0 or 1" >&2; exit 2; }
TASK_PROMPTS_PER_BATCH=${PROMPTS_PER_BATCH:-2}
if [[ ${TASK_MODE} == grpo ]]; then
  TASK_SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-4}
else
  TASK_SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-1}
fi
for TASK_INPUT_PATH in "${HF_CHECKPOINT}/config.json" "${PROMPT_DATA}" "${OMNI_CONFIG}"; do
  [[ -f ${TASK_INPUT_PATH} ]] || { echo "Missing input file: ${TASK_INPUT_PATH}" >&2; exit 2; }
done
for TASK_COUNT in "${TASK_ACTOR_NODES}" "${TASK_ACTOR_GPUS}" "${TASK_ROLLOUT_GPUS}" \
                  "${TASK_PROMPTS_PER_BATCH}" "${TASK_SAMPLES_PER_PROMPT}"; do
  [[ ${TASK_COUNT} =~ ^[1-9][0-9]*$ ]] || { echo "Resource and batch counts must be positive integers" >&2; exit 2; }
done
TASK_GPUS_PER_NODE=${GPUS_PER_NODE:-$((TASK_ACTOR_GPUS + TASK_ROLLOUT_GPUS))}
if [[ ${TASK_COLOCATE} == 1 ]]; then
  TASK_GPUS_PER_NODE=${GPUS_PER_NODE:-${TASK_ACTOR_GPUS}}
fi
TASK_GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((TASK_PROMPTS_PER_BATCH * TASK_SAMPLES_PER_PROMPT))}
for TASK_COUNT in "${TASK_GPUS_PER_NODE}" "${TASK_GLOBAL_BATCH_SIZE}"; do
  [[ ${TASK_COUNT} =~ ^[1-9][0-9]*$ ]] || { echo "Resource and batch counts must be positive integers" >&2; exit 2; }
done
TASK_DP_SIZE=$((TASK_ACTOR_NODES * TASK_ACTOR_GPUS))
(( TASK_GLOBAL_BATCH_SIZE % TASK_DP_SIZE == 0 )) || {
  echo "GLOBAL_BATCH_SIZE must be divisible by the learner DP size (${TASK_DP_SIZE})" >&2; exit 2;
}

export PYTHONPATH="${TASK_REPO_ROOT}${MEGATRON_ROOT:+:${MEGATRON_ROOT}}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
# Explicitly carry the model import path into remote Megatron workers.
TASK_TRAIN_ENV=$("${TASK_TRAIN_PYTHON}" - <<'PY'
import json
import os
print(json.dumps({key: os.environ[key] for key in (
    "PYTHONPATH", "PYTHONUNBUFFERED", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_CUMEM_ENABLE"
)}))
PY
)

# Produce ordinary Megatron flags before its parser and HF validation run.
# A file preserves argument boundaries and propagates configuration failures.
TASK_ARCH_ARGS_FILE=$(mktemp)
trap 'rm -f -- "${TASK_ARCH_ARGS_FILE}"' EXIT
"${TASK_TRAIN_PYTHON}" - "${HF_CHECKPOINT}" > "${TASK_ARCH_ARGS_FILE}" <<'PY'
import sys

from slime_plugins.models.prism import prism_megatron_cli_args

for argument in prism_megatron_cli_args(sys.argv[1]):
    sys.stdout.buffer.write(argument.encode() + b"\0")
PY
mapfile -d '' -t TASK_ARCH_ARGS < "${TASK_ARCH_ARGS_FILE}"
rm -f -- "${TASK_ARCH_ARGS_FILE}"
trap - EXIT

TASK_ARGS=(
  "${TASK_ARCH_ARGS[@]}"
  --actor-num-nodes "${TASK_ACTOR_NODES}"
  --actor-num-gpus-per-node "${TASK_ACTOR_GPUS}"
  --rollout-num-gpus "${TASK_ROLLOUT_GPUS}"
  --num-gpus-per-node "${TASK_GPUS_PER_NODE}"
  --rollout-num-gpus-per-engine 1
  --rollout-backend sglang-omni
  --omni-python "${TASK_OMNI_PYTHON}"
  --omni-config "${OMNI_CONFIG}"
  --omni-model-name student
  --omni-router-policy least_request
  --hf-checkpoint "${HF_CHECKPOINT}"
  --model-name moss_tts_prism
  --custom-model-provider-path slime_plugins.models.prism.model_provider
  --custom-megatron-log-prob-function-path slime_plugins.models.prism.extract_policy_log_probs
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout
  --custom-generate-function-path slime_plugins.omni_rollout.generate
  --custom-config-path "${TASK_EXAMPLE_DIR}/${TASK_MODE}.yaml"
  --train-env-vars "${TASK_TRAIN_ENV}"
  --prompt-data "${PROMPT_DATA}"
  --input-key script
  --metadata-key metadata
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT:-100}"
  --rollout-batch-size "${TASK_PROMPTS_PER_BATCH}"
  --n-samples-per-prompt "${TASK_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${MAX_FRAMES:-512}"
  --rollout-temperature 1
  --rollout-top-p 1
  --rollout-top-k -1
  --sglang-server-concurrency "${ROLLOUT_CONCURRENCY_PER_REPLICA:-2}"
  --global-batch-size "${TASK_GLOBAL_BATCH_SIZE}"
  --micro-batch-size 1
  --advantage-estimator grpo
  --loss-type policy_loss
  --kl-coef 0
  --entropy-coef 0
  --eps-clip 0.2
  --eps-clip-high 0.2
  --policy-stop-loss-weight "${STOP_LOSS_WEIGHT:-1}"
  --update-weight-mode full
  --update-weight-transport nccl
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --seq-length "${TRAIN_SEQ_LENGTH:-4096}"
  --bf16
  --transformer-impl transformer_engine
  --attention-dropout 0
  --hidden-dropout 0
  --accumulate-allreduce-grads-in-fp32
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --optimizer adam
  --lr "${LEARNING_RATE:-1e-6}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY:-0.01}"
  --adam-beta1 0.9
  --adam-beta2 0.98
  --save "${TASK_OUTPUT_DIR}"
  --save-interval "${SAVE_INTERVAL:-10}"
)

if [[ ${TASK_COLOCATE} == 1 ]]; then
  TASK_ARGS+=(--colocate)
fi

if [[ -n ${RL_LOAD:-} ]]; then
  [[ -d ${RL_LOAD} ]] || { echo "Missing RL checkpoint directory: ${RL_LOAD}" >&2; exit 2; }
  TASK_ARGS+=(--load "${RL_LOAD}")
else
  : "${PRISM_PRETRAINED_CHECKPOINT:?Set PRISM_PRETRAINED_CHECKPOINT for the first run, or RL_LOAD for resume}"
  [[ -d ${PRISM_PRETRAINED_CHECKPOINT} ]] || {
    echo "Missing native Prism checkpoint: ${PRISM_PRETRAINED_CHECKPOINT}" >&2; exit 2;
  }
  TASK_ARGS+=(
    --custom-megatron-model-initializer-path slime_plugins.models.prism.load_prism_pretrained
    --pretrained-checkpoint "${PRISM_PRETRAINED_CHECKPOINT}"
  )
  if [[ -n ${PRISM_PRETRAINED_STEP:-} ]]; then
    TASK_ARGS+=(--ckpt-step "${PRISM_PRETRAINED_STEP}")
  fi
fi

if [[ ${TASK_MODE} == grpo ]]; then
  : "${REWARD_CONFIG:?Set REWARD_CONFIG to frozen WER/SIM/JudgeCLAP service bindings}"
  [[ -f ${REWARD_CONFIG} ]] || { echo "Missing reward config: ${REWARD_CONFIG}" >&2; exit 2; }
  TASK_ARGS+=(
    --custom-rm-path slime_plugins.tts_reward.reward
    --omni-reward-config "${REWARD_CONFIG}"
    --omni-audio-dir "${AUDIO_DIR:-${TASK_OUTPUT_DIR}/audio}"
  )
else
  : "${TEACHER_CONFIG:?Set TEACHER_CONFIG to domain-to-frozen-Prism-teacher bindings}"
  [[ -f ${TEACHER_CONFIG} ]] || { echo "Missing teacher config: ${TEACHER_CONFIG}" >&2; exit 2; }
  TASK_ARGS+=(
    --use-opd
    --opd-type sglang
    --opd-kl-coef "${OPD_KL_COEF:-1}"
    --rollout-sample-hook-path slime_plugins.omni_rollout.score_teacher
    --omni-teacher-config "${TEACHER_CONFIG}"
    --custom-rm-path examples.prism.hooks.zero_reward
  )
fi

# Keep actual Omni scores for diagnostics; no --use-rollout-logprobs means
# Megatron recomputes and freezes old selected scores before optimization.
if [[ ${PRISM_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${TASK_TRAIN_PYTHON}" "${TASK_REPO_ROOT}/train.py" "${TASK_ARGS[@]}" "$@"
  printf '\n'
  exit 0
fi
exec "${TASK_TRAIN_PYTHON}" "${TASK_REPO_ROOT}/train.py" "${TASK_ARGS[@]}" "$@"
