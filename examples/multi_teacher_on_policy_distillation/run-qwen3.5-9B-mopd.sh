#!/bin/bash
# Multi-teacher on-policy distillation (MOPD) for Qwen3.5-9B.
#
# Every rollout sample is scored by the teacher of its `domain` label, so the
# teacher count follows the data instead of the command line.
#
#   8 GPUs: 4 actor (TP=2, CP=1) + 4 rollout (two TP=2 SGLang engines)
#
# Required paths (override as environment variables):
#   MODEL_DIR        HF student checkpoint
#   PROMPT_DATA      prompt file with a `domain` label per row
#   MOPD_TEACHER_*   per-domain teacher checkpoints, see config/
#
# The student is loaded from HF weights and trained with the OPD KL penalty, so
# no torch_dist student checkpoint or reference model is needed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)}"

export PYTHONUNBUFFERED=1
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the HF student checkpoint}"
PROMPT_DATA="${PROMPT_DATA:?set PROMPT_DATA to the domain-labelled prompt file}"
SAVE_PATH="${SAVE_PATH:-/tmp/slime_mopd_qwen3.5-9B}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SAVE_PATH}/tensorboard}"
MOPD_CONFIG="${MOPD_CONFIG:-${SCRIPT_DIR}/config/qwen3.5.three-teachers.yaml}"

MOPD_TEACHER_MATH="${MOPD_TEACHER_MATH:?set MOPD_TEACHER_MATH to the math teacher checkpoint}"
MOPD_TEACHER_CODE="${MOPD_TEACHER_CODE:-}"
MOPD_TEACHER_IFEVAL="${MOPD_TEACHER_IFEVAL:-}"

NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"

NUM_ROLLOUT="${NUM_ROLLOUT:-84}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
LR="${LR:-1e-6}"

source "${SLIME_ROOT}/scripts/models/qwen3.5-9B.sh"

WORK_DIR="${WORK_DIR:-${SAVE_PATH}/mopd-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$WORK_DIR"
# Resolve ${VAR} here so the job reads a plain YAML and needs no extra env.
RESOLVED_CONFIG="${WORK_DIR}/mopd_config.yaml"
export MOPD_TEACHER_MATH MOPD_TEACHER_CODE MOPD_TEACHER_IFEVAL
python3 - "${MOPD_CONFIG}" "${RESOLVED_CONFIG}" <<'RESOLVE_EOF'
import os
import sys

with open(sys.argv[1]) as src, open(sys.argv[2], "w") as dst:
    dst.write(os.path.expandvars(src.read()))
RESOLVE_EOF
MOPD_CONFIG="$RESOLVED_CONFIG"

# slime validates --opd-teacher-load even for the megatron teacher backend, which
# loads every teacher from the config; the first entry satisfies the check.
OPD_TEACHER_LOAD="$(python3 - "${SCRIPT_DIR}" "${MOPD_CONFIG}" <<'LOAD_EOF'
import sys

sys.path.insert(0, sys.argv[1])
from slime_mopd.config import load_mopd_config

args = type("Args", (), {"mopd": None, "mopd_config": sys.argv[2]})
print(next(t.load for t in load_mopd_config(args()).teachers if t.load))
LOAD_EOF
)"

CKPT_ARGS=(
   --hf-checkpoint "$MODEL_DIR"
   --load "$MODEL_DIR"
   --save "$SAVE_PATH"
   --save-interval "${SAVE_INTERVAL:-25}"
)

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA"
   --input-key prompt
   --apply-chat-template
   --data-source-path slime_mopd.data_source.MopdRolloutDataSource
   --custom-config-path "$MOPD_CONFIG"
   --custom-rm-path slime_mopd.rm.mopd_zero_reward
   --custom-convert-samples-to-train-data-path slime_mopd.train_data.mopd_convert_samples_to_train_data
   --rollout-shuffle
   --start-rollout-id 0
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
   --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
   --num-steps-per-rollout 1
   --rollout-max-prompt-len "$MAX_PROMPT_LEN"
   --rollout-max-response-len "$MAX_RESPONSE_LEN"
   --global-batch-size "$GLOBAL_BATCH_SIZE"
   --rollout-temperature 1
   --rollout-top-p 1
   --balance-data
   --flush-cache-interval 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type megatron
   --opd-kl-coef "${OPD_KL_COEF:-1.0}"
   --opd-teacher-load "$OPD_TEACHER_LOAD"
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.0
   --eps-clip-high 0.0
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE:-1}"
   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-4096}"
   --calculate-per-token-loss
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "$LR"
   --lr-decay-style constant
   --lr-warmup-iters 0
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
)

# SGLang assumes CUDA; a build with its own device library needs the flag.
SGLANG_DEVICE_ARGS=()
SGLANG_DETECTED_DEVICE="$(python3 - <<'DEVICE_EOF' 2>/dev/null || true
import torch

for name in ("musa", "xpu"):
    backend = getattr(torch, name, None)
    if backend is not None and backend.is_available():
        print(name)
        break
DEVICE_EOF
)"
if [[ -n "$SGLANG_DETECTED_DEVICE" ]]; then
   SGLANG_DEVICE_ARGS+=(--sglang-device "$SGLANG_DETECTED_DEVICE")
fi

SGLANG_ARGS=(
   "${SGLANG_DEVICE_ARGS[@]}"
   --sglang-mem-fraction-static 0.7
   --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
   --sglang-chunked-prefill-size 4096
   --sglang-page-size 1
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # Megatron backend selector; "auto" lets the installed stack choose.
   --attention-backend "${ATTENTION_BACKEND:-auto}"
)

LOGGING_ARGS=(
   --use-tensorboard
   --tensorboard-dir "$TENSORBOARD_DIR"
   --seed "${SEED:-0}"
)

# Only the import path and Megatron's TP/CP requirement; other variables reach
# the workers from the raylet environment, so platform settings stay in the
# caller's shell.
RUNTIME_ENV_JSON="$(PYTHONPATH_FOR_JOB="${PYTHONPATH_FOR_JOB:-${SLIME_ROOT}:${SCRIPT_DIR}}" \
   TENSORBOARD_DIR_FOR_JOB="${TENSORBOARD_DIR}" python3 - <<'ENV_EOF'
import json
import os

env = {
    "PYTHONPATH": os.environ["PYTHONPATH_FOR_JOB"],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TENSORBOARD_DIR": os.environ["TENSORBOARD_DIR_FOR_JOB"],
}
print(json.dumps({"env_vars": env}))
ENV_EOF
)"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
# Set RAY_GCS_PORT when another service already owns the default 6379.
RAY_GCS_ARGS=()
if [[ -n "${RAY_GCS_PORT:-}" ]]; then
   RAY_GCS_ARGS+=(--port "$RAY_GCS_PORT")
fi

if ! ray status >/dev/null 2>&1; then
   ray start --head \
      --node-ip-address "$MASTER_ADDR" \
      "${RAY_GCS_ARGS[@]}" \
      --num-gpus "$NUM_GPUS" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port "$RAY_DASHBOARD_PORT"
fi

export RAY_ADDRESS="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
JOB_ID="mopd_qwen3.5-9B_${RUN_ID}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}}"
mkdir -p "$LOG_DIR"
echo "[mopd] actor=${ACTOR_GPUS}GPU rollout=${ROLLOUT_GPUS}GPU steps=${NUM_ROLLOUT} log=${LOG_DIR}/run.log"

ray job submit \
   --address="$RAY_ADDRESS" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   --submission-id "$JOB_ID" \
   --no-wait \
   -- python3 -m slime_mopd.train_mopd \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$ACTOR_GPUS" \
   --rollout-num-gpus "$ROLLOUT_GPUS" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${LOGGING_ARGS[@]}"

ray job logs "$JOB_ID" --address="$RAY_ADDRESS" -f 2>&1 \
   | sed -u -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g' | tee -a "${LOG_DIR}/run.log"
