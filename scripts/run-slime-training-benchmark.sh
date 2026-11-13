#!/usr/bin/env bash
# Run a deterministic, train-only slime benchmark for comparison with Weaver.
#
# Required environment variables:
#   HF_CHECKPOINT  HuggingFace checkpoint used to initialise the actor
#   TOKEN_DATA     JSONL file with one unique token sequence per line
#   MODEL_CONFIG   shell file defining MODEL_ARGS=( ... )
#
# Optional optimizer variables make the benchmark useful for comparing the
# optimizer/update path as well as forward/backward:
#   OPTIMIZER (default: adam), LR, LR_DECAY_STYLE, WEIGHT_DECAY,
#   ADAM_BETA1, ADAM_BETA2
#
# Perf times are per rollout. Each rollout performs
# ROLLOUT_BATCH_SIZE / GLOBAL_BATCH_SIZE optimizer updates (one by default).
# SEQ_LEN includes the first prompt-boundary token; each sample has
# SEQ_LEN - 1 supervised next-token targets.
#
# The model config can be one of slime's existing files, for example
# scripts/models/qwen2.5-0.5B.sh. No process is killed by this script; set
# START_RAY=0 when a Ray cluster is already running.
#
# Example (gradient accumulation = 2 on a 2-GPU, TP=2 run):
#   HF_CHECKPOINT=/models/Qwen2.5-0.5B \
#   TOKEN_DATA=/datasets/wildchat_tokens.jsonl \
#   MODEL_CONFIG=scripts/models/qwen2.5-0.5B.sh \
#   NUM_GPUS=2 TP_SIZE=2 GLOBAL_BATCH_SIZE=4 MICRO_BATCH_SIZE=2 \
#   scripts/run-slime-training-benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

absolute_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        printf '%s' "${path}"
    else
        printf '%s/%s' "$(pwd -P)" "${path}"
    fi
}

: "${HF_CHECKPOINT:?set HF_CHECKPOINT to the HuggingFace checkpoint}"
: "${TOKEN_DATA:?set TOKEN_DATA to a token JSONL file}"
: "${MODEL_CONFIG:?set MODEL_CONFIG to a file defining MODEL_ARGS=(...)}"

REF_LOAD="${REF_LOAD:-${HF_CHECKPOINT}}"
NUM_GPUS="${NUM_GPUS:-2}"
TP_SIZE="${TP_SIZE:-${NUM_GPUS}}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
WARMUP_ROLLOUTS="${WARMUP_ROLLOUTS:-1}"
MEASURED_ROLLOUTS="${MEASURED_ROLLOUTS:-3}"
SEQ_LEN="${SEQ_LEN:-4097}"
DATA_PAD_SIZE_MULTIPLIER="${DATA_PAD_SIZE_MULTIPLIER:-1}"
OPTIMIZER="${OPTIMIZER:-adam}"
LR="${LR:-1e-6}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
SEED="${SEED:-1234}"
RESULT_DIR="${RESULT_DIR:-${REPO_DIR}/benchmark-results/slime-$(date +%Y%m%d-%H%M%S)}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}"
RAY_NODE_IP="${RAY_NODE_IP:-127.0.0.1}"
RAY_GCS_ADDRESS="${RAY_GCS_ADDRESS:-${RAY_NODE_IP}:6379}"
START_RAY="${START_RAY:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Ray uploads REPO_DIR as its working directory. Keep local artifacts and
# externally mounted checkpoints addressable from the worker process.
RESULT_DIR="$(absolute_path "${RESULT_DIR}")"
TOKEN_DATA="$(absolute_path "${TOKEN_DATA}")"
if [[ "${HF_CHECKPOINT}" != /* && -e "${HF_CHECKPOINT}" ]]; then
    HF_CHECKPOINT="$(absolute_path "${HF_CHECKPOINT}")"
fi
if [[ "${REF_LOAD}" != /* && -e "${REF_LOAD}" ]]; then
    REF_LOAD="$(absolute_path "${REF_LOAD}")"
fi

for value_name in NUM_GPUS TP_SIZE ROLLOUT_BATCH_SIZE GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE WARMUP_ROLLOUTS MEASURED_ROLLOUTS SEQ_LEN DATA_PAD_SIZE_MULTIPLIER; do
    value="${!value_name}"
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "${value_name} must be a non-negative integer" >&2; exit 2; }
done
[[ "${SEED}" =~ ^-?[0-9]+$ ]] || { echo "SEED must be an integer" >&2; exit 2; }
[[ "${START_RAY}" =~ ^[01]$ ]] || { echo "START_RAY must be 0 or 1" >&2; exit 2; }
[[ "${DRY_RUN}" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
(( NUM_GPUS > 0 && TP_SIZE > 0 && NUM_GPUS % TP_SIZE == 0 )) || {
    echo "NUM_GPUS must be a positive multiple of TP_SIZE" >&2
    exit 2
}
(( ROLLOUT_BATCH_SIZE > 0 && GLOBAL_BATCH_SIZE > 0 && MICRO_BATCH_SIZE > 0 )) || {
    echo "batch sizes must be positive" >&2
    exit 2
}
(( GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE == 0 )) || {
    echo "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE for static micro-batches" >&2
    exit 2
}
(( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * (NUM_GPUS / TP_SIZE)) == 0 )) || {
    echo "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE * data_parallel_size" >&2
    exit 2
}
(( ROLLOUT_BATCH_SIZE >= GLOBAL_BATCH_SIZE && ROLLOUT_BATCH_SIZE % GLOBAL_BATCH_SIZE == 0 )) || {
    echo "ROLLOUT_BATCH_SIZE must be a positive multiple of GLOBAL_BATCH_SIZE" >&2
    exit 2
}
(( SEQ_LEN >= 2 && DATA_PAD_SIZE_MULTIPLIER > 0 )) || {
    echo "SEQ_LEN must be >= 2 and DATA_PAD_SIZE_MULTIPLIER must be positive" >&2
    exit 2
}
(( MEASURED_ROLLOUTS > 0 )) || { echo "MEASURED_ROLLOUTS must be positive" >&2; exit 2; }

TOTAL_ROLLOUTS=$((WARMUP_ROLLOUTS + MEASURED_ROLLOUTS))
ROLLOUT_DIR="${RESULT_DIR}/rollout_data"
if [[ -e "${RESULT_DIR}" ]]; then
    echo "RESULT_DIR already exists; choose a fresh directory: ${RESULT_DIR}" >&2
    exit 2
fi
mkdir -p "${RESULT_DIR}"

if [[ "${MODEL_CONFIG}" != /* && ! -f "${MODEL_CONFIG}" && -f "${REPO_DIR}/${MODEL_CONFIG}" ]]; then
    MODEL_CONFIG="${REPO_DIR}/${MODEL_CONFIG}"
fi
[[ -f "${MODEL_CONFIG}" ]] || { echo "MODEL_CONFIG does not exist: ${MODEL_CONFIG}" >&2; exit 2; }
MODEL_CONFIG="$(absolute_path "${MODEL_CONFIG}")"
[[ -f "${TOKEN_DATA}" ]] || { echo "TOKEN_DATA does not exist: ${TOKEN_DATA}" >&2; exit 2; }

# MODEL_ARGS is intentionally supplied by a model-specific file so this
# benchmark does not duplicate architecture flags or silently pick a model.
# shellcheck disable=SC1090
source "${MODEL_CONFIG}"
declare -p MODEL_ARGS >/dev/null 2>&1 || {
    echo "${MODEL_CONFIG} must define a MODEL_ARGS bash array" >&2
    exit 2
}

cat >"${RESULT_DIR}/config.json" <<EOF
{
  "hf_checkpoint": "${HF_CHECKPOINT}",
  "ref_load": "${REF_LOAD}",
  "model_config": "${MODEL_CONFIG}",
  "token_data": "${TOKEN_DATA}",
  "num_gpus": ${NUM_GPUS},
  "tensor_model_parallel_size": ${TP_SIZE},
  "rollout_batch_size": ${ROLLOUT_BATCH_SIZE},
  "global_batch_size": ${GLOBAL_BATCH_SIZE},
  "micro_batch_size": ${MICRO_BATCH_SIZE},
  "gradient_accumulation_per_rank": $((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * (NUM_GPUS / TP_SIZE)))),
  "optimizer_steps_per_rollout": $((ROLLOUT_BATCH_SIZE / GLOBAL_BATCH_SIZE)),
  "optimizer": "${OPTIMIZER}",
  "lr": "${LR}",
  "lr_decay_style": "${LR_DECAY_STYLE}",
  "weight_decay": "${WEIGHT_DECAY}",
  "adam_beta1": "${ADAM_BETA1}",
  "adam_beta2": "${ADAM_BETA2}",
  "seed": ${SEED},
  "sequence_length": ${SEQ_LEN},
  "data_pad_size_multiplier": ${DATA_PAD_SIZE_MULTIPLIER},
  "warmup_rollouts": ${WARMUP_ROLLOUTS},
  "measured_rollouts": ${MEASURED_ROLLOUTS},
  "dry_run": ${DRY_RUN}
}
EOF

echo "Generating ${TOTAL_ROLLOUTS} rollout files in ${ROLLOUT_DIR}" >&2
PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" python3 "${REPO_DIR}/tools/generate_slime_debug_rollout.py" \
    --input "${TOKEN_DATA}" \
    --output-dir "${ROLLOUT_DIR}" \
    --num-rollout "${TOTAL_ROLLOUTS}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --expected-seq-len "${SEQ_LEN}" \
    >"${RESULT_DIR}/rollout-manifest.json"

TRAIN_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node "${NUM_GPUS}"
    --num-gpus-per-node "${NUM_GPUS}"
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${REF_LOAD}"
    --megatron-to-hf-mode bridge
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --disable-rollout-global-dataset
    --num-rollout "${TOTAL_ROLLOUTS}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --seq-length "${SEQ_LEN}"
    --optimizer "${OPTIMIZER}"
    --lr "${LR}"
    --lr-decay-style "${LR_DECAY_STYLE}"
    --weight-decay "${WEIGHT_DECAY}"
    --adam-beta1 "${ADAM_BETA1}"
    --adam-beta2 "${ADAM_BETA2}"
    --seed "${SEED}"
    --debug-train-only
    --load-debug-rollout-data "${ROLLOUT_DIR}/rollout_{rollout_id}.pt"
    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
)

printf '%q ' python3 train.py "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" >"${RESULT_DIR}/command.txt"
printf '\n' >>"${RESULT_DIR}/command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: generated rollout data and command only; skipping Ray submission" >&2
    : >"${RESULT_DIR}/train.log"
    printf '[]\n' >"${RESULT_DIR}/perf.json"
    printf '{}\n' >"${RESULT_DIR}/perf-summary.json"
    exit 0
fi

command -v ray >/dev/null 2>&1 || { echo "ray CLI is required to run the benchmark" >&2; exit 2; }

if [[ "${START_RAY}" == "1" ]]; then
    # `ray status` speaks to the GCS bootstrap address, while `ray job submit`
    # uses the dashboard URL. Probe the GCS address so an existing local head
    # is reused instead of starting a second head on occupied ports.
    if ray status --address="${RAY_GCS_ADDRESS}" >/dev/null 2>&1; then
        echo "Reusing existing Ray cluster at ${RAY_GCS_ADDRESS} (${RAY_ADDRESS})" >&2
    else
        echo "Starting Ray head at ${RAY_ADDRESS}" >&2
        ray start --head \
            --node-ip-address "${RAY_NODE_IP}" \
            --num-gpus "${NUM_GPUS}" \
            --disable-usage-stats \
            --dashboard-host=0.0.0.0 \
            --dashboard-port "${RAY_DASHBOARD_PORT}" \
            >"${RESULT_DIR}/ray-start.log" 2>&1
    fi
fi

RUNTIME_ENV_JSON="$(PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" python3 - <<'PY'
import json
import os

print(json.dumps({"env_vars": {"PYTHONUNBUFFERED": "1", "PYTHONPATH": os.environ["PYTHONPATH"]}}))
PY
)"

echo "Submitting train-only job; logs: ${RESULT_DIR}/train.log" >&2
set -o pipefail
ray job submit \
    --address="${RAY_ADDRESS}" \
    --working-dir="${REPO_DIR}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py "${MODEL_ARGS[@]}" "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RESULT_DIR}/train.log"

RESULT_DIR="${RESULT_DIR}" WARMUP_ROLLOUTS="${WARMUP_ROLLOUTS}" MEASURED_ROLLOUTS="${MEASURED_ROLLOUTS}" python3 - <<'PY'
"""Print the measured perf dictionaries emitted by slime's train actor."""

import ast
import json
import os
import re
from pathlib import Path

log_path = Path(os.environ["RESULT_DIR"]) / "train.log"
warmup = int(os.environ["WARMUP_ROLLOUTS"])
pattern = re.compile(r"perf\s+(\d+):\s+(\{.*\})")
rows = []
for line in log_path.read_text(errors="replace").splitlines():
    match = pattern.search(line)
    if not match:
        continue
    try:
        metrics = ast.literal_eval(match.group(2))
    except (SyntaxError, ValueError):
        continue
    rollout_id = int(match.group(1))
    if rollout_id >= warmup:
        rows.append((rollout_id, metrics))

serializable_rows = [
    {"rollout_id": rollout_id, "metrics": metrics}
    for rollout_id, metrics in rows
]
(Path(os.environ["RESULT_DIR"]) / "perf.json").write_text(
    json.dumps(serializable_rows, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)

numeric_keys = {
    key
    for _rollout_id, metrics in rows
    for key, value in metrics.items()
    if isinstance(value, (int, float)) and not isinstance(value, bool)
}
expected_ids = set(range(warmup, warmup + int(os.environ["MEASURED_ROLLOUTS"])))
actual_ids = {rollout_id for rollout_id, _metrics in rows}
if len(rows) != len(expected_ids) or actual_ids != expected_ids:
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    raise SystemExit(f"missing or unexpected perf records: missing={missing}, unexpected={unexpected}")
required_keys = {"perf/actor_train_time", "perf/train_time", "perf/step_time"}
for rollout_id, metrics in rows:
    missing_keys = sorted(required_keys - metrics.keys())
    if missing_keys:
        raise SystemExit(f"rollout {rollout_id} is missing required perf metrics: {missing_keys}")
summary = {
    key: sum(metrics[key] for _rollout_id, metrics in rows if key in metrics)
    / sum(key in metrics for _rollout_id, metrics in rows)
    for key in sorted(numeric_keys)
}
(Path(os.environ["RESULT_DIR"]) / "perf-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

keys = (
    "perf/data_preprocess_time",
    "perf/actor_train_time",
    "perf/train_time",
    "perf/step_time",
    "perf/actor_train_tok_per_s",
)
print("Measured slime perf (rollout_id >= warmup):")
for rollout_id, metrics in rows:
    print(f"  rollout {rollout_id}:")
    for key in keys:
        if key in metrics:
            print(f"    {key}: {metrics[key]}")
print("Average measured perf:")
for key in keys:
    if key in summary:
        print(f"  {key}: {summary[key]}")
PY
