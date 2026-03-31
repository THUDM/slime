#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
AVALANCHE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

NUM_NODES=${NUM_NODES:-3}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-8}

MODEL_DIR=${MODEL_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints/iter_0001432_hf}
TORCH_DIST_DIR=${TORCH_DIST_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints}
WORK_ROOT=${WORK_ROOT:-${AVALANCHE_ROOT}/experiments/multidomain_v0_3node}

STEM_NEMOTRON_KNOWLEDGE_MCQA=${STEM_NEMOTRON_KNOWLEDGE_MCQA:-${AVALANCHE_ROOT}/data/open_data/stem/nemotron_knowledge_mcqa/data/train-00000-of-00004.parquet}
STEM_MMLU_PRO=${STEM_MMLU_PRO:-${AVALANCHE_ROOT}/data/open_data/stem/mmlu_pro/data/test-00000-of-00001.parquet}
STEM_AI2_ARC=${STEM_AI2_ARC:-${AVALANCHE_ROOT}/data/open_data/stem/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet}
STEM_SCIENCEQA=${STEM_SCIENCEQA:-${AVALANCHE_ROOT}/data/open_data/stem/scienceqa/data/train-00000-of-00001-1028f23e353fbe3e.parquet}
STEM_MEDMCQA=${STEM_MEDMCQA:-${AVALANCHE_ROOT}/data/open_data/stem/medmcqa/data/train-00000-of-00001.parquet}
STEM_OPENBOOKQA=${STEM_OPENBOOKQA:-${AVALANCHE_ROOT}/data/open_data/stem/openbookqa/main/train-00000-of-00001.parquet}
STEM_SCIQ=${STEM_SCIQ:-${AVALANCHE_ROOT}/data/open_data/stem/sciq/data/train-00000-of-00001.parquet}
TOOL_APIGEN=${TOOL_APIGEN:-${AVALANCHE_ROOT}/data/open_data/tool_call/apigen_mt_5k/apigen-mt_5k.json}
TOOL_XLAM=${TOOL_XLAM:-${AVALANCHE_ROOT}/data/open_data/tool_call/xlam_function_calling_60k/xlam-function-calling-60k.parquet}
STRUCTURED_NEMOTRON=${STRUCTURED_NEMOTRON:-${AVALANCHE_ROOT}/data/open_data/structured_output/nemotron_structured_outputs/structured_outputs_251027_nano_v3_sdg_json_train.jsonl}
STRUCTURED_IFEVAL=${STRUCTURED_IFEVAL:-${AVALANCHE_ROOT}/data/open_data/structured_output/ifeval/ifeval_input_data.jsonl}

SLIME_DIR=${SLIME_DIR:-/root/slime}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
VAL_SAMPLES=${VAL_SAMPLES:-1024}
DATA_CACHE_DIR="${WORK_ROOT}/data_cache"
LOG_DIR="${WORK_ROOT}/logs"
SAVE_DIR="${WORK_ROOT}/checkpoints"
NORMALIZED_TRAIN="${DATA_CACHE_DIR}/mixed_domain_train.normalized.jsonl"
NORMALIZED_VAL="${DATA_CACHE_DIR}/mixed_domain_val.normalized.jsonl"

TOOLCALL_ROLLOUT_BATCH_SIZE=${TOOLCALL_ROLLOUT_BATCH_SIZE:-32}
TOOLCALL_SAMPLES_PER_PROMPT=${TOOLCALL_SAMPLES_PER_PROMPT:-16}
TOOLCALL_GLOBAL_BATCH_SIZE=${TOOLCALL_GLOBAL_BATCH_SIZE:-512}
TOOLCALL_STEPS_PER_ROLLOUT=${TOOLCALL_STEPS_PER_ROLLOUT:-1}
TOOLCALL_MAX_CONTEXT_LEN=${TOOLCALL_MAX_CONTEXT_LEN:-32768}
TOOLCALL_MAX_RESPONSE_LEN=${TOOLCALL_MAX_RESPONSE_LEN:-16384}
TOOLCALL_LR=${TOOLCALL_LR:-1e-6}
TOOLCALL_ADAM_BETA2=${TOOLCALL_ADAM_BETA2:-0.98}
TOOLCALL_COLOCATE=${TOOLCALL_COLOCATE:-0}
TOOLCALL_RESUME_TRAINING=${TOOLCALL_RESUME_TRAINING:-0}
TOOL_CALL_LOAD_DIR=${TOOL_CALL_LOAD_DIR:-/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/experiments/ifrl_qwen3_30b_a3b/checkpoints}
TOOL_CALL_WANDB_PROJECT=${TOOL_CALL_WANDB_PROJECT:-slime-multidomain-v0}
TOOL_CALL_WANDB_GROUP=${TOOL_CALL_WANDB_GROUP:-qwen3-30b-a3b-multidomain-v0-3node}

mkdir -p "${DATA_CACHE_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

cleanup_local_processes() {
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
}

get_local_ip() {
  hostname -I 2>/dev/null | awk '{print $1}' || hostname
}

wait_for_full_ray_cluster() {
  local expected_gpus=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
  local expected_nodes=${NUM_NODES}
  local attempt

  for attempt in $(seq 1 30); do
    if python3 - <<PY
import json
import sys
import urllib.request

expected_gpus = ${expected_gpus}
expected_nodes = ${expected_nodes}

try:
    with urllib.request.urlopen("http://127.0.0.1:${DASHBOARD_PORT}/api/cluster_status", timeout=5) as response:
        payload = json.load(response)
except Exception:
    sys.exit(1)

report = payload.get("data", {}).get("clusterStatus", {}).get("autoscalerReport", {})
active_nodes = report.get("activeNodes") or {}
usage = payload.get("data", {}).get("clusterStatus", {}).get("loadMetricsReport", {}).get("usage", {})
gpu_total = (usage.get("GPU") or [0.0, 0.0])[1]

if len(active_nodes) >= expected_nodes and gpu_total >= expected_gpus:
    sys.exit(0)

sys.exit(1)
PY
    then
      echo "Ray cluster is ready with ${expected_nodes} nodes and ${expected_gpus} GPUs."
      return 0
    fi
    echo "Waiting for Ray workers to join (${attempt}/30)..."
    sleep 10
  done

  echo "Ray workers did not join the cluster in time." >&2
  ray status --address="${MASTER_ADDR}:${MASTER_PORT}" || true
  return 1
}

start_ray_worker_with_retry() {
  local attempt
  local rc
  local node_name="${WORKER_ID:-${HOSTNAME:-worker-${NODE_RANK}}}"

  for attempt in $(seq 1 30); do
    set +e
    ray start \
      --address="${MASTER_ADDR}:${MASTER_PORT}" \
      --num-gpus "${NUM_GPUS_PER_NODE}" \
      --node-ip-address "${NODE_IP}" \
      --node-name "${node_name}" \
      --dashboard-port="${DASHBOARD_PORT}" \
      --disable-usage-stats
    rc=$?
    set -e

    if [ "${rc}" -eq 0 ]; then
      echo "Ray worker joined on attempt ${attempt}."
      return 0
    fi

    echo "Ray worker join failed on attempt ${attempt}, retrying..."
    ray stop --force 2>/dev/null || true
    sleep 5
  done

  echo "Ray worker failed to join cluster after retries." >&2
  return 1
}

prepare_data() {
  python3 "${SCRIPT_DIR}/prepare_mixed_domain_data.py" \
    --source "${STEM_NEMOTRON_KNOWLEDGE_MCQA}" --dataset-format nemotron_knowledge_mcqa --source-ratio 25 \
    --source "${STEM_MMLU_PRO}" --dataset-format mmlu_pro --source-ratio 15 \
    --source "${STEM_AI2_ARC}" --dataset-format ai2_arc --source-ratio 5 \
    --source "${STEM_SCIENCEQA}" --dataset-format scienceqa --source-ratio 5 \
    --source "${STEM_MEDMCQA}" --dataset-format medmcqa --source-ratio 2 \
    --source "${STEM_OPENBOOKQA}" --dataset-format openbookqa --source-ratio 1.5 \
    --source "${STEM_SCIQ}" --dataset-format sciq --source-ratio 1.5 \
    --source "${TOOL_APIGEN}" --dataset-format apigen_mt_5k --source-ratio 20 \
    --source "${TOOL_XLAM}" --dataset-format xlam_function_calling_60k --source-ratio 10 \
    --source "${STRUCTURED_NEMOTRON}" --dataset-format nemotron_structured_outputs --source-ratio 10 \
    --source "${STRUCTURED_IFEVAL}" --dataset-format ifeval --source-ratio 5 \
    --dest "${NORMALIZED_TRAIN}" \
    --skip-samples "${VAL_SAMPLES}"

  python3 "${SCRIPT_DIR}/prepare_mixed_domain_data.py" \
    --source "${STEM_NEMOTRON_KNOWLEDGE_MCQA}" --dataset-format nemotron_knowledge_mcqa --source-ratio 25 \
    --source "${STEM_MMLU_PRO}" --dataset-format mmlu_pro --source-ratio 15 \
    --source "${STEM_AI2_ARC}" --dataset-format ai2_arc --source-ratio 5 \
    --source "${STEM_SCIENCEQA}" --dataset-format scienceqa --source-ratio 5 \
    --source "${STEM_MEDMCQA}" --dataset-format medmcqa --source-ratio 2 \
    --source "${STEM_OPENBOOKQA}" --dataset-format openbookqa --source-ratio 1.5 \
    --source "${STEM_SCIQ}" --dataset-format sciq --source-ratio 1.5 \
    --source "${TOOL_APIGEN}" --dataset-format apigen_mt_5k --source-ratio 20 \
    --source "${TOOL_XLAM}" --dataset-format xlam_function_calling_60k --source-ratio 10 \
    --source "${STRUCTURED_NEMOTRON}" --dataset-format nemotron_structured_outputs --source-ratio 10 \
    --source "${STRUCTURED_IFEVAL}" --dataset-format ifeval --source-ratio 5 \
    --dest "${NORMALIZED_VAL}" \
    --max-samples "${VAL_SAMPLES}"
}

ensure_torch_dist_checkpoint() {
  if [ -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "Found torch_dist checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi
  if [ -f "${TORCH_DIST_DIR}/common.pt" ] && [ -f "${TORCH_DIST_DIR}/metadata.json" ]; then
    echo "Found torch_dist iteration checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi

  cd "${SLIME_DIR}"
  source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"
  PYTHONPATH="${MEGATRON_PATH}:${SLIME_DIR}" torchrun \
    --nproc-per-node "${NUM_GPUS_PER_NODE}" \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --save "${TORCH_DIST_DIR}"
}

submit_ray_job() {
  cd "${SLIME_DIR}"
  source "${SLIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
  if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
  else
    HAS_NVLINK=0
  fi

  LOAD_ARGS=()
  if [[ "${TOOLCALL_RESUME_TRAINING}" == "1" ]] && [[ -n "${TOOL_CALL_LOAD_DIR}" ]]; then
    LOAD_ARGS+=(--load "${TOOL_CALL_LOAD_DIR}")
  fi

  ROLLOUT_ARGS=(
    --prompt-data "${NORMALIZED_TRAIN}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --tool-key tools
    --apply-chat-template
    --rollout-shuffle
    --num-epoch 1
    --rollout-batch-size "${TOOLCALL_ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${TOOLCALL_SAMPLES_PER_PROMPT}"
    --rollout-max-context-len "${TOOLCALL_MAX_CONTEXT_LEN}"
    --rollout-max-response-len "${TOOLCALL_MAX_RESPONSE_LEN}"
    --rollout-temperature 0.7
    --rollout-top-p 1.0
    --global-batch-size "${TOOLCALL_GLOBAL_BATCH_SIZE}"
    --num-steps-per-rollout "${TOOLCALL_STEPS_PER_ROLLOUT}"
    --balance-data
  )

  EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data mixed_domain_val "${NORMALIZED_VAL}"
    --eval-input-key prompt
    --eval-label-key label
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${TOOLCALL_MAX_RESPONSE_LEN}"
  )

  PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
  )

  GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.2
  )

  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${TOOLCALL_LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 "${TOOLCALL_ADAM_BETA2}"
  )

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-mem-fraction-static 0.7
    --sglang-ep-size 8
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
  )

  MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
  )

  CUSTOM_ARGS=(
    --custom-rm-path reward_mixed_domain.reward_func
    --custom-rollout-log-function-path log_mixed_domain.log_rollout_data
    --custom-eval-rollout-log-function-path log_mixed_domain.log_eval_rollout_data
  )

  WANDB_ARGS=()
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS+=(
      --use-wandb
      --wandb-host "${WANDB_BASE_URL:-https://wandb.ai}"
      --wandb-project "${TOOL_CALL_WANDB_PROJECT}"
      --wandb-group "${TOOL_CALL_WANDB_GROUP}"
      --wandb-key "${WANDB_API_KEY}"
      --disable-wandb-random-suffix
    )
  fi

  RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${SCRIPT_DIR}:${MEGATRON_PATH}:${SLIME_DIR}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"MASTER_ADDR\":\"${MASTER_ADDR}\",\"WANDB_API_KEY\":\"${WANDB_API_KEY:-}\",\"WANDB_BASE_URL\":\"${WANDB_BASE_URL:-}\"}}"

  TRAINING_RESOURCE_ARGS=(
    --actor-num-nodes "${ACTOR_NUM_NODES}"
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
  )
  if [[ "${TOOLCALL_COLOCATE}" == "1" ]]; then
    TRAINING_RESOURCE_ARGS+=(--colocate)
  else
    TRAINING_RESOURCE_ARGS+=(--rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}")
  fi

  ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train_async.py" \
    "${TRAINING_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --ref-load "${TORCH_DIST_DIR}" \
    "${LOAD_ARGS[@]}" \
    --save "${SAVE_DIR}" \
    --save-interval 20 \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
}

NODE_RANK=${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(get_local_ip)}}"
MASTER_PORT=${MASTER_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
NODE_IP="$(get_local_ip)"

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

cleanup_local_processes

if [[ "${NODE_RANK}" -eq 0 ]]; then
  prepare_data
  ensure_torch_dist_checkpoint
  ray start --head \
    --port="${MASTER_PORT}" \
    --node-ip-address "${MASTER_ADDR}" \
    --node-name "${WORKER_ID:-${HOSTNAME:-head-0}}" \
    --num-gpus "${NUM_GPUS_PER_NODE}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}"
  wait_for_full_ray_cluster
  submit_ray_job
  ray stop --force || true
else
  sleep 5
  start_ray_worker_with_retry
  while ray status --address="${MASTER_ADDR}:${MASTER_PORT}" >/dev/null 2>&1; do
    sleep 60
  done
fi
