#!/bin/bash
#SBATCH --job-name=slime-qwen3-opd10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/checkpoint_utils.sh"

mkdir -p "${SLURM_LOG_DIR}" "${OPD_SAVE_DIR}" "${OPD_HF_SNAPSHOT_DIR}" "${OPD_ROLLOUT_LOG_DIR}" "${TEACHER_LOG_DIR}" "${CHECKPOINT_REPORT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export RAY_TMPDIR="/tmp/${USER:-suryadv}/slime-ray-${SLURM_JOB_ID}"
  mkdir -p "${RAY_TMPDIR}"
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

for required_path in \
  "${OPD_JSONL}" \
  "${STUDENT_HF_DIR}" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" \
  "${TEACHER_HF_DIR}" \
  "${SFT_FINAL_HF_DIR}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

echo "Starting OPD on $(hostname) at $(date)"
echo "OPD data: ${OPD_JSONL}"
echo "Teacher: ${TEACHER_HF_DIR} on physical GPU ${OPD_TEACHER_GPU}"
echo "Ray GPUs: 0-$((OPD_RAY_GPUS - 1))"
echo "Save dir: ${OPD_SAVE_DIR}"
echo "HF eval snapshots: ${OPD_HF_SNAPSHOT_DIR}"
echo "Expected final OPD rollout id: ${OPD_FINAL_ROLLOUT_ID}"

checkpoint_start_pruner opd "${OPD_SAVE_DIR}" "${CHECKPOINT_PRUNE_INTERVAL_SECONDS}" "${CHECKPOINT_REPORT_DIR}"
cleanup_pruner() {
  checkpoint_stop_pruner "${CHECKPOINT_PRUNER_PID:-}"
}
trap cleanup_pruner EXIT

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"

ray stop --force >/dev/null 2>&1 || true

TEACHER_IP="127.0.0.1"
TEACHER_PORT="${OPD_TEACHER_PORT}"
TEACHER_LOG="${TEACHER_LOG_DIR}/sglang_teacher_${SLURM_JOB_ID:-manual}.log"

CUDA_VISIBLE_DEVICES="${OPD_TEACHER_GPU}" python3 -m sglang.launch_server \
  --model-path "${TEACHER_HF_DIR}" \
  --host 0.0.0.0 \
  --port "${TEACHER_PORT}" \
  --tp 1 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static "${OPD_TEACHER_MEM_FRACTION}" \
  >"${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!

cleanup() {
  set +e
  if [[ -n "${TEACHER_PID:-}" ]]; then
    kill "${TEACHER_PID}" >/dev/null 2>&1 || true
  fi
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Waiting for teacher server on ${TEACHER_IP}:${TEACHER_PORT}"
until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; do
  tail -n 20 "${TEACHER_LOG}" || true
  sleep 10
done
curl "http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info" || true
sleep 10

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source scripts/models/qwen3-8B.sh

if [[ -f "${OPD_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  OPD_LOAD_DIR="${OPD_SAVE_DIR}"
  echo "Resuming OPD from ${OPD_LOAD_DIR}"
else
  OPD_LOAD_DIR="${SFT_FINAL_HF_DIR}"
  echo "Starting OPD from final SFT HF snapshot ${OPD_LOAD_DIR}"
fi

CKPT_ARGS=(
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${STUDENT_HF_DIR}"
  --ref-load "${STUDENT_TORCH_DIST_DIR}"
  --load "${OPD_LOAD_DIR}"
  --save "${OPD_SAVE_DIR}"
  --save-interval "${OPD_SAVE_INTERVAL}"
  --save-hf "${OPD_HF_SNAPSHOT_TEMPLATE}"
)

ROLLOUT_ARGS=(
  --prompt-data "${OPD_JSONL}"
  --input-key prompt
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${OPD_NUM_ROLLOUT}"
  --rollout-batch-size "${OPD_ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${OPD_N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${OPD_MAX_RESPONSE_LEN}"
  --rollout-temperature 1
  --rollout-top-p 1
  --global-batch-size "${OPD_GLOBAL_BATCH_SIZE}"
  --balance-data
  --save-debug-rollout-data "${OPD_ROLLOUT_LOG_DIR}/rollout_{rollout_id}.pt"
)

RM_ARGS=(
  --custom-rm-path slime.rollout.on_policy_distillation.reward_func
  --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
  --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
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
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${OPD_MAX_TOKENS_PER_GPU}"
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

WANDB_ARGS=()
if [[ "${WANDB_MODE}" != "disabled" ]]; then
  WANDB_ARGS+=(--wandb-project slime-tillicum --wandb-group qwen3-8b-opd10k)
fi

SGLANG_ARGS=(
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

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ray start --head --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${OPD_RAY_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${SLIME_REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"HF_HOME\": \"${HF_HOME}\"
  }
}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${OPD_ACTOR_GPUS}" \
  --rollout-num-gpus "${OPD_ROLLOUT_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${RM_ARGS[@]}"
'

checkpoint_stop_pruner "${CHECKPOINT_PRUNER_PID:-}"
checkpoint_prune_old opd "${OPD_SAVE_DIR}"
checkpoint_report opd "${OPD_SAVE_DIR}" "${CHECKPOINT_REPORT_DIR}"
checkpoint_verify_single_latest opd "${OPD_SAVE_DIR}"

if [[ -f "${SPLIT_METADATA}" ]]; then
  cp -f "${SPLIT_METADATA}" "${OPD_SAVE_DIR}/split_metadata.json"
fi

if [[ ! -d "${OPD_FINAL_HF_DIR}" ]]; then
  echo "Missing final OPD HF eval snapshot: ${OPD_FINAL_HF_DIR}" >&2
  exit 1
fi

echo "Finished OPD at $(date)"
