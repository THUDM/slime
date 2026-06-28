#!/bin/bash
#SBATCH --job-name=slime-qwen3-math500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${SLURM_LOG_DIR}" "${EVAL_OUTPUT_DIR}" "${DATA_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export RAY_TMPDIR="/tmp/${USER:-suryadv}/slime-ray-${SLURM_JOB_ID}"
  mkdir -p "${RAY_TMPDIR}"
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Starting MATH-500 greedy eval on $(hostname) at $(date)"
echo "Eval targets: ${EVAL_TARGETS}"
echo "Eval output: ${EVAL_OUTPUT_DIR}"
echo "Eval skip completed: ${EVAL_SKIP_COMPLETED}"

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"

aggregate_if_available() {
  local dir="$1"
  local label="$2"
  if [[ -d "${dir}" ]] && find "${dir}" -mindepth 2 -maxdepth 2 -name summary.json -print -quit | grep -q .; then
    echo "Aggregating ${label}: ${dir}"
    python3 examples/qwen3_8b_opd_tillicum/summarize_eval.py \
      --aggregate-dir "${dir}" \
      --out-json "${dir}/summary_all.json"
  else
    echo "No per-stage summaries to aggregate for ${label}: ${dir}"
  fi
}

write_combined_report() {
  local found_summary=0
  local report_dir
  for report_dir in "${BASE_EVAL_OUTPUT_DIR}" "${SFT_EVAL_OUTPUT_DIR}" "${OPD_EVAL_OUTPUT_DIR}"; do
    if [[ -d "${report_dir}" ]] && find "${report_dir}" -mindepth 2 -maxdepth 2 -name summary.json -print -quit | grep -q .; then
      found_summary=1
      break
    fi
  done

  if [[ "${found_summary}" -eq 1 ]]; then
    mkdir -p "${COMBINED_EVAL_OUTPUT_DIR}"
    echo "Writing combined eval report: ${COMBINED_EVAL_OUTPUT_DIR}"
    python3 examples/qwen3_8b_opd_tillicum/summarize_eval.py \
      --combined-output-dir "${COMBINED_EVAL_OUTPUT_DIR}" \
      --base-dir "${BASE_EVAL_OUTPUT_DIR}" \
      --sft-dir "${SFT_EVAL_OUTPUT_DIR}" \
      --opd-dir "${OPD_EVAL_OUTPUT_DIR}" \
      --sft-total-samples "$((SFT_NUM_ROLLOUT * SFT_ROLLOUT_BATCH_SIZE))" \
      --sft-rollout-batch-size "${SFT_ROLLOUT_BATCH_SIZE}" \
      --opd-rollout-batch-size "${OPD_ROLLOUT_BATCH_SIZE}" \
      --out-json "${COMBINED_EVAL_OUTPUT_DIR}/summary_all.json"
  else
    echo "No summaries available for combined report yet."
  fi
}

write_all_reports() {
  aggregate_if_available "${BASE_EVAL_OUTPUT_DIR}" base
  aggregate_if_available "${SFT_EVAL_OUTPUT_DIR}" sft
  aggregate_if_available "${OPD_EVAL_OUTPUT_DIR}" opd
  write_combined_report
}

if [[ "${EVAL_TARGETS}" != "report" ]]; then
python3 - <<PY
import json
import os
from pathlib import Path
from datasets import load_dataset

out = Path(os.environ["MATH500_JSONL"])
cfg = Path(os.environ["MATH500_CONFIG"])
out.parent.mkdir(parents=True, exist_ok=True)

if not out.exists():
    ds = load_dataset(os.environ.get("MATH500_DATASET", "HuggingFaceH4/MATH-500"), split="test", cache_dir=os.environ.get("HF_HOME"))
    with out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            prompt = row.get("problem") or row.get("question") or row.get("prompt")
            label = row.get("answer") or row.get("final_answer") or row.get("label")
            if prompt is None or label is None:
                raise RuntimeError(f"Cannot infer prompt/label fields from MATH-500 row keys: {list(row.keys())}")
            f.write(json.dumps({"prompt": prompt, "label": label, "metadata": {"source_row_id": idx}}, ensure_ascii=False) + "\n")

max_response_len = int(os.environ.get("EVAL_MAX_RESPONSE_LEN", "32768"))
cfg.write_text(
    "eval:\n"
    "  defaults:\n"
    f"    max_response_len: {max_response_len}\n"
    "    temperature: 0\n"
    "    top_p: 1\n"
    "    n_samples_per_eval_prompt: 1\n"
    "    input_key: prompt\n"
    "    label_key: label\n"
    "    apply_chat_template: true\n"
    "  datasets:\n"
    "    - name: math500\n"
    f"      path: {out}\n"
    "      rm_type: math\n",
    encoding="utf-8",
)
print(f"Wrote {out}")
print(f"Wrote {cfg}")
PY
fi

run_eval() {
  local stage="$1"
  local load_dir="$2"
  local train_samples="$3"
  local stage_dir="${EVAL_OUTPUT_DIR}/${stage}"
  local summary_file="${stage_dir}/summary.json"
  local debug_file="${stage_dir}/debug_eval_0.pt"
  mkdir -p "${stage_dir}"

  summarize_stage() {
    python3 examples/qwen3_8b_opd_tillicum/summarize_eval.py \
      --stage "${stage}" \
      --debug-file "${debug_file}" \
      --out-json "${summary_file}" \
      --max-response-len "${EVAL_MAX_RESPONSE_LEN}" \
      --expected-samples "${EVAL_EXPECTED_SAMPLES}" \
      --train-samples "${train_samples}"
  }

  if [[ "${EVAL_SKIP_COMPLETED}" == "1" && -s "${summary_file}" ]]; then
    echo "EVAL_SKIP_COMPLETED stage=${stage} summary=${summary_file}"
    aggregate_if_available "${EVAL_OUTPUT_DIR}" current
    return
  fi

  if [[ "${EVAL_SKIP_COMPLETED}" == "1" && -s "${debug_file}" && ! -s "${summary_file}" ]]; then
    echo "Found existing debug file for ${stage}; attempting complete-debug salvage: ${debug_file}"
    if summarize_stage; then
      echo "EVAL_DEBUG_SALVAGE_OK stage=${stage}"
      aggregate_if_available "${EVAL_OUTPUT_DIR}" current
      return
    fi
    echo "EVAL_DEBUG_SALVAGE_FAILED stage=${stage}; rerunning full stage to preserve fidelity"
  fi

  if [[ ! -e "${load_dir}" ]]; then
    echo "Missing eval load dir for ${stage}: ${load_dir}" >&2
    exit 1
  fi

  echo "Evaluating ${stage} from ${load_dir} train_samples=${train_samples}"
  ray stop --force >/dev/null 2>&1 || true

  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l || true)
  if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
    HAS_NVLINK=1
  else
    HAS_NVLINK=0
  fi

  source scripts/models/qwen3-8B.sh

  CKPT_ARGS=(
    --megatron-to-hf-mode bridge
    --hf-checkpoint "${load_dir}"
    --ref-load "${STUDENT_TORCH_DIST_DIR}"
    --load "${load_dir}"
    --save "${stage_dir}/unused_save"
    --save-interval 100000
    --no-save-optim
  )

  ROLLOUT_ARGS=(
    --prompt-data "${MATH500_JSONL}"
    --input-key prompt
    --apply-chat-template
    --num-rollout 0
    --rollout-batch-size "${EVAL_ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt 1
    --rollout-max-response-len 1024
    --rollout-temperature 0
    --rollout-top-p 1
    --global-batch-size "${EVAL_ROLLOUT_BATCH_SIZE}"
    --rm-type math
    --save-debug-rollout-data "${stage_dir}/debug_{rollout_id}.pt"
  )

  EVAL_ARGS=(
    --eval-interval 1
    --eval-config "${MATH500_CONFIG}"
    --n-samples-per-eval-prompt 1
    --eval-temperature 0
    --eval-top-p 1
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --log-passrate
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
    --max-tokens-per-gpu "${EVAL_MAX_RESPONSE_LEN}"
  )

  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
  )

  GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
  )

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.7
    --sglang-server-concurrency "${EVAL_SGLANG_SERVER_CONCURRENCY}"
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
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${EVAL_ROLLOUT_NUM_GPUS}" \
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

  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --debug-rollout-only \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus "${EVAL_ROLLOUT_NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"

  ray stop --force >/dev/null 2>&1 || true

  summarize_stage
  aggregate_if_available "${EVAL_OUTPUT_DIR}" current
}

hf_snapshot_dir() {
  local root="$1"
  local rollout_id="$2"
  printf "%s/iter_%07d" "${root}" "${rollout_id}"
}

sample_count_for_rollout() {
  local rollout_id="$1"
  local batch_size="$2"
  local total_size="$3"
  local count=$(((rollout_id + 1) * batch_size))
  if [[ "${count}" -gt "${total_size}" ]]; then
    count="${total_size}"
  fi
  printf "%s" "${count}"
}

run_sft_curve() {
  local rid samples dir stage
  for rid in ${SFT_MILESTONE_ROLLOUT_IDS}; do
    samples="$(sample_count_for_rollout "${rid}" "${SFT_ROLLOUT_BATCH_SIZE}" "${SFT_SIZE}")"
    dir="$(hf_snapshot_dir "${SFT_HF_SNAPSHOT_DIR}" "${rid}")"
    stage="$(printf "sft_%06d" "${samples}")"
    run_eval "${stage}" "${dir}" "${samples}"
  done
}

run_opd_curve() {
  local rid samples dir stage
  for rid in ${OPD_MILESTONE_ROLLOUT_IDS}; do
    samples="$(sample_count_for_rollout "${rid}" "${OPD_ROLLOUT_BATCH_SIZE}" "${OPD_SIZE}")"
    dir="$(hf_snapshot_dir "${OPD_HF_SNAPSHOT_DIR}" "${rid}")"
    stage="$(printf "opd_%06d" "${samples}")"
    run_eval "${stage}" "${dir}" "${samples}"
  done
}

trap "ray stop --force >/dev/null 2>&1 || true" EXIT

for target in ${EVAL_TARGETS}; do
  case "${target}" in
    base)
      run_eval base "${STUDENT_HF_DIR}" 0
      ;;
    sft)
      run_sft_curve
      ;;
    opd)
      run_opd_curve
      ;;
    report)
      write_all_reports
      ;;
    *)
      echo "Unknown EVAL_TARGETS entry: ${target}" >&2
      exit 2
      ;;
  esac
done

aggregate_if_available "${EVAL_OUTPUT_DIR}" current
'

echo "Finished MATH-500 eval at $(date)"
