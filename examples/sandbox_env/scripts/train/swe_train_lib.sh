#!/usr/bin/env bash
# Shared function library for Qwen3.5-35B multi-node SWE training.
# Source this file AFTER setting ALL config and path variables in the calling script.

mkdir -p "${DATA_CACHE_DIR}" "${LOG_DIR}" "${SAVE_DIR}"

if [[ ! -d "${INSPIRE_SANDBOX_SITE_PACKAGES}" ]]; then
  echo "INSPIRE_SANDBOX_SITE_PACKAGES not found: ${INSPIRE_SANDBOX_SITE_PACKAGES}" >&2
  exit 1
fi

DEFAULT_SWE_LOAD_DIR=""
if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  DEFAULT_SWE_LOAD_DIR="${SAVE_DIR}"
fi
if [[ "${SWE_RESUME_TRAINING}" == "auto" ]]; then
  SWE_RESUME_TRAINING=0
  [[ -n "${DEFAULT_SWE_LOAD_DIR}" ]] && SWE_RESUME_TRAINING=1
fi
if [[ -z "${SWE_LOAD_DIR}" && -n "${DEFAULT_SWE_LOAD_DIR}" ]]; then
  SWE_LOAD_DIR="${DEFAULT_SWE_LOAD_DIR}"
fi

SWE_CLEANUP_SANDBOXES_ON_EXIT="${SWE_CLEANUP_SANDBOXES_ON_EXIT:-1}"
SWE_CLEANUP_STARTED_AT_EPOCH="${SWE_CLEANUP_STARTED_AT_EPOCH:-$(date +%s)}"
export SWE_CLEANUP_SANDBOXES_ON_EXIT SWE_CLEANUP_STARTED_AT_EPOCH
export SWE_WORK_ROOT="${WORK_ROOT}"

exec > >(tee "${LOG_DIR}/run.log") 2>&1

# Print an error message and stop the script.
die() { echo "$*" >&2; exit 1; }

# Best-effort cleanup for sandboxes observed in sample logs.
kill_sandboxes_from_batch_log_dir() {
  local batch_log_dir=$1
  local started_at=${2:-0}
  local label=${3:-"sandbox cleanup"}

  [[ -d "${batch_log_dir}" ]] || return 0

  echo "${label}: cleaning sandboxes from ${batch_log_dir}"
  PYTHONPATH="${INSPIRE_SANDBOX_SITE_PACKAGES}:${PYTHONPATH:-}" \
    python3 - "${batch_log_dir}" "${started_at}" <<'PY' || true
import json
import re
import sys
from pathlib import Path

from inspire_sandbox import Sandbox

path = Path(sys.argv[1])
started_at = float(sys.argv[2] or 0)
sandbox_ids: set[str] = set()
sandbox_id_re = re.compile(r"\bsandbox_id=([A-Za-z0-9_-]+)")


def is_current_run_file(file_path: Path) -> bool:
    try:
        return file_path.stat().st_mtime >= started_at - 5
    except OSError:
        return False


for artifact_path in path.glob("sample_*/sample_artifacts.json"):
    if not is_current_run_file(artifact_path):
        continue
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    for value in (
        artifact.get("sandbox_id"),
        artifact.get("extra_metadata", {}).get("sandbox_id")
        if isinstance(artifact.get("extra_metadata"), dict)
        else None,
    ):
        if value:
            sandbox_ids.add(str(value))

for log_path in path.glob("sample_*/sandbox/agent_output.log"):
    if not is_current_run_file(log_path):
        continue
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    sandbox_ids.update(match.group(1) for match in sandbox_id_re.finditer(text))

print(f"sandbox cleanup: {len(sandbox_ids)} ids found")
for sandbox_id in sorted(sandbox_ids):
    try:
        ok = Sandbox.kill(sandbox_id)
        print(f"killed {sandbox_id}: {ok}")
    except Exception as exc:
        print(f"failed {sandbox_id}: {exc}", file=sys.stderr)
PY
}

# Best-effort cleanup for sandboxes observed in this run's sample logs.
cleanup_run_sandboxes() {
  local rc=$?
  trap - EXIT INT TERM
  local batch_log_dir="${SWE_LOG_ROOT}/current_batch"

  if [[ "${NODE_RANK:-0}" -eq 0 ]] \
    && [[ "${SWE_CLEANUP_SANDBOXES_ON_EXIT}" == "1" ]] \
    && [[ "${SWE_KEEP_CONTAINERS:-0}" != "1" ]]; then
    kill_sandboxes_from_batch_log_dir "${batch_log_dir}" "${SWE_CLEANUP_STARTED_AT_EPOCH}" "exit cleanup"
  fi

  exit "${rc}"
}

# Kill sandboxes left by the previous training run before this run resets
# current_batch. Platform-side job stop can bypass EXIT cleanup.
cleanup_previous_run_sandboxes() {
  local batch_log_dir="${SWE_LOG_ROOT}/current_batch"

  if [[ "${NODE_RANK:-0}" -eq 0 ]] \
    && [[ "${SWE_CLEANUP_SANDBOXES_ON_EXIT}" == "1" ]] \
    && [[ "${SWE_KEEP_CONTAINERS:-0}" != "1" ]]; then
    kill_sandboxes_from_batch_log_dir "${batch_log_dir}" 0 "startup cleanup"
  fi
}

# Stop local Ray/SGLang processes before starting a new training job.
cleanup_local_processes() {
  pkill -9 sglang 2>/dev/null || true
  ray stop --force 2>/dev/null || true
  pkill -9 ray 2>/dev/null || true
  # Clear Triton kernel autotune cache to avoid loading half-written / corrupted
  # cubin from a prior crash (observed 2026-05-11: RuntimeError CUDA illegal memory
  # in fla.fused_norm_gate's autotune.check_disk_cache → load_binary path).
  rm -rf ~/.triton/cache /tmp/triton_cache /tmp/triton_cache_* /root/.triton 2>/dev/null || true
}

# Return the first host IP address, falling back to hostname when unavailable.
get_local_ip() {
  hostname -I 2>/dev/null | awk '{print $1}' || hostname
}

# Wait until the Ray dashboard reports all expected nodes and GPUs.
wait_for_full_ray_cluster() {
  local expected_gpus=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
  local expected_nodes=${NUM_NODES}
  local attempt

  for attempt in $(seq 1 300); do
    if python3 - <<PY
import json, sys, urllib.request
expected_gpus = ${expected_gpus}
expected_nodes = ${expected_nodes}
try:
    with urllib.request.urlopen("http://127.0.0.1:${DASHBOARD_PORT}/api/cluster_status", timeout=5) as r:
        payload = json.load(r)
except Exception:
    sys.exit(1)
report = payload.get("data", {}).get("clusterStatus", {}).get("autoscalerReport", {})
active_nodes = report.get("activeNodes") or {}
usage = payload.get("data", {}).get("clusterStatus", {}).get("loadMetricsReport", {}).get("usage", {})
gpu_total = (usage.get("GPU") or [0.0, 0.0])[1]
sys.exit(0 if len(active_nodes) >= expected_nodes and gpu_total >= expected_gpus else 1)
PY
    then
      echo "Ray cluster ready: ${expected_nodes} nodes, ${expected_gpus} GPUs."
      return 0
    fi
    echo "Waiting for Ray workers to join (${attempt}/300)..."
    sleep 10
  done

  echo "Ray workers did not join the cluster in time." >&2
  ray status --address="${MASTER_ADDR}:${MASTER_PORT}" || true
  return 1
}

# Start a Ray worker and retry transient join failures.
start_ray_worker_with_retry() {
  local attempt rc
  local node_name="${WORKER_ID:-${HOSTNAME:-worker-${NODE_RANK}}}"

  for attempt in $(seq 1 150); do
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

    if [[ "${rc}" -eq 0 ]]; then
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

# Return 1 when NVLink connections are detected, otherwise 0.
detect_nvlink() {
  local count
  count="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
  [[ "${count}" -gt 0 ]] && printf '1' || printf '0'
}

# Patch Qwen3.5-35B-A3B model args after sourcing the shared Slime model script.
override_qwen35_model_args() {
  local i
  for ((i=0; i<${#MODEL_ARGS[@]}; i++)); do
    case "${MODEL_ARGS[i]}" in
      --ffn-hidden-size)  MODEL_ARGS[i+1]=5632 ;;
      --num-query-groups) MODEL_ARGS[i+1]=2 ;;
      --hidden-size)      MODEL_ARGS[i+1]=2048 ;;
      --num-experts)      MODEL_ARGS[i+1]=256 ;;
    esac
  done
}

# Build normalized SWE-rebench train/val JSONL files used by rollout.
prepare_data() {
  DATA_ARGS=(
    --tasks-json "${REBENCH_TASKS_JSON}"
    --train-dest "${NORMALIZED_TRAIN}"
    --val-dest "${NORMALIZED_VAL}"
    --train-max "${SWE_TRAIN_MAX_PER_SOURCE}"
    --val-max "${SWE_VAL_MAX_PER_SOURCE}"
    --seed "${SWE_DATA_SEED}"
    --conversation-prompt
  )
  [[ "${SWE_SHUFFLE_DATA}" == "1" ]] && DATA_ARGS+=(--shuffle)
  [[ -n "${SWE_CONSUMABLE_TEMPLATE_MANIFEST}" ]] && DATA_ARGS+=(--consumable-template-manifest "${SWE_CONSUMABLE_TEMPLATE_MANIFEST}")
  [[ "${SWE_REQUIRE_CONSUMABLE_TEMPLATES}" == "1" ]] && DATA_ARGS+=(--require-consumable-templates)
  python3 "${SANDBOX_ENV_DIR}/build_rebench_runtime_data.py" "${DATA_ARGS[@]}"
}

# Convert the HF checkpoint into torch_dist format when the converted checkpoint is missing.
ensure_torch_dist_checkpoint() {
  if [[ "${FORCE_REBUILD_TORCH_DIST}" == "1" ]]; then
    echo "FORCE_REBUILD_TORCH_DIST=1, rebuilding ${TORCH_DIST_DIR}"
    rm -rf "${TORCH_DIST_DIR}"
  fi
  if [[ -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "Found torch_dist checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi
  if [[ -f "${TORCH_DIST_DIR}/common.pt" && -f "${TORCH_DIST_DIR}/metadata.json" ]]; then
    echo "Found torch_dist iteration checkpoint at ${TORCH_DIST_DIR}"
    return 0
  fi

  cd "${SLIME_DIR}"
  # shellcheck disable=SC1091
  source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"
  override_qwen35_model_args
  PYTHONPATH="${MEGATRON_PATH}:${SLIME_DIR}:${PYTHONPATH:-}" torchrun \
    --nproc-per-node "${NUM_GPUS_PER_NODE}" \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${MODEL_DIR}" \
    --save "${TORCH_DIST_DIR}"
}

# Validate cluster topology, batch sizing, and model parallelism before launch.
validate_layout() {
  local rollout_product=$(( SWE_ROLLOUT_BATCH_SIZE * SWE_SAMPLES_PER_PROMPT ))
  local actor_total_gpus=$(( ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE ))
  local num_query_groups=2 hidden_size=2048 num_experts=256

  (( NUM_NODES > 0 && NUM_GPUS_PER_NODE > 0 )) || die "NUM_NODES and NUM_GPUS_PER_NODE must be positive."
  (( ACTOR_NUM_NODES > 0 && ACTOR_GPUS_PER_NODE > 0 )) || die "ACTOR_NUM_NODES and ACTOR_GPUS_PER_NODE must be positive."
  (( ROLLOUT_GPUS_TOTAL > 0 )) || die "ROLLOUT_GPUS_TOTAL must be positive."
  (( ACTOR_NUM_NODES <= NUM_NODES )) || die "ACTOR_NUM_NODES cannot exceed NUM_NODES."
  (( ACTOR_GPUS_PER_NODE <= NUM_GPUS_PER_NODE )) || die "ACTOR_GPUS_PER_NODE cannot exceed NUM_GPUS_PER_NODE."
  (( actor_total_gpus + ROLLOUT_GPUS_TOTAL <= NUM_NODES * NUM_GPUS_PER_NODE )) || die "Actor GPUs plus rollout GPUs exceed total cluster GPUs."
  (( SWE_STEPS_PER_ROLLOUT > 0 )) || die "SWE_STEPS_PER_ROLLOUT must be positive."
  (( rollout_product % SWE_STEPS_PER_ROLLOUT == 0 )) || die "rollout_batch_size * n_samples_per_prompt must be divisible by num_steps_per_rollout."
  (( SWE_TENSOR_MODEL_PARALLEL_SIZE > 0 && SWE_EXPERT_MODEL_PARALLEL_SIZE > 0 )) || die "TP and EP must be positive."
  (( num_query_groups % SWE_TENSOR_MODEL_PARALLEL_SIZE == 0 )) || die "Qwen3.5-35B-A3B num_query_groups=2 must be divisible by TP."
  (( hidden_size % SWE_TENSOR_MODEL_PARALLEL_SIZE == 0 )) || die "Qwen3.5-35B-A3B hidden_size=2048 must be divisible by TP."
  (( num_experts % SWE_EXPERT_MODEL_PARALLEL_SIZE == 0 )) || die "Qwen3.5-35B-A3B num_experts=256 must be divisible by EP."
  (( actor_total_gpus % SWE_TENSOR_MODEL_PARALLEL_SIZE == 0 )) || die "Actor GPU topology must be divisible by TP."
  (( actor_total_gpus % SWE_CONTEXT_PARALLEL_SIZE == 0 )) || die "Actor GPU topology must be divisible by CP."
  (( actor_total_gpus % (SWE_TENSOR_MODEL_PARALLEL_SIZE * SWE_CONTEXT_PARALLEL_SIZE) == 0 )) || die "Actor GPU topology must be divisible by TP*CP."
  (( actor_total_gpus % SWE_EXPERT_MODEL_PARALLEL_SIZE == 0 )) || die "Actor GPU topology must be divisible by EP."
  (( ROLLOUT_GPUS_TOTAL % SWE_ROLLOUT_NUM_GPUS_PER_ENGINE == 0 )) || die "ROLLOUT_GPUS_TOTAL must be divisible by SWE_ROLLOUT_NUM_GPUS_PER_ENGINE."
  (( SWE_ROLLOUT_NUM_GPUS_PER_ENGINE % SWE_SGLANG_EP_SIZE == 0 )) || die "SWE_ROLLOUT_NUM_GPUS_PER_ENGINE must be divisible by SWE_SGLANG_EP_SIZE."
  (( num_experts % SWE_SGLANG_EP_SIZE == 0 )) || die "Qwen3.5-35B-A3B num_experts=256 must be divisible by SGLang EP."

  if [[ -z "${SWE_GLOBAL_BATCH_SIZE}" ]]; then
    SWE_GLOBAL_BATCH_SIZE=$(( rollout_product / SWE_STEPS_PER_ROLLOUT ))
  fi
  local expected_gbs=$(( rollout_product / SWE_STEPS_PER_ROLLOUT ))
  (( SWE_GLOBAL_BATCH_SIZE == expected_gbs )) \
    || die "SWE_GLOBAL_BATCH_SIZE=${SWE_GLOBAL_BATCH_SIZE} is inconsistent; expected ${expected_gbs}."
  local actor_dp=$(( actor_total_gpus / (SWE_TENSOR_MODEL_PARALLEL_SIZE * SWE_CONTEXT_PARALLEL_SIZE) ))
  (( actor_dp > 0 && SWE_GLOBAL_BATCH_SIZE % actor_dp == 0 )) \
    || die "SWE_GLOBAL_BATCH_SIZE (${SWE_GLOBAL_BATCH_SIZE}) must be divisible by actor DP (${actor_dp})."
  (( SWE_GROUP_CONCURRENCY > 0 && SWE_GROUP_CONCURRENCY <= SWE_ROLLOUT_BATCH_SIZE )) \
    || die "SWE_GROUP_CONCURRENCY must be in (0, SWE_ROLLOUT_BATCH_SIZE]."
  (( SWE_SAMPLE_CONCURRENCY >= SWE_SAMPLES_PER_PROMPT )) \
    || die "SWE_SAMPLE_CONCURRENCY must be >= SWE_SAMPLES_PER_PROMPT."
}

# Assemble train_async.py arguments, write the Ray job entrypoint, and submit it.
submit_ray_job() {
  cd "${SLIME_DIR}"
  # shellcheck disable=SC1091
  source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"
  override_qwen35_model_args

  local has_nvlink
  has_nvlink="$(detect_nvlink)"

  LOAD_ARGS=()
  [[ "${SWE_RESUME_TRAINING}" == "1" && -n "${SWE_LOAD_DIR}" ]] && LOAD_ARGS+=(--load "${SWE_LOAD_DIR}")

  ROLLOUT_ARGS=(
    --prompt-data "${NORMALIZED_TRAIN}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    # --apply-chat-template
    --rollout-shuffle
    --num-rollout "${SWE_NUM_ROLLOUT}"
    --rollout-batch-size "${SWE_ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${SWE_SAMPLES_PER_PROMPT}"
    --rollout-max-context-len "${SWE_MAX_CONTEXT_LEN}"
    --rollout-max-response-len "${SWE_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --rollout-top-p 0.95
    --rollout-stop "</tool_call>" "</tool_call>\n" "\n</tool_call>\n" "\n</function>"
    --global-batch-size "${SWE_GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${SWE_MICRO_BATCH_SIZE}"
    --num-steps-per-rollout "${SWE_STEPS_PER_ROLLOUT}"
    --loss-mask-type qwen3_5
    --rollout-function-path examples.sandbox_env.swe_rollout.generate_rollout
  )
  # Optional: GRPO-style dynamic sampling — oversample prompts then drop groups
  # whose rewards have zero std (no advantage signal). slime ships
  # `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`.
  # Enable by setting SWE_DYNAMIC_SAMPLING_FILTER_PATH + SWE_OVER_SAMPLING_BATCH_SIZE.
  [[ -n "${SWE_OVER_SAMPLING_BATCH_SIZE:-}" ]] && ROLLOUT_ARGS+=(--over-sampling-batch-size "${SWE_OVER_SAMPLING_BATCH_SIZE}")
  [[ -n "${SWE_DYNAMIC_SAMPLING_FILTER_PATH:-}" ]] && ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${SWE_DYNAMIC_SAMPLING_FILTER_PATH}")

  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${SWE_LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 "${SWE_ADAM_BETA2}"
  )

  GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef "${SWE_KL_LOSS_COEF}"
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
  )

  PERF_ARGS=(
    --tensor-model-parallel-size "${SWE_TENSOR_MODEL_PARALLEL_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size "${SWE_CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${SWE_EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --max-tokens-per-gpu "${SWE_MAX_TOKENS_PER_GPU}"
    --log-probs-chunk-size 512
    --recompute-loss-function
  )
  [[ "${SWE_USE_DYNAMIC_BATCH_SIZE}" == "1" ]] && PERF_ARGS+=(--use-dynamic-batch-size)

  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${SWE_ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SWE_SGLANG_MEM_FRACTION_STATIC}"
    --sglang-ep-size "${SWE_SGLANG_EP_SIZE}"
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
  )
  # Optional SGLang throughput knobs — opt-in via env (defaults preserve old behaviour).
  # H200 max-throughput template:
  #   SWE_SGLANG_SCHEDULE_CONSERVATIVENESS=1.2
  #   SWE_SGLANG_MAX_RUNNING_REQUESTS=192
  #   SWE_SGLANG_ENABLE_PREFILL_DELAYER=1
  #   SWE_SGLANG_PREFILL_DELAYER_LOW_WATERMARK=0.9
  #   SWE_SGLANG_SCHEDULE_POLICY=lpm
  #   SWE_SGLANG_ALLOW_AUTO_TRUNCATE=1
  #   SWE_SGLANG_REASONING_PARSER=qwen3
  #   SWE_SGLANG_TOOL_CALL_PARSER=qwen3_coder
  #   SWE_SGLANG_MAMBA_SCHEDULER_STRATEGY=extra_buffer
  [[ -n "${SWE_SGLANG_SCHEDULE_CONSERVATIVENESS:-}" ]] && SGLANG_ARGS+=(--sglang-schedule-conservativeness "${SWE_SGLANG_SCHEDULE_CONSERVATIVENESS}")
  [[ -n "${SWE_SGLANG_MAX_RUNNING_REQUESTS:-}" ]] && SGLANG_ARGS+=(--sglang-max-running-requests "${SWE_SGLANG_MAX_RUNNING_REQUESTS}")
  [[ "${SWE_SGLANG_ENABLE_PREFILL_DELAYER:-0}" == "1" ]] && SGLANG_ARGS+=(--sglang-enable-prefill-delayer)
  [[ -n "${SWE_SGLANG_PREFILL_DELAYER_LOW_WATERMARK:-}" ]] && SGLANG_ARGS+=(--sglang-prefill-delayer-token-usage-low-watermark "${SWE_SGLANG_PREFILL_DELAYER_LOW_WATERMARK}")
  [[ -n "${SWE_SGLANG_SCHEDULE_POLICY:-}" ]] && SGLANG_ARGS+=(--sglang-schedule-policy "${SWE_SGLANG_SCHEDULE_POLICY}")
  [[ "${SWE_SGLANG_ALLOW_AUTO_TRUNCATE:-0}" == "1" ]] && SGLANG_ARGS+=(--sglang-allow-auto-truncate)
  [[ -n "${SWE_SGLANG_REASONING_PARSER:-}" ]] && SGLANG_ARGS+=(--sglang-reasoning-parser "${SWE_SGLANG_REASONING_PARSER}")
  [[ -n "${SWE_SGLANG_TOOL_CALL_PARSER:-}" ]] && SGLANG_ARGS+=(--sglang-tool-call-parser "${SWE_SGLANG_TOOL_CALL_PARSER}")
  [[ -n "${SWE_SGLANG_MAMBA_SCHEDULER_STRATEGY:-}" ]] && SGLANG_ARGS+=(--sglang-mamba-scheduler-strategy "${SWE_SGLANG_MAMBA_SCHEDULER_STRATEGY}")
  # MTP speculative decoding on rollout side — uses SFT ckpt's MTP head to draft
  # K future tokens per forward pass. Requires the SFT ckpt to have mtp weights.
  if [[ "${SGLANG_ENABLE_MTP_ROLLOUT:-0}" == "1" ]]; then
    SGLANG_ARGS+=(
      --sglang-speculative-algorithm "${SGLANG_SPECULATIVE_ALGORITHM:-EAGLE}"
      --sglang-speculative-num-steps "${SGLANG_SPECULATIVE_NUM_STEPS:-3}"
      --sglang-speculative-eagle-topk "${SGLANG_SPECULATIVE_EAGLE_TOPK:-1}"
      --sglang-speculative-num-draft-tokens "${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
    )
  fi

  MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
  )
  # Train the MTP head jointly with the main model. Auxiliary loss with scale
  # MTP_LOSS_SCALING_FACTOR keeps MTP head fresh so rollout-side speculative
  # decoding maintains high accept rate as actor weights drift.
  if [[ "${ENABLE_MTP_TRAINING:-0}" == "1" ]]; then
    MISC_ARGS+=(
      --enable-mtp-training
      --mtp-num-layers "${MTP_NUM_LAYERS:-1}"
      --mtp-loss-scaling-factor "${MTP_LOSS_SCALING_FACTOR:-0.2}"
    )
  fi
  # R3 (Rollout Routing Replay): record MoE expert routing decisions on rollout
  # side and replay them in actor backward. Keeps training perfectly on-policy
  # at the router level when actor/rollout TP/EP layouts differ. Now combined
  # with MTP — cherry-picked upstream f8879db2 makes the actor side tolerate
  # MTP-only routers (skip_replay) so the asserts don't fire.
  if [[ "${USE_R3:-0}" == "1" ]]; then
    MISC_ARGS+=(--use-rollout-routing-replay)
  fi

  WANDB_ARGS=()
  if [[ "${SWE_USE_WANDB}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS+=(
      --use-wandb
      --wandb-host "${WANDB_BASE_URL:-https://wandb.ai}"
      --wandb-project "${SWE_WANDB_PROJECT}"
      --wandb-group "${SWE_WANDB_GROUP}"
      --wandb-run-id "${SWE_WANDB_RUN_ID}"
      --wandb-key "${WANDB_API_KEY}"
      --disable-wandb-random-suffix
    )
  fi

  DEBUG_ARGS=()
  [[ "${SWE_DEBUG_ROLLOUT_ONLY}" == "1" ]] && DEBUG_ARGS+=(--debug-rollout-only)

  OFFLOAD_ARGS=()
  [[ "${SWE_OPTIMIZER_CPU_OFFLOAD}" == "1" ]] && OFFLOAD_ARGS+=(--optimizer-cpu-offload)
  [[ "${SWE_OVERLAP_CPU_OPTIMIZER_D2H_H2D}" == "1" ]] && OFFLOAD_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)
  [[ "${SWE_USE_PRECISION_AWARE_OPTIMIZER}" == "1" ]] && OFFLOAD_ARGS+=(--use-precision-aware-optimizer)

  RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${SHARE_WORKSPACE}:${INSPIRE_SANDBOX_SITE_PACKAGES}:${WORKSPACE_ROOT}:${AVALANCHE_ROOT}:${MEGATRON_PATH}:${SLIME_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TRITON_CACHE_DIR": "/tmp/triton_cache",
    "NCCL_NVLS_ENABLE": "${has_nvlink}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "WANDB_API_KEY": "${WANDB_API_KEY:-}",
    "WANDB_BASE_URL": "${WANDB_BASE_URL:-}",
    "SBX_API_KEY": "${SBX_API_KEY:-}",
    "SBX_API_URL": "${SBX_API_URL:-}",
    "INSP_GITHUB_TOKEN": "${INSP_GITHUB_TOKEN:-}",
    "SWE_AGENT_HARNESS": "${SWE_AGENT_HARNESS}",
    "AGENTIC_PROTOCOL_ROOT": "${AGENTIC_PROTOCOL_ROOT:-}",
    "SWE_MODEL_PROXY_PORT": "${SWE_MODEL_PROXY_PORT}",
    "SWE_WSTUNNEL_SERVER_PORT": "${SWE_WSTUNNEL_SERVER_PORT}",
    "SWE_MAX_TURNS": "${SWE_MAX_TURNS}",
    "SWE_AGENT_FINISH_TIMEOUT": "${SWE_AGENT_FINISH_TIMEOUT}",
    "SWE_WAIT_TIMEOUT": "${SWE_WAIT_TIMEOUT}",
    "SWE_KEEP_CONTAINERS": "${SWE_KEEP_CONTAINERS}",
    "SWE_SANDBOX_START_RETRY_TIMES": "${SWE_SANDBOX_START_RETRY_TIMES}",
    "SWE_SANDBOX_START_RETRY_INTERVAL": "${SWE_SANDBOX_START_RETRY_INTERVAL}",
    "SWE_LOG_ROOT": "${SWE_LOG_ROOT}",
    "SWE_GROUP_CONCURRENCY": "${SWE_GROUP_CONCURRENCY}",
    "SWE_SAMPLE_CONCURRENCY": "${SWE_SAMPLE_CONCURRENCY}",
    "SWE_OVER_SAMPLING_BATCH_SIZE": "${SWE_OVER_SAMPLING_BATCH_SIZE}"
  }
}
EOF
)

  JOB_CMD=(
    python3 "${SLIME_DIR}/train_async.py"
    --actor-num-nodes "${ACTOR_NUM_NODES}"
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
    --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}"
    "${OFFLOAD_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${TORCH_DIST_DIR}"
    "${LOAD_ARGS[@]}"
    --save "${SAVE_DIR}"
    --save-interval 5
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${WANDB_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${DEBUG_ARGS[@]}"
  )

  {
    printf '#!/bin/bash\n'
    printf 'set -euo pipefail\n'
    printf 'cd %q\n' "${SLIME_DIR}"
    printf 'exec '
    printf '%q ' "${JOB_CMD[@]}"
    printf '\n'
  } > "${JOB_ENTRYPOINT_SCRIPT}"
  chmod +x "${JOB_ENTRYPOINT_SCRIPT}"

  ray job submit --address="http://127.0.0.1:${DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- bash "${JOB_ENTRYPOINT_SCRIPT}"
}

# ── Main execution ────────────────────────────────────────────────────────────

NODE_RANK=${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-10.244.68.85}}"
MASTER_PORT=${MASTER_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
NODE_IP="$(get_local_ip)"

if [[ "${NODE_RANK}" -eq 0 ]]; then
  trap cleanup_run_sandboxes EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
fi

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"

cleanup_previous_run_sandboxes
cleanup_local_processes
validate_layout

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
