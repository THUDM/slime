#!/usr/bin/env bash
# Plan-D long-arm E2 v3: per-segment fan-out + PR #1933 per-rollout reducer
# + Plan B wall-clock guard against external builtin TimeoutError.
#
# Branch: list_trajectory_0522_pr1933_e2_v3 (off pr1933 HEAD @ 82135cc8)
#   * 12 PR #1933 port commits + cp_size hotfix + log skip-list hotfix
#   * 1 plan-d wall-clock guard commit (82135cc8)
#   * 47/47 smoke tests pass
#
# DIFF vs planD_e2_pr1933_fanout.sh (E2 retry 2 that died at 27 min on a
# hung evaluate sandbox):
#   * SWE_GENERATE_GUARD_SEC=1980 explicit (default in code is same)
#   * Scenario renamed to planD_e2_v3_with_guard
#   * No other config change — same fan-out, same archive baseline, same
#     batch shape so we can compare directly against E2 retry 2.
#
# PASS criteria (relaxed from earlier):
#   - Reach at least 1 train step PASS without ray job entrypoint OOM
#   - If a trajectory hangs > 1980s, observe `wall_clock_timeout` abort
#     in log AND ray job continues (not dies)
#
# EXPECTED OUTCOME:
#   v3 should reproduce the conditions that killed E2 retry 2, but the
#   guard catches the hung sample as an _abort and the rest of the rollout
#   continues. Best case: 4-step long-run completes. Worst case: a few
#   wall_clock_timeout aborts logged but the run survives.

set -euo pipefail
set -x

SLIME_DIR="${SLIME_DIR:-/mnt/jingshenghang/code/slime_swe/slime-0522-impl}"

SANDBOX_METADATA_DEFAULT="/mnt/jingshenghang/code/slime_swe/0521/configs/sandbox_metadata.example.json"

source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
HOSTFILE="${HOSTFILE:-/root/mpi_rack_hostfile}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
SOCKET_IFNAME="${SOCKET_IFNAME:-eth5200}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/jingshenghang/storage/checkpoints/Qwen3.6-35B-A3B}"
REF_LOAD="${REF_LOAD:-/mnt/jingshenghang/storage/checkpoints/Qwen3.6-35B-A3B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/jingshenghang/code/slime_swe/datasets/swe-train-1545-localcache.jsonl}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SCENARIO="${SCENARIO:-planD_e2_v3_with_guard}"
RUN_ROOT="${RUN_ROOT:-/mnt/jingshenghang/code/slime_swe/0522/logs/${SCENARIO}_${STAMP}}"

# --- claude-code / middleware knobs (mirror archive base) ----------------
export E2B_API_KEY="${E2B_API_KEY:-glm-platform}"
export SLIME_HEAD_HOST="${SLIME_HEAD_HOST:-172.27.14.209}"
export SWE_HOST_NODE_TARBALL="${SWE_HOST_NODE_TARBALL:-/mnt/jingshenghang/software/node-v22.20.0-linux-x64.tar.xz}"
export SWE_HOST_CC_TARBALL="${SWE_HOST_CC_TARBALL:-/mnt/jingshenghang/storage/claude_code/anthropic-ai-claude-code-2.1.143-local-linux-x64.tgz}"
export SWE_TIME_BUDGET_SEC="${SWE_TIME_BUDGET_SEC:-1200}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-8}"
export SWE_SAVE_TRAJECTORY_TREE="${SWE_SAVE_TRAJECTORY_TREE:-1}"
export SWE_MAX_RESPONSE_TOKENS="${SWE_MAX_RESPONSE_TOKENS:-10240}"
export SWE_TOOL_PARSER="${SWE_TOOL_PARSER:-qwen25}"
export SWE_REASONING_PARSER="${SWE_REASONING_PARSER:-qwen3}"
export SHIM_BIND_HOST="${SHIM_BIND_HOST:-0.0.0.0}"
export SHIM_PORT="${SHIM_PORT:-18001}"
export SWE_SANDBOX_METADATA_FILE="${SWE_SANDBOX_METADATA_FILE:-${SANDBOX_METADATA_DEFAULT}}"
export SWE_SANDBOX_IMAGE_METADATA_KEY="${SWE_SANDBOX_IMAGE_METADATA_KEY:-glm-platform/image}"
export SWE_ABORT_POLL_INTERVAL="${SWE_ABORT_POLL_INTERVAL:-0.5}"
export SWE_ABORT_MAX_WAIT_SEC="${SWE_ABORT_MAX_WAIT_SEC:-1800}"
export SWE_ABORT_RESUME_MAX_ATTEMPTS="${SWE_ABORT_RESUME_MAX_ATTEMPTS:-8}"
export SWE_ABORT_RESUME_MIN_TOKENS="${SWE_ABORT_RESUME_MIN_TOKENS:-16}"

# *** E2 KEY KNOB: enable per-segment fan-out ***
export SWE_LIST_TRAJECTORY=1

# *** E2 v3: explicit wall-clock guard for the entire generate() call ***
# Default in generate.py is SWE_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 180
# (= 1200 + 600 + 180 = 1980s, ~33 min). We export it explicitly so the
# value is captured in the runtime env JSON below. A single hung trajectory
# > 1980s will now _abort instead of killing the whole ray job via external
# builtin TimeoutError (see r5/r6).
export SWE_GENERATE_GUARD_SEC="${SWE_GENERATE_GUARD_SEC:-1980}"

# --- scenario knobs: compact_aggressive verbatim from archive -----------
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":100000}'
export SWE_CLAUDE_EXTRA_ARGS="${SWE_CLAUDE_EXTRA_ARGS:---settings '${SETTINGS_JSON}' --disable-slash-commands --disallowedTools Agent Task WebFetch WebSearch}"

# --- batch / parallel config: SAME as E0 v1 retry for fair comparison ---
export NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
# NOTE: With PR #1933, --global-batch-size unit is now "rollout" not "sample".
# Pre-fan-out we have 16 rollouts (rollout_batch*n=4*4); each rollout
# fan-outs to K segments. GBS=16 means train all 16 rollouts per step.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-200000}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-4}"
export EP_SIZE="${EP_SIZE:-8}"
export ETP_SIZE="${ETP_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${CP_SIZE}}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"

# DELIBERATELY UNSET (match E0 v1 retry):
#   * TRAIN_MEMORY_MARGIN_BYTES (let slime default 1 GiB; 4 GiB is a kill-switch)
#   * SLIME_MAX_SAMPLE_TOKENS    (no per-sample cap; pure fan-out)
#   * --use-dynamic-global-batch-size flag (PR #1933 makes new behavior default;
#     setting the legacy flag now triggers a deprecation warning + fallback)

mkdir -p "${RUN_ROOT}/rollout_dumps"

# Persist the exact commit that produced this run so archival can record it.
( cd "${SLIME_DIR}" && git log -1 --format='%H %s' > "${RUN_ROOT}/slime_commit.txt" )

cd "${SLIME_DIR}"

INTERNAL_NO_PROXY="localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,100.64.0.0/10,172.16.0.0/12,${MASTER_ADDR},${SLIME_HEAD_HOST}"
export no_proxy="${no_proxy:+${no_proxy},}${INTERNAL_NO_PROXY}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${INTERNAL_NO_PROXY}"

if [[ "${SKIP_RAY_START:-0}" != "1" ]]; then
  ray stop --force || true
  pkill -9 -f '^ray::' || true
  pkill -9 -x sglang || true
  pkill -9 -x slime || true
  pkill -9 -x redis || true

  ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

  if [[ -f "${HOSTFILE}" ]]; then
    n=0
    while read -r worker_ip _ || [[ -n "${worker_ip:-}" ]]; do
      [[ -z "${worker_ip}" ]] && continue
      n=$((n + 1))
      (( n > ACTOR_NUM_NODES )) && break
      [[ "${worker_ip}" == "${MASTER_ADDR}" ]] && continue
      ssh -o StrictHostKeyChecking=no "root@${worker_ip}" \
        "ray stop --force || true; \
         pkill -9 -f '^ray::' || true; \
         pkill -9 -x sglang || true; \
         pkill -9 -x slime || true; \
         pkill -9 -x redis || true; \
         ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
           --node-ip-address ${worker_ip} --disable-usage-stats" &
    done < "${HOSTFILE}"
    wait
  fi
fi

RUNTIME_ENV_JSON="$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "E2B_API_KEY", "SLIME_HEAD_HOST",
    "SWE_HOST_NODE_TARBALL", "SWE_HOST_CC_TARBALL",
    "SWE_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_BOOT_CONCURRENCY",
    "SWE_SAVE_TRAJECTORY_TREE", "SWE_MAX_RESPONSE_TOKENS",
    "SWE_TOOL_PARSER", "SWE_REASONING_PARSER",
    "SHIM_BIND_HOST", "SHIM_PORT",
    "SWE_CLAUDE_EXTRA_ARGS",
    "SWE_SANDBOX_METADATA_FILE", "SWE_SANDBOX_IMAGE_METADATA_KEY",
    "SWE_ABORT_POLL_INTERVAL", "SWE_ABORT_MAX_WAIT_SEC",
    "SWE_ABORT_RESUME_MAX_ATTEMPTS", "SWE_ABORT_RESUME_MIN_TOKENS",
    "SWE_LIST_TRAJECTORY",
    "SWE_GENERATE_GUARD_SEC",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = "${MASTER_ADDR}"
env["GLOO_SOCKET_IFNAME"] = "${SOCKET_IFNAME}"
env["TP_SOCKET_IFNAME"] = "${SOCKET_IFNAME}"
env["NCCL_SOCKET_IFNAME"] = "${SOCKET_IFNAME}"
env["PYTHONPATH"] = f"/root/Megatron-LM/:${SLIME_DIR}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${REF_LOAD}" \
    --custom-generate-function-path examples.coding_agent_rl.generate.generate \
    --prompt-data "${PROMPT_DATA}" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --rollout-shuffle \
    --num-rollout "${NUM_ROLLOUT}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}" \
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
    --rollout-temperature 1.0 \
    --rollout-stop-token-ids 248046 248044 \
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --micro-batch-size 1 \
    --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt" \
    --advantage-estimator gspo \
    --kl-loss-coef 0.00 \
    --kl-loss-type low_var_kl \
    --kl-coef 0.00 \
    --entropy-coef 0.00 \
    --eps-clip 1e-4 \
    --eps-clip-high 2e-4 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --optimizer-cpu-offload \
    --overlap-cpu-optimizer-d2h-h2d \
    --use-precision-aware-optimizer \
    --tensor-model-parallel-size "${TP_SIZE}" \
    --sequence-parallel \
    --pipeline-model-parallel-size "${PP_SIZE}" \
    --context-parallel-size "${CP_SIZE}" \
    --expert-model-parallel-size "${EP_SIZE}" \
    --expert-tensor-parallel-size "${ETP_SIZE}" \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}" \
    --use-dynamic-batch-size \
    --rollout-num-gpus 64 \
    --rollout-num-gpus-per-engine 8 \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
    --sglang-enable-dp-attention \
    --sglang-dp-size 8 \
    --sglang-ep-size 8 \
    --sglang-enable-dp-lm-head \
    --sglang-moe-dense-tp-size 1 \
    --sglang-tool-call-parser qwen25 \
    --sglang-reasoning-parser qwen3 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --colocate \
  2>&1 | tee "${RUN_ROOT}/run.log"

EXIT_CODE=${PIPESTATUS[0]}

echo "exit_code=${EXIT_CODE}" > "${RUN_ROOT}/done.marker"
echo "RUN_ROOT=${RUN_ROOT}" >> "${RUN_ROOT}/done.marker"
echo "RUN_ROOT=${RUN_ROOT}"
exit "${EXIT_CODE}"
