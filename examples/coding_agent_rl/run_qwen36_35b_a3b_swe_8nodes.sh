#!/usr/bin/env bash
# End-to-end SWE coding-agent RL on 8 nodes. See README.md for the dataset
# schema, env vars, and fan-out semantics. Run from a long-lived shell / tmux
# session on the Ray head node (a short-lived nohup launcher gets its Ray child
# processes cleaned up with it).

# Best-effort cleanup so a rerun does not collide with stale workers.
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ============ model parallelism ============
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export ETP_SIZE="${ETP_SIZE:-1}"

# ============ rollout engine ============
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-8}"
ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-8}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.75}"

# ============ Qwen3.5-35B-A3B architecture ============
NLAYERS=40
FIRST_K_DENSE_REPLACE=0

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-96000}"
MAX_GEN_LEN="${MAX_GEN_LEN:-32768}"

# ============ paths — override before launching ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/Qwen3.6-35B-A3B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/path/to/Qwen3.6-35B-A3B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/path/to/swe_train.jsonl}"

EXP_TAG="${EXP_TAG:-agent_only}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"

# ============ logging ============
LOG_DIR="${RUN_ROOT}"
mkdir -p "${LOG_DIR}/rollout_dumps"
LOG_FILE="${LOG_DIR}/run.log"
echo "======================================================================"
echo "Training log: ${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "======================================================================"

MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 2
   --kv-channels 256
   --num-layers 40
   --hidden-size 2048
   --ffn-hidden-size 512
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # moe
   --moe-ffn-hidden-size 512
   --moe-shared-expert-intermediate-size 512
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "$MOE_LAYER_FREQ"
   --num-experts 256
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0

   # qwen3.5 specific
   --attention-output-gate
   --moe-shared-expert-gate
)

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
)

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --rollout-stop-token-ids 248046 248044
   --num-steps-per-rollout 1
   --global-batch-size 64
   --micro-batch-size 1
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size ${ETP_SIZE}
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # max-tokens-per-gpu is one CP rank's slice of MAX_CONTEXT_LEN; log-probs are
   # chunked along T to avoid OOM on long single trajectories.
   --max-tokens-per-gpu $((MAX_CONTEXT_LEN / CP_SIZE))
   --log-probs-chunk-size 1024
   --use-dynamic-batch-size
)

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus 64
   --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE}
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION}
   --sglang-enable-dp-attention
   --sglang-dp-size ${ROLLOUT_DP_SIZE}
   --sglang-ep-size ${ROLLOUT_EP_SIZE}
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
   --colocate
)

# ============ ray cluster network ============
# Set MASTER_ADDR before the SWE block: ADAPTER_PUBLIC_HOST below falls back to it.
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ SWE / claude-code rollout knobs ============

export SWE_AGENT="${SWE_AGENT:-claude_code}"
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-scaleswe}"
export E2B_API_KEY="${E2B_API_KEY:-e2b_0000000000000000000000000000000000000000}"
# Metadata key your gateway routes images by; `image` is the neutral default.
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY="${SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY:-image}"
export SLIME_AGENT_NODE_TARBALL="${SLIME_AGENT_NODE_TARBALL:-/path/to/node-v22.x-linux-x64.tar.xz}"
export SLIME_AGENT_CC_TARBALL="${SLIME_AGENT_CC_TARBALL:-/path/to/anthropic-ai-claude-code-local-linux-x64.tgz}"

# ADAPTER_PUBLIC_HOST must be routable from inside the sandbox (not 127.0.0.1).
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-1800}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-16}"

# autoCompactWindow (80k) < MAX_CONTEXT_LEN (96k) so the CLI compacts before any
# segment crosses the training-side cap. `investigator` is a read-only sub-agent
# (a concrete dispatch target). WebFetch/WebSearch off (no outbound internet).
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":80000}'
AGENTS_JSON='{"investigator":{"description":"Searches the repo for relevant files before any edit","prompt":"You are an investigator sub-agent. Use Grep/Read/Glob to find every file relevant to the user task, then return a short bulleted summary. Do NOT edit anything.","tools":["Grep","Read","Glob"]}}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --agents '${AGENTS_JSON}' --disallowedTools WebFetch WebSearch"

# Optional: require dispatching the investigator before any edit, to maximize sub-agent fan-out.
# export SWE_CC_PROMPT="Read PROBLEM_STATEMENT.md. BEFORE editing any file, dispatch the 'investigator' sub-agent (via the Agent tool with subagent_type=investigator) to locate every file relevant to the issue. Then fix the issue and run the tests."

# ============ proxy bypass for in-cluster traffic ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"

# ============ bring up ray cluster ============
HOSTFILE="${HOSTFILE:-/root/mpi_rack_hostfile}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${MLP_WORKER_NUM:-8}}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -f "${HOSTFILE}" ]]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    [[ -z "${WORKER_IP}" ]] && continue
    [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]] && continue
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh -o StrictHostKeyChecking=no "root@${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; \
       ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
         --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

echo "Waiting for Ray cluster to stabilize..."
sleep 30
ray status

# ============ runtime env propagated to ray workers ============
export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "SWE_AGENT",
    "E2B_API_KEY", "ADAPTER_PUBLIC_HOST",
    "SLIME_AGENT_NODE_TARBALL", "SLIME_AGENT_CC_TARBALL",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_BOOT_CONCURRENCY",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "SLIME_AGENT_CC_EXTRA_ARGS",
    "SLIME_AGENT_CC_EXTRA_ENVS",
    "SWE_CC_PROMPT",
    "SWE_TRAIN_PROTOCOL",
    "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
