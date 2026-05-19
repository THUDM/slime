#!/bin/bash
# Coding-Agent RL — reference launch script (GLM-4.7-355B-A32B, 8 nodes × 8 GPU,
# colocate, E2B sandbox).
#
# This is a TEMPLATE: every path / host you see below must be adapted to your
# cluster. The training hyper-params, parallelism layout, SGLang/Megatron knobs
# and runtime-env wiring are the actually-reusable bits — copy this and edit
# the variables at the top.
#
# Required env vars (set in your shell or a sourced ``.env``):
#
#   E2B_API_KEY               — E2B cloud API key
#   SLIME_HEAD_HOST           — public host/IP the sandboxes use to reach the middleware
#   SWE_HOST_NODE_TARBALL     — host path to a Node 22 tarball
#   SWE_HOST_CC_TARBALL       — host path to the Claude Code npm tarball
#   HF_CHECKPOINT             — HF checkpoint dir (or HF id) used as actor init
#   REF_LOAD                  — Megatron torch_dist ref checkpoint dir
#   PROMPT_DATA               — jsonl with {prompt,label,metadata} per row
#
# Optional:
#
#   E2B_ENV_FILE              — path to a .env to source for E2B_API_KEY etc.
#   SWE_SANDBOX_METADATA_JSON — JSON object passed verbatim into
#                                 AsyncSandbox.create(metadata=...). Use this if
#                                 your backend reads routing/size tags from
#                                 metadata; default is empty.
#   MASTER_ADDR               — defaults to ${MLP_WORKER_0_HOST}
#   HOSTFILE                  — defaults to /root/mpi_rack_hostfile
#   SOCKET_IFNAME             — NCCL/GLOO socket interface name
#   ACTOR_NUM_NODES           — defaults to 8
#   ACTOR_NUM_GPUS_PER_NODE   — defaults to 8
#
# Wire-up to slime:
#
#   --custom-generate-function-path examples.coding_agent_rl.generate.generate
#
# To swap model: change CKPT_ARGS + ``SWE_TOOL_PARSER`` / ``SWE_REASONING_PARSER``
# env (e.g. Qwen3: ``tool_parser=qwen25 reasoning_parser=qwen3``). The
# generate/middleware code is model-agnostic.

set -ex

# clean local node (slime scripts handle the rest of the cluster)
pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 2
pkill -9 ray || true
pkill -9 python || true

export PYTHONBUFFERED=16
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-${SCRIPT_DIR}/../..}"   # examples/coding_agent_rl/.. -> slime root

# Optionally source a .env with E2B_API_KEY etc. Set E2B_ENV_FILE to point at
# your secrets file, or export E2B_API_KEY in your shell directly.
if [ -n "${E2B_ENV_FILE:-}" ] && [ -f "${E2B_ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${E2B_ENV_FILE}"
  set +a
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

: "${E2B_API_KEY:?E2B_API_KEY must be set (export or via E2B_ENV_FILE)}"
: "${SLIME_HEAD_HOST:?SLIME_HEAD_HOST must be set to a host/IP reachable from sandboxes}"
: "${SWE_HOST_NODE_TARBALL:?SWE_HOST_NODE_TARBALL must point to a Node 22 tarball}"
: "${SWE_HOST_CC_TARBALL:?SWE_HOST_CC_TARBALL must point to the Claude Code tarball}"
: "${HF_CHECKPOINT:?HF_CHECKPOINT must be set}"
: "${REF_LOAD:?REF_LOAD must be set (Megatron torch_dist ref checkpoint dir)}"
: "${PROMPT_DATA:?PROMPT_DATA must be set (jsonl dataset path)}"

export SLIME_HEAD_HOST

# Agent / sandbox knobs (read by examples.coding_agent_rl.generate at import).
export SWE_HOST_NODE_TARBALL
export SWE_HOST_CC_TARBALL
export SWE_TIME_BUDGET_SEC=${SWE_TIME_BUDGET_SEC:-3600}
export SWE_EVAL_TIMEOUT_SEC=${SWE_EVAL_TIMEOUT_SEC:-300}
# Optional JSON dict passed into AsyncSandbox.create(metadata=...). Most users
# can leave this unset; only set it if your sandbox backend reads routing tags
# from metadata, e.g. SWE_SANDBOX_METADATA_JSON='{"my-platform/size":"lg"}'.
export SWE_SANDBOX_METADATA_JSON=${SWE_SANDBOX_METADATA_JSON:-}
export SWE_TOOL_PARSER=${SWE_TOOL_PARSER:-glm47}
export SWE_REASONING_PARSER=${SWE_REASONING_PARSER:-glm45}
export SHIM_BIND_HOST=${SHIM_BIND_HOST:-0.0.0.0}
export SHIM_PORT=${SHIM_PORT:-18001}

cd "${SLIME_DIR}"

# Pull GLM-4.7-355B MODEL_ARGS via the existing model recipe.
source "${SCRIPT_DIR}/../../../models/glm4.5-355B-A32B.sh" 2>/dev/null \
  || source "${SCRIPT_DIR}/../../../../models/glm4.5-355B-A32B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
)

if [ ! -f "${PROMPT_DATA}" ]; then
  echo "PROMPT_DATA not found: ${PROMPT_DATA}" >&2
  exit 1
fi

ROLLOUT_ARGS=(
   # NOTE: NOT setting --rollout-function-path; we keep slime's default sglang
   # outer loop and only swap the per-sample generate via --custom-...
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle

   --num-rollout ${NUM_ROLLOUT:-20}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-32}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
   --rollout-max-context-len 32768
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --rollout-stop-token-ids 151329 151336 151338

   --num-steps-per-rollout ${NUM_STEPS_PER_ROLLOUT:-4}
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 2
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
   --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE:-1024}
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 1e-4
   --eps-clip-high 2e-4
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
   --rollout-num-gpus-per-engine 16
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-dp-size 2
   --sglang-ep-size 16
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   # SGLang server-side parsers are not strictly needed (the middleware does its own
   # parsing on /generate output), but keep them on for /v1/chat consumers.
   --sglang-tool-call-parser glm47
   --sglang-reasoning-parser glm45
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-8}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_NODES=${ROLLOUT_NUM_NODES:-0}
SOCKET_IFNAME=${SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}

EXTRA_ARGS=( --colocate )

MASTER_ADDR=${MASTER_ADDR:-${MLP_WORKER_0_HOST:?MASTER_ADDR or MLP_WORKER_0_HOST must be set}}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

HOSTFILE=${HOSTFILE:-/root/mpi_rack_hostfile}
NEEDED=$(( ACTOR_NUM_NODES + ROLLOUT_NUM_NODES ))
if [ -f "${HOSTFILE}" ]; then
  N=0
  for WORKER_IP in $(awk 'NF{print $1}' "${HOSTFILE}"); do
    if (( N >= NEEDED )); then break; fi
    N=$((N+1))
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then continue; fi
    ssh -o StrictHostKeyChecking=no root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; \
       ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
                 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/:${SLIME_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "SLIME_HEAD_HOST": "${SLIME_HEAD_HOST}",
    "E2B_API_KEY": "${E2B_API_KEY:-}",
    "SWE_HOST_NODE_TARBALL": "${SWE_HOST_NODE_TARBALL}",
    "SWE_HOST_CC_TARBALL": "${SWE_HOST_CC_TARBALL}",
    "SWE_TIME_BUDGET_SEC": "${SWE_TIME_BUDGET_SEC}",
    "SWE_EVAL_TIMEOUT_SEC": "${SWE_EVAL_TIMEOUT_SEC}",
    "SWE_SANDBOX_METADATA_JSON": "${SWE_SANDBOX_METADATA_JSON}",
    "SWE_TOOL_PARSER": "${SWE_TOOL_PARSER}",
    "SWE_REASONING_PARSER": "${SWE_REASONING_PARSER}",
    "SHIM_BIND_HOST": "${SHIM_BIND_HOST}",
    "SHIM_PORT": "${SHIM_PORT}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${EXTRA_ARGS[@]}
