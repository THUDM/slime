#!/usr/bin/env bash
# DRY base for 4 SWE coding-agent RL scenarios (linear / subagent / compact /
# mixed). Sourced from each scenario's run_qwen36_<scenario>_2steps.sh which
# sets scenario-specific SWE_CLAUDE_EXTRA_ARGS / cc settings before invoking
# exec_train.
#
# See SPEC §8 for the 4 scenarios + acceptance gates.
#
# Required env (caller may override):
#   ACTOR_NUM_NODES                  default 1
#   ACTOR_NUM_GPUS_PER_NODE          default 8
#   HF_CHECKPOINT                    REQUIRED (e.g. /path/to/Qwen3-35B-A3B)
#   SGLANG_ROUTER_IP / SGLANG_ROUTER_PORT
#   SWE_HOST_NODE_TARBALL / SWE_HOST_CC_TARBALL
#   SLIME_HEAD_HOST                  REQUIRED for E2B
#   SWE_DATA_PATH                    REQUIRED dataset jsonl

set -euo pipefail

: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_NUM_GPUS_PER_NODE:=8}"
: "${HF_CHECKPOINT:?HF_CHECKPOINT must be set}"
: "${SGLANG_ROUTER_IP:?SGLANG_ROUTER_IP must be set}"
: "${SGLANG_ROUTER_PORT:?SGLANG_ROUTER_PORT must be set}"
: "${SWE_HOST_NODE_TARBALL:?SWE_HOST_NODE_TARBALL must be set}"
: "${SWE_HOST_CC_TARBALL:?SWE_HOST_CC_TARBALL must be set}"
: "${SLIME_HEAD_HOST:?SLIME_HEAD_HOST must be set}"
: "${SWE_DATA_PATH:?SWE_DATA_PATH must be set}"

: "${NUM_ROLLOUT:=1}"
: "${NUM_STEPS_PER_ROLLOUT:=2}"
: "${ROLLOUT_BATCH_SIZE:=2}"
: "${N_SAMPLES_PER_PROMPT:=4}"
: "${GLOBAL_BATCH_SIZE:=8}"

: "${SWE_TIME_BUDGET_SEC:=900}"
: "${SWE_EVAL_TIMEOUT_SEC:=600}"
: "${SWE_TOOL_PARSER:=glm47}"
: "${SWE_REASONING_PARSER:=glm45}"
: "${SWE_DUMP_RAW_TRAJECTORY:=0}"
: "${SWE_ABORT_POLL_INTERVAL:=0.5}"
: "${SWE_ABORT_MAX_WAIT_SEC:=1800}"

# RUNTIME_ENV_JSON keys forwarded to ray workers. Only env actually used by
# the new (0522) middleware/generate; legacy SWE_LIST_TRAJECTORY* removed.
RUNTIME_ENV_JSON=$(cat <<JSON
{
  "env_vars": {
    "SWE_HOST_NODE_TARBALL": "${SWE_HOST_NODE_TARBALL}",
    "SWE_HOST_CC_TARBALL":   "${SWE_HOST_CC_TARBALL}",
    "SWE_TIME_BUDGET_SEC":   "${SWE_TIME_BUDGET_SEC}",
    "SWE_EVAL_TIMEOUT_SEC":  "${SWE_EVAL_TIMEOUT_SEC}",
    "SWE_TOOL_PARSER":       "${SWE_TOOL_PARSER}",
    "SWE_REASONING_PARSER":  "${SWE_REASONING_PARSER}",
    "SWE_DUMP_RAW_TRAJECTORY": "${SWE_DUMP_RAW_TRAJECTORY}",
    "SWE_ABORT_POLL_INTERVAL": "${SWE_ABORT_POLL_INTERVAL}",
    "SWE_ABORT_MAX_WAIT_SEC":  "${SWE_ABORT_MAX_WAIT_SEC}",
    "SLIME_HEAD_HOST":       "${SLIME_HEAD_HOST}",
    "SWE_CLAUDE_EXTRA_ARGS": "${SWE_CLAUDE_EXTRA_ARGS:-}",
    "SWE_CC_PROMPT":         "${SWE_CC_PROMPT:-}"
  }
}
JSON
)

# Ray bootstrap (cluster mode; head + workers come up via slime helper if
# the host script handles it).
exec_train() {
  python -m slime.trainer.megatron_train \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --custom-generate-function-path examples.coding_agent_rl.generate.generate \
    --sglang-router-ip "${SGLANG_ROUTER_IP}" \
    --sglang-router-port "${SGLANG_ROUTER_PORT}" \
    --sglang-tool-call-parser "${SWE_TOOL_PARSER}" \
    --sglang-reasoning-parser "${SWE_REASONING_PARSER}" \
    --num-rollout "${NUM_ROLLOUT}" \
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --prompt-data "${SWE_DATA_PATH}" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    ${SWE_SEGMENT_REDUCER_PATH:+--swe-segment-reducer-path "${SWE_SEGMENT_REDUCER_PATH}"} \
    "$@"
}
