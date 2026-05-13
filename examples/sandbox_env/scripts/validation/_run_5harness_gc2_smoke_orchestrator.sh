#!/bin/bash
# 5-harness G_C2 spec validation smoke (Qwen3.5-0.8B, single GPU per harness).
#
# For each of the 5 canonical agent harnesses, runs ROLLOUT-only with batch
# size 3 against three re-spec'd G_C2 templates:
#   - 0xs34n__starknet.js-508
#   - 0xs34n__starknet.js-538
#   - 1c-syntax__bsl-language-server-3207
#
# Validates that the 2c/8GB G_C2 spec is sufficient for the rollout pipeline
# (sandbox boot + wstunnel tunnel + agent + repo test execution).
#
# Inputs (must already exist):
#   data_output/swe_rebench_tasks_smoke_gc2_3rows.json     ← 3-row tasks-json subset
#   data_output/swe_rebench_scaffold_template_smoke_gc2.jsonl  ← 3-row manifest with G_C2 aliases
#   templates `swe-rebench-gc2-probe-{slug}` already built (see /tmp/probe_respec_template.py)
#
# Outputs:
#   output/qwen3_5_0_8b_smoke_gc2/orchestrator_5harness_gc2.log
#   output/qwen3_5_0_8b_smoke_gc2/qwen3.5-0.8b-smoke-gc2-{harness}.log
#   output/qwen3_5_0_8b_smoke_gc2/smoke-gc2-{harness}/...  (per-run logs/checkpoints)
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SANDBOX_ENV_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_OUT="${SANDBOX_ENV_DIR}/data_output"

GC2_TASKS_JSON="${DATA_OUT}/swe_rebench_tasks_smoke_gc2_3rows.json"
GC2_MANIFEST="${DATA_OUT}/swe_rebench_scaffold_template_smoke_gc2.jsonl"

if [[ ! -s "${GC2_TASKS_JSON}" ]]; then
  echo "missing tasks-json subset: ${GC2_TASKS_JSON}" >&2; exit 1
fi
if [[ ! -s "${GC2_MANIFEST}" ]]; then
  echo "missing G_C2 consumable manifest: ${GC2_MANIFEST}" >&2; exit 1
fi

LOG_DIR="${SANDBOX_ENV_DIR}/output/qwen3_5_0_8b_smoke_gc2"
mkdir -p "${LOG_DIR}"

ORCH_LOG="${LOG_DIR}/orchestrator_5harness_gc2.log"
HARNESSES=(qwen_code claude_code codex open_code openhands)

{
  echo "[$(date -u +%FT%TZ)] orchestrator start, pid=$$"
  echo "[$(date -u +%FT%TZ)] tasks_json: ${GC2_TASKS_JSON}"
  echo "[$(date -u +%FT%TZ)] manifest:   ${GC2_MANIFEST}"
  echo "[$(date -u +%FT%TZ)] harnesses:  ${HARNESSES[*]}"
} >>"${ORCH_LOG}"

for h in "${HARNESSES[@]}"; do
  log="${LOG_DIR}/qwen3.5-0.8b-smoke-gc2-${h}.log"
  echo "[$(date -u +%FT%TZ)] starting harness=${h} log=${log}" >>"${ORCH_LOG}"

  REBENCH_TASKS_JSON="${GC2_TASKS_JSON}" \
  SWE_CONSUMABLE_TEMPLATE_MANIFEST="${GC2_MANIFEST}" \
  WORK_ROOT_BASE="${LOG_DIR}" \
  SWE_AGENT_HARNESS="${h}" \
  SWE_DEBUG_MAX_TRAIN_PER_SOURCE=3 \
  SWE_DEBUG_MAX_VAL_PER_SOURCE=0 \
  SWE_NUM_ROLLOUT=1 \
  SWE_ROLLOUT_BATCH_SIZE=3 \
  SWE_SAMPLES_PER_PROMPT=1 \
  SWE_GLOBAL_BATCH_SIZE=3 \
  SWE_STEPS_PER_ROLLOUT=1 \
  SWE_DEBUG_ROLLOUT_ONLY=1 \
  SWE_SAMPLE_CONCURRENCY=3 \
  SWE_WANDB_GROUP="smoke-gc2" \
  SWE_WANDB_RUN_ID="qwen3.5-0.8b-smoke-gc2-${h}" \
    bash "${SCRIPT_DIR}/../train/run_qwen3_5_0_8b_swe_inspire_agentic_protocol_smoke.sh" \
    >"${log}" 2>&1
  rc=$?

  echo "[$(date -u +%FT%TZ)] finished harness=${h} exit=${rc}" >>"${ORCH_LOG}"
done

echo "[$(date -u +%FT%TZ)] orchestrator done" >>"${ORCH_LOG}"
