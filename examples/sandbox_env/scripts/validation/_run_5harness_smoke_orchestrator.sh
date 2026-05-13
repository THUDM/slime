#!/bin/bash
# Sequentially runs the 35B 1-node smoke for each of the 5 canonical agent
# harnesses, in rollout-only mode. Outputs go to per-harness logs.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SANDBOX_ENV_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SANDBOX_ENV_DIR}/output/qwen3_5_35b_1node"
mkdir -p "${LOG_DIR}"

ORCH_LOG="${LOG_DIR}/orchestrator_5harness.log"
HARNESSES=(qwen_code claude_code codex open_code openhands)
# HARNESSES=(claude_code)

{
  echo "[$(date -u +%FT%TZ)] orchestrator start, pid=$$"
  echo "[$(date -u +%FT%TZ)] harnesses: ${HARNESSES[*]}"
} >>"${ORCH_LOG}"

for h in "${HARNESSES[@]}"; do
  log="${LOG_DIR}/qwen3.5-35b-a3b-1node-smoke-${h}.log"
  echo "[$(date -u +%FT%TZ)] starting harness=${h} log=${log}" >>"${ORCH_LOG}"

  SWE_AGENT_HARNESS="${h}" \
  SWE_DEBUG_ROLLOUT_ONLY=1 \
    bash "${SCRIPT_DIR}/../train/run_qwen3_5_35b_a3b_swe_inspire_1node_smoke.sh" \
    >"${log}" 2>&1
  rc=$?

  echo "[$(date -u +%FT%TZ)] finished harness=${h} exit=${rc}" >>"${ORCH_LOG}"
done

echo "[$(date -u +%FT%TZ)] orchestrator done" >>"${ORCH_LOG}"
