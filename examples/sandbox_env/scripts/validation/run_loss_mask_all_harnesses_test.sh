#!/bin/bash
# Sequentially smoke-tests each of the five supported agent harnesses
# (qwen_code, claude_code, codex, open_code, openhands) against the proxy
# step_loss_mask fix and verifies the resulting sample_artifacts.json:
#   - assistant messages produced by the proxy itself  -> step_loss_mask = 1
#   - assistant messages injected by the harness CLI   -> step_loss_mask = 0
#
# Each harness reuses the existing smoke script with a shared all-harnesses
# manifest; outputs land in sibling directories under
# output/qwen3_5_0_8b_smoke_agentic_protocol/loss-mask-all-harnesses-<harness>/.
#
# Override HARNESSES to test a subset, e.g.
#   HARNESSES="qwen_code codex" bash run_loss_mask_all_harnesses_test.sh
#
# Continues past individual failures and prints a summary at the end. Exits 0
# only when every harness ran AND verified successfully.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SANDBOX_ENV_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_HARNESSES=(qwen_code claude_code codex open_code openhands)
read -r -a HARNESSES_ARR <<<"${HARNESSES:-${DEFAULT_HARNESSES[*]}}"

TEST_GROUP="loss-mask-all-harnesses"
ALL_HARNESSES_MANIFEST="${SANDBOX_ENV_DIR}/data_output/swe_rebench_scaffold_template_success.jsonl"
SMOKE_SCRIPT="${SCRIPT_DIR}/../train/run_qwen3_5_0_8b_swe_inspire_agentic_protocol_smoke.sh"
VERIFY_SCRIPT="${SANDBOX_ENV_DIR}/tools/verify_step_loss_mask.py"

if [[ ! -s "${ALL_HARNESSES_MANIFEST}" ]]; then
  echo "all-harnesses manifest missing: ${ALL_HARNESSES_MANIFEST}" >&2
  exit 1
fi
if [[ ! -x "${SMOKE_SCRIPT}" ]] && [[ ! -f "${SMOKE_SCRIPT}" ]]; then
  echo "smoke script missing: ${SMOKE_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${VERIFY_SCRIPT}" ]]; then
  echo "verifier missing: ${VERIFY_SCRIPT}" >&2
  exit 1
fi

OUTPUT_BASE="${SANDBOX_ENV_DIR}/output/qwen3_5_0_8b_smoke_agentic_protocol"
SUMMARY_LOG="${OUTPUT_BASE}/${TEST_GROUP}.summary.log"
mkdir -p "${OUTPUT_BASE}"

# Shared across all child smoke invocations.
export SWE_CONSUMABLE_TEMPLATE_MANIFEST="${ALL_HARNESSES_MANIFEST}"
export SWE_WANDB_GROUP="${TEST_GROUP}"

declare -A RUN_STATUS=()
declare -A VERIFY_STATUS=()

run_one_harness() {
  local harness="$1"
  echo
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] Running harness: ${harness}"
  echo "=================================================================="

  if SWE_AGENT_HARNESS="${harness}" bash "${SMOKE_SCRIPT}"; then
    RUN_STATUS["${harness}"]="ok"
  else
    RUN_STATUS["${harness}"]="failed"
    return 0
  fi

  local artifact="${OUTPUT_BASE}/${TEST_GROUP}-${harness}/logs/current_batch/sample_0/sample_artifacts.json"
  if [[ ! -f "${artifact}" ]]; then
    VERIFY_STATUS["${harness}"]="missing-artifact"
    return 0
  fi

  if python3 "${VERIFY_SCRIPT}" "${artifact}" "${harness}"; then
    VERIFY_STATUS["${harness}"]="ok"
  else
    VERIFY_STATUS["${harness}"]="mismatch"
  fi
}

for h in "${HARNESSES_ARR[@]}"; do
  run_one_harness "${h}"
done

# ── Summary ───────────────────────────────────────────────────────────────────
{
  echo
  echo "=================================================================="
  echo "Test summary"
  echo "=================================================================="
  all_ok=1
  for h in "${HARNESSES_ARR[@]}"; do
    rs="${RUN_STATUS[${h}]:-not-run}"
    vs="${VERIFY_STATUS[${h}]:-not-verified}"
    if [[ "${rs}" == "ok" && "${vs}" == "ok" ]]; then
      printf "  %-12s OK   run=%s verify=%s\n" "${h}" "${rs}" "${vs}"
    else
      all_ok=0
      printf "  %-12s FAIL run=%s verify=%s\n" "${h}" "${rs}" "${vs}"
    fi
  done
  if (( all_ok )); then
    echo
    echo "All harnesses passed."
  else
    echo
    echo "One or more harnesses failed (see logs under ${OUTPUT_BASE}/${TEST_GROUP}-*/logs/run.log)."
  fi
} | tee "${SUMMARY_LOG}"

# Exit code reflects overall pass/fail.
for h in "${HARNESSES_ARR[@]}"; do
  if [[ "${RUN_STATUS[${h}]:-}" != "ok" ]] || [[ "${VERIFY_STATUS[${h}]:-}" != "ok" ]]; then
    exit 1
  fi
done
exit 0
