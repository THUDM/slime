#!/usr/bin/env bash
# Batch re-spec all G_C4 templates in the success_manifest to G_C2.
#
# Source:  swe_rebench_scaffold_template_success.aux_workers20_20260506T1300Z.jsonl
# Output:  swe_rebench_scaffold_template_success.respec_gc2.jsonl  (new G_C2 aliases)
#          swe_rebench_scaffold_template_failure.respec_gc2.jsonl  (any failures)
#
# Resumable: re-running picks up where it left off (skips rows already in
# success_manifest; alias state is also checked against Inspire as a safety net).
#
# Expected wallclock: ~8150 rows × ~13s / workers ≈ 10h at workers=3.
set -euo pipefail

LOGIN_SH="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/jy_workspace/login.sh"
WORKDIR="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/public/avalanche/jy_workspace"
DATA_OUT="${WORKDIR}/slime/examples/sandbox_env/data_output"

SOURCE_MANIFEST="${SOURCE_MANIFEST:-${DATA_OUT}/swe_rebench_scaffold_template_all_clean.jsonl}"
SUCCESS_MANIFEST="${SUCCESS_MANIFEST:-${DATA_OUT}/swe_rebench_scaffold_template_success.respec_gc2_all_clean.jsonl}"
FAILURE_MANIFEST="${FAILURE_MANIFEST:-${DATA_OUT}/swe_rebench_scaffold_template_failure.respec_gc2_all_clean.jsonl}"
WORKERS="${WORKERS:-6}"

LOG_PATH="${DATA_OUT}/respec_gc2_$(date -u '+%Y%m%dT%H%MZ').log"

source "${LOGIN_SH}" >/dev/null
cd "${WORKDIR}"

/usr/bin/date -u '+%Y-%m-%dT%H:%M:%SZ respec_gc2 batch start'
echo "source=${SOURCE_MANIFEST}"
echo "success=${SUCCESS_MANIFEST}"
echo "failure=${FAILURE_MANIFEST}"
echo "workers=${WORKERS}"
echo "log=${LOG_PATH}"

/usr/bin/python3 "${WORKDIR}/slime/examples/sandbox_env/tools/respec_templates_to_gc2.py" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --success-manifest "${SUCCESS_MANIFEST}" \
  --failure-manifest "${FAILURE_MANIFEST}" \
  --workers "${WORKERS}" 2>&1 | tee -a "${LOG_PATH}"

/usr/bin/date -u '+%Y-%m-%dT%H:%M:%SZ respec_gc2 batch done'
