#!/bin/bash
#SBATCH --job-name=slime-qwen3-data25k10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
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

mkdir -p "${SLURM_LOG_DIR}" "${DATA_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export RAY_TMPDIR="/tmp/${USER:-suryadv}/slime-ray-${SLURM_JOB_ID}"
  mkdir -p "${RAY_TMPDIR}"
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Preparing OpenThoughts3 math data at $(date)"
echo "SFT rows: ${SFT_SIZE} -> ${SFT_PARQUET}"
echo "OPD rows: ${OPD_SIZE} -> ${OPD_JSONL}"
echo "Metadata: ${SPLIT_METADATA}"

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"

needs_prepare=0
for path in "${SFT_PARQUET}" "${OPD_JSONL}" "${SPLIT_METADATA}"; do
  if [[ ! -s "${path}" ]]; then
    needs_prepare=1
  fi
done

if [[ "${needs_prepare}" == "0" ]]; then
  if ! python3 - <<PY
import json
import os
from pathlib import Path

sft_path = Path(os.environ["SFT_PARQUET"])
opd_path = Path(os.environ["OPD_JSONL"])
metadata_path = Path(os.environ["SPLIT_METADATA"])
sft_expected = int(os.environ["SFT_SIZE"])
opd_expected = int(os.environ["OPD_SIZE"])

def count_jsonl(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
sft_count = count_jsonl(sft_path)
opd_count = count_jsonl(opd_path)
sft_ids = set(metadata["source_row_ids"]["sft"])
opd_ids = set(metadata["source_row_ids"]["opd"])

if sft_count != sft_expected or opd_count != opd_expected or sft_ids.intersection(opd_ids):
    raise SystemExit(1)
PY
  then
    echo "Existing data files do not match the requested split; regenerating."
    needs_prepare=1
  fi
fi

if [[ "${needs_prepare}" == "1" ]]; then
  python3 examples/qwen3_8b_opd_tillicum/02_prepare_openthoughts3_math_sample.py --force
else
  echo "Data files already exist; validating counts."
fi

python3 - <<PY
import json
import os
from pathlib import Path

sft_path = Path(os.environ["SFT_PARQUET"])
opd_path = Path(os.environ["OPD_JSONL"])
metadata_path = Path(os.environ["SPLIT_METADATA"])
sft_expected = int(os.environ["SFT_SIZE"])
opd_expected = int(os.environ["OPD_SIZE"])

def count_jsonl(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)

sft_count = count_jsonl(sft_path)
opd_count = count_jsonl(opd_path)
if sft_count != sft_expected:
    raise SystemExit(f"SFT row count mismatch: {sft_count} != {sft_expected}")
if opd_count != opd_expected:
    raise SystemExit(f"OPD row count mismatch: {opd_count} != {opd_expected}")

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
sft_ids = set(metadata["source_row_ids"]["sft"])
opd_ids = set(metadata["source_row_ids"]["opd"])
overlap = sft_ids.intersection(opd_ids)
if overlap:
    raise SystemExit(f"SFT/OPD metadata overlap detected: {sorted(overlap)[:10]}")

print(f"DATA_PREP_OK sft_rows={sft_count} opd_rows={opd_count} metadata={metadata_path}")
PY
'

echo "Finished data prep at $(date)"
