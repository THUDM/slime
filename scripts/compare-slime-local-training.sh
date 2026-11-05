#!/usr/bin/env bash
# Run the same train-only slime workload with several micro-batch sizes.
#
# This is intentionally a thin comparison wrapper around
# run-slime-training-benchmark.sh.  It keeps the model, token data, optimizer,
# rollout count, and global batch fixed while changing only
# MICRO_BATCH_SIZE.  Each run gets an isolated result directory and the final
# comparison.json contains the numeric perf summary for easy ingestion by the
# Weaver benchmark report.
#
# Required environment variables are the same as the underlying script:
#   HF_CHECKPOINT, TOKEN_DATA, MODEL_CONFIG
#
# Example:
#   HF_CHECKPOINT=/models/Qwen2.5-0.5B \
#   TOKEN_DATA=/datasets/wildchat_tokens.jsonl \
#   MODEL_CONFIG=scripts/models/qwen2.5-0.5B.sh \
#   GLOBAL_BATCH_SIZE=4 ROLLOUT_BATCH_SIZE=4 \
#   MICRO_BATCH_SIZES="1 2 4" \
#   scripts/compare-slime-local-training.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run-slime-training-benchmark.sh"

: "${HF_CHECKPOINT:?set HF_CHECKPOINT to the HuggingFace checkpoint}"
: "${TOKEN_DATA:?set TOKEN_DATA to a token JSONL file}"
: "${MODEL_CONFIG:?set MODEL_CONFIG to a file defining MODEL_ARGS=(...)}"

MICRO_BATCH_SIZES="${MICRO_BATCH_SIZES:-1 2}"
COMPARISON_DIR="${COMPARISON_DIR:-${REPO_DIR}/benchmark-results/slime-compare-$(date +%Y%m%d-%H%M%S)}"
START_RAY_FIRST="${START_RAY_FIRST:-1}"

[[ -x "${RUN_SCRIPT}" ]] || {
    echo "benchmark script is not executable: ${RUN_SCRIPT}" >&2
    exit 2
}

if [[ -e "${COMPARISON_DIR}" ]]; then
    echo "COMPARISON_DIR already exists; choose a fresh directory: ${COMPARISON_DIR}" >&2
    exit 2
fi
mkdir -p "${COMPARISON_DIR}"
REQUESTED_RUNS_FILE="${COMPARISON_DIR}/requested-runs.txt"
: >"${REQUESTED_RUNS_FILE}"

# Validate the list before starting any remote job.  This avoids a partially
# populated comparison directory when a typo is passed in the environment.
read -r -a micro_batch_sizes <<<"${MICRO_BATCH_SIZES}"
(( ${#micro_batch_sizes[@]} > 0 )) || {
    echo "MICRO_BATCH_SIZES must contain at least one positive integer" >&2
    exit 2
}
for micro_batch_size in "${micro_batch_sizes[@]}"; do
    [[ "${micro_batch_size}" =~ ^[1-9][0-9]*$ ]] || {
        echo "invalid micro-batch size: ${micro_batch_size}" >&2
        exit 2
    }
done

run_index=0
for micro_batch_size in "${micro_batch_sizes[@]}"; do
    run_dir="${COMPARISON_DIR}/micro-batch-${micro_batch_size}"
    echo "=== slime train-only run: MICRO_BATCH_SIZE=${micro_batch_size} ===" >&2
    START_RAY="${START_RAY_FIRST}"
    if (( run_index > 0 )); then
        START_RAY=0
    fi

    case " ${seen_sizes:-} " in
        *" ${micro_batch_size} "*)
            echo "duplicate micro-batch size: ${micro_batch_size}" >&2
            exit 2
            ;;
    esac
    seen_sizes="${seen_sizes:-} ${micro_batch_size}"
    printf '%s\n' "${run_dir}" >>"${REQUESTED_RUNS_FILE}"

    HF_CHECKPOINT="${HF_CHECKPOINT}" \
    TOKEN_DATA="${TOKEN_DATA}" \
    MODEL_CONFIG="${MODEL_CONFIG}" \
    MICRO_BATCH_SIZE="${micro_batch_size}" \
    RESULT_DIR="${run_dir}" \
    START_RAY="${START_RAY}" \
    "${RUN_SCRIPT}"
    run_index=$((run_index + 1))
done

COMPARISON_DIR="${COMPARISON_DIR}" REQUESTED_RUNS_FILE="${REQUESTED_RUNS_FILE}" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["COMPARISON_DIR"])
rows = []
requested_runs = [Path(line) for line in Path(os.environ["REQUESTED_RUNS_FILE"]).read_text().splitlines() if line]
for run_dir in requested_runs:
    summary_path = run_dir / "perf-summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing perf summary for requested run: {run_dir}")
    config_path = run_dir / "config.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    rows.append(
        {
            "micro_batch_size": config.get("micro_batch_size", run_dir.name.removeprefix("micro-batch-")),
            "gradient_accumulation_per_rank": config.get("gradient_accumulation_per_rank"),
            "perf": {
                key: summary[key]
                for key in (
                    "perf/actor_train_time",
                    "perf/train_time",
                    "perf/step_time",
                    "perf/actor_train_tok_per_s",
                )
                if key in summary
            },
            "result_dir": str(run_dir),
        }
    )

output = {
    "comparison_dir": str(root),
    "runs": rows,
}
(root / "comparison.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print("micro_batch_size\tgradient_accumulation\tactor_train_time\ttrain_time\tstep_time\tactor_train_tok_per_s")
for row in rows:
    perf = row["perf"]
    values = [
        row["micro_batch_size"],
        row["gradient_accumulation_per_rank"],
        perf.get("perf/actor_train_time", ""),
        perf.get("perf/train_time", ""),
        perf.get("perf/step_time", ""),
        perf.get("perf/actor_train_tok_per_s", ""),
    ]
    print("\t".join("" if value is None else str(value) for value in values))
PY

echo "comparison written to ${COMPARISON_DIR}/comparison.json" >&2
