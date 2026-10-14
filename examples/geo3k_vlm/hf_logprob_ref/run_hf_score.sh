#!/bin/bash
# Score the rollout dumps under <dump_dir> with HF transformers, on one node.
#   HF_MODEL_PATH=/path/to/NVIDIA-Nemotron-3.5-Super-EA-09112026 ./run_hf_score.sh <dump_dir> [hf_score.py flags]
# <dump_dir> is the --dump-details directory of a slime run (it holds rollout_data/{eval_0,0}.pt).
# Writes <dump_dir>/hf_score.pt. The model is sharded over every visible GPU (device_map=auto).
#
# Environment: transformers >= 5.8 and torch, with mamba-ssm and causal-conv1d NOT installed.
#   * The reference side has to run transformers' own torch SSM path. If the fused kernels
#     (mamba-ssm, causal-conv1d) are installed, transformers switches to them and hf_score.py
#     exits when it detects that.
#   * The slime image itself ships without these two packages, so it can be used as is
#     (transformers 5.12.1 there works): start a fresh container from it for this step.
#   * Do not run this in a container or shell where the training has already been set up:
#     run_geo3k_nemotron35.sh begins with `pip install ... causal-conv1d mamba-ssm`, and so
#     does any setup that installs prebuilt wheels of both (e.g. a per-node training entry
#     script). Once either has run, the fused kernels are present and scoring exits.
set -euo pipefail
DUMP=$(realpath "$1"); shift
: "${HF_MODEL_PATH:?set HF_MODEL_PATH to the HF checkpoint directory}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DATA=()
for f in eval_0 0; do
  [ -f "$DUMP/rollout_data/$f.pt" ] && DATA+=(--rollout-data "$DUMP/rollout_data/$f.pt")
done
[ ${#DATA[@]} -gt 0 ] || { echo "no rollout_data/{eval_0,0}.pt under $DUMP" >&2; exit 1; }
python3 "$HERE/hf_score.py" "${DATA[@]}" --model-path "$HF_MODEL_PATH" --output "$DUMP/hf_score.pt" "$@" \
  2>&1 | tee "$DUMP/hf_score.log"
