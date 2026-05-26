#!/usr/bin/env bash
# End-to-end SWE coding-agent RL smoke for Qwen3.6-35B-A3B on 8 nodes.
#
# This is the reusable version of the live cluster test. Run it from a long-lived
# shell/tmux session on the Ray head node; do not wrap it in a short-lived nohup
# launcher, otherwise Ray child processes may be cleaned up with the launcher.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
HOSTFILE="${HOSTFILE:-/root/mpi_rack_hostfile}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
SOCKET_IFNAME="${SOCKET_IFNAME:-eth5200}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/jingshenghang/storage/checkpoints/Qwen3.6-35B-A3B}"
REF_LOAD="${REF_LOAD:-/mnt/jingshenghang/storage/checkpoints/Qwen3.6-35B-A3B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/jingshenghang/code/slime_swe/datasets/swe-train-1545-localcache.jsonl}"
RUN_ROOT="${RUN_ROOT:-/mnt/jingshenghang/storage/slime_swe_runs/qwen36_cagent_$(date +%Y%m%d_%H%M%S)}"

export E2B_API_KEY="${E2B_API_KEY:-glm-platform}"
export SLIME_HEAD_HOST="${SLIME_HEAD_HOST:-172.27.14.209}"
export SWE_HOST_NODE_TARBALL="${SWE_HOST_NODE_TARBALL:-/mnt/jingshenghang/software/node-v22.20.0-linux-x64.tar.xz}"
export SWE_HOST_CC_TARBALL="${SWE_HOST_CC_TARBALL:-/mnt/jingshenghang/storage/claude_code/anthropic-ai-claude-code-2.1.143-local-linux-x64.tgz}"
export SWE_TIME_BUDGET_SEC="${SWE_TIME_BUDGET_SEC:-600}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-300}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-8}"
export SWE_SAVE_TRAJECTORY_TREE="${SWE_SAVE_TRAJECTORY_TREE:-0}"
export SWE_MAX_RESPONSE_TOKENS="${SWE_MAX_RESPONSE_TOKENS:-0}"
export SWE_TOOL_PARSER="${SWE_TOOL_PARSER:-qwen25}"
export SWE_REASONING_PARSER="${SWE_REASONING_PARSER:-qwen3}"
export SHIM_BIND_HOST="${SHIM_BIND_HOST:-0.0.0.0}"
export SHIM_PORT="${SHIM_PORT:-18001}"

mkdir -p "${RUN_ROOT}/rollout_dumps"

cd "${SLIME_DIR}"

INTERNAL_NO_PROXY="localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,100.64.0.0/10,172.16.0.0/12,${MASTER_ADDR},${SLIME_HEAD_HOST}"
export no_proxy="${no_proxy:+${no_proxy},}${INTERNAL_NO_PROXY}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${INTERNAL_NO_PROXY}"

if [[ "${SKIP_RAY_START:-0}" != "1" ]]; then
  ray stop --force || true
  pkill -9 -f '^ray::' || true
  pkill -9 -x sglang || true
  pkill -9 -x slime || true
  pkill -9 -x redis || true

  ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

  if [[ -f "${HOSTFILE}" ]]; then
    n=0
    while read -r worker_ip _ || [[ -n "${worker_ip:-}" ]]; do
      [[ -z "${worker_ip}" ]] && continue
      n=$((n + 1))
      (( n > ACTOR_NUM_NODES )) && break
      [[ "${worker_ip}" == "${MASTER_ADDR}" ]] && continue
      ssh -o StrictHostKeyChecking=no "root@${worker_ip}" \
        "ray stop --force || true; \
         pkill -9 -f '^ray::' || true; \
         pkill -9 -x sglang || true; \
         pkill -9 -x slime || true; \
         pkill -9 -x redis || true; \
         ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
           --node-ip-address ${worker_ip} --disable-usage-stats" &
    done < "${HOSTFILE}"
    wait
  fi
fi

RUNTIME_ENV_JSON="$(python3 - <<PY
import json, os
env = {
    "no_proxy": os.environ["no_proxy"],
    "NO_PROXY": os.environ["NO_PROXY"],
    "MASTER_ADDR": "${MASTER_ADDR}",
    "SLIME_HEAD_HOST": os.environ["SLIME_HEAD_HOST"],
    "E2B_API_KEY": os.environ["E2B_API_KEY"],
    "SWE_HOST_NODE_TARBALL": os.environ["SWE_HOST_NODE_TARBALL"],
    "SWE_HOST_CC_TARBALL": os.environ["SWE_HOST_CC_TARBALL"],
    "SWE_TIME_BUDGET_SEC": os.environ["SWE_TIME_BUDGET_SEC"],
    "SWE_EVAL_TIMEOUT_SEC": os.environ["SWE_EVAL_TIMEOUT_SEC"],
    "SWE_BOOT_CONCURRENCY": os.environ["SWE_BOOT_CONCURRENCY"],
    "SWE_SAVE_TRAJECTORY_TREE": os.environ["SWE_SAVE_TRAJECTORY_TREE"],
    "SWE_MAX_RESPONSE_TOKENS": os.environ["SWE_MAX_RESPONSE_TOKENS"],
    "SWE_TOOL_PARSER": os.environ["SWE_TOOL_PARSER"],
    "SWE_REASONING_PARSER": os.environ["SWE_REASONING_PARSER"],
    "SHIM_BIND_HOST": os.environ["SHIM_BIND_HOST"],
    "SHIM_PORT": os.environ["SHIM_PORT"],
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "PYTHONPATH": f"/root/Megatron-LM/:${SLIME_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
}
print(json.dumps({"env_vars": env}))
PY
)"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${REF_LOAD}" \
    --custom-generate-function-path examples.coding_agent_rl.generate.generate \
    --prompt-data "${PROMPT_DATA}" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --rollout-shuffle \
    --num-rollout "${NUM_ROLLOUT:-1}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-2}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}" \
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-200000}" \
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-32768}" \
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}" \
    --rollout-stop-token-ids 248046 248044 \
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-1}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE:-16}" \
    --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
    --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt" \
    --advantage-estimator gspo \
    --kl-loss-coef 0.00 \
    --kl-loss-type low_var_kl \
    --kl-coef 0.00 \
    --entropy-coef 0.00 \
    --eps-clip 1e-4 \
    --eps-clip-high 2e-4 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --optimizer-cpu-offload \
    --overlap-cpu-optimizer-d2h-h2d \
    --use-precision-aware-optimizer \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 2 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}" \
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-64}" \
    --rollout-num-gpus 64 \
    --rollout-num-gpus-per-engine 8 \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.75}" \
    --sglang-enable-dp-attention \
    --sglang-dp-size 8 \
    --sglang-ep-size 8 \
    --sglang-enable-dp-lm-head \
    --sglang-moe-dense-tp-size 1 \
    --sglang-tool-call-parser qwen25 \
    --sglang-reasoning-parser qwen3 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --colocate

python3 - <<PY
from pathlib import Path
import torch

run_root = Path("${RUN_ROOT}")
dump_path = run_root / "rollout_dumps" / "rollout_0.pt"
want_tree = "${SWE_SAVE_TRAJECTORY_TREE}" == "1"

if not dump_path.exists():
    raise SystemExit(f"missing rollout dump: {dump_path}")

payload = torch.load(dump_path, map_location="cpu")
samples = payload.get("samples") or []
if not samples:
    raise SystemExit(f"empty rollout dump: {dump_path}")

completed = [s for s in samples if str(s.get("status")).lower() == "completed"]
if not completed:
    raise SystemExit(f"no completed samples in {dump_path}")

bad_lengths = [i for i, s in enumerate(samples) if s.get("response_length") != len(s.get("loss_mask") or [])]
if bad_lengths:
    raise SystemExit(f"response_length/loss_mask mismatch at samples {bad_lengths[:5]} in {dump_path}")

trees = [(s.get("metadata") or {}).get("trajectory_tree") for s in samples]
non_empty_trees = [t for t in trees if t and t.get("turns")]
if want_tree and not non_empty_trees:
    raise SystemExit(f"SWE_SAVE_TRAJECTORY_TREE=1 but no non-empty trajectory_tree in {dump_path}")
if not want_tree and any(trees):
    raise SystemExit(f"SWE_SAVE_TRAJECTORY_TREE=0 but trajectory_tree metadata exists in {dump_path}")

print(
    "validated rollout dump:",
    dump_path,
    "samples=", len(samples),
    "completed=", len(completed),
    "max_tokens=", max(len(s.get("tokens") or []) for s in samples),
    "max_response=", max(s.get("response_length") or 0 for s in samples),
    "tree=", bool(non_empty_trees),
)
PY
