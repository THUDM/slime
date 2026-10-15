#!/usr/bin/env bash
# Launch on an existing six-node Ray cluster (3 training + 3 inference).
# Sandbox nodes are configured separately in SUNABAKO_CLUSTER; see README.md.
set -euo pipefail

SLIME_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
HF_CHECKPOINT=${HF_CHECKPOINT:?Set HF_CHECKPOINT to Qwen3.8-27B}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the prepared MiMo data directory}
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to a fresh shared experiment directory}
LOG_ROOT=${LOG_ROOT:-$RUN_ROOT}
export RAY_ADDRESS=${RAY_ADDRESS:?Set RAY_ADDRESS to the existing Ray cluster}
RAY_DASHBOARD_ADDRESS=${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the Ray dashboard URL}
PROMPT_DATA=${PROMPT_DATA:-$DATA_ROOT/train-mixed.jsonl}
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * SAMPLES_PER_PROMPT))
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-3}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-24}
ROLLOUT_GPUS_PER_ENGINE=${ROLLOUT_GPUS_PER_ENGINE:-2}
SERVER_CONCURRENCY=${SERVER_CONCURRENCY:-14}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-131072}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
FLUSH_CACHE_INTERVAL=${FLUSH_CACHE_INTERVAL:-3}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLIME_DIR:${MEGATRON_DIR:?Set MEGATRON_DIR to your Megatron-LM checkout}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GOMAXPROCS=${GOMAXPROCS:-2}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME:?Set SOCKET_IFNAME to the inter-node network interface}
export NCCL_SOCKET_IFNAME=$SOCKET_IFNAME TP_SOCKET_IFNAME=$SOCKET_IFNAME
export SWE_SANDBOX_PROVIDER=sunabako SWE_AGENT=${SWE_AGENT:-claude_code} SWE_TRAIN_PROTOCOL=scaleswe
export SWE_TRACE_DIR=${SWE_TRACE_DIR-$LOG_ROOT/traces}
export SUNABAKO_CLUSTER=${SUNABAKO_CLUSTER:?Set SUNABAKO_CLUSTER to the two sandbox nodes configuration}
export SUNABAKO_IMAGES=${SUNABAKO_IMAGES:-$DATA_ROOT/images.json}
export SUNABAKO_ARTIFACTS="$LOG_ROOT/sandboxes"
export SUNABAKO_MEMORY_MB=${SUNABAKO_MEMORY_MB:-8192}
export SUNABAKO_PIDS_MAX=${SUNABAKO_PIDS_MAX:-1024}
export SUNABAKO_ALLOW_TEST_MEMORY=${SUNABAKO_ALLOW_TEST_MEMORY:-0}
export SUNABAKO_DEFER_ROOTFS_DELETE=${SUNABAKO_DEFER_ROOTFS_DELETE:-1}
export SWE_BOOT_CONCURRENCY=${SWE_BOOT_CONCURRENCY:-0}
export SWE_AGENT_TIME_BUDGET_SEC=${SWE_AGENT_TIME_BUDGET_SEC:-3600}
export SWE_EVAL_TIMEOUT_SEC=${SWE_EVAL_TIMEOUT_SEC:-600}
export SWE_ROLLOUT_GUARD_SEC=${SWE_ROLLOUT_GUARD_SEC:-7200}
export SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS:-0}
export SLIME_AGENT_CC_MAX_COMPACTIONS=${SLIME_AGENT_CC_MAX_COMPACTIONS:-2}
export SLIME_AGENT_CC_COMPACT_PCT=${SLIME_AGENT_CC_COMPACT_PCT:-70}
export SLIME_AGENT_CODEX_MAX_COMPACTIONS=${SLIME_AGENT_CODEX_MAX_COMPACTIONS:-2}
export SLIME_AGENT_CODEX_CONTEXT_WINDOW=${SLIME_AGENT_CODEX_CONTEXT_WINDOW:-$MAX_CONTEXT_LEN}
export SLIME_AGENT_CODEX_COMPACT_TOKEN_LIMIT=${SLIME_AGENT_CODEX_COMPACT_TOKEN_LIMIT:-$((MAX_CONTEXT_LEN * 70 / 100))}
export ADAPTER_BIND_HOST=0.0.0.0 ADAPTER_PUBLIC_HOST=${ADAPTER_PUBLIC_HOST:-auto}
# Each distributed rollout worker can bind independently, including after recovery.
export ADAPTER_PORT=${ADAPTER_PORT:-0}
export SLIME_AGENT_CC_EXTRA_ARGS=${SLIME_AGENT_CC_EXTRA_ARGS:-'--tools Bash,Read,Edit,Write,Glob,Grep --disable-slash-commands'}
export SWE_CC_PROMPT=${SWE_CC_PROMPT:-'Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. Edit source files only; do not modify tests or PROBLEM_STATEMENT.md. The .harness directory contains your own live execution logs, so exclude it from searches. Implement the fix, run the relevant tests, then print a one-line summary and exit. Do not commit.'}
export MAX_CONTEXT_LEN MAX_RESPONSE_LEN
SLIME_AGENT_CC_EXTRA_ENVS=$(python - <<'PY'
import json, os
settings = {
    "MAX_THINKING_TOKENS": "4096",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": os.environ["MAX_RESPONSE_LEN"],
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": os.environ["MAX_CONTEXT_LEN"],
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": os.environ["SLIME_AGENT_CC_COMPACT_PCT"],
}
settings.update(json.loads(os.environ.get("SLIME_AGENT_CC_EXTRA_ENVS", "{}")))
print(json.dumps(settings))
PY
)
export SLIME_AGENT_CC_EXTRA_ENVS

test -s "$PROMPT_DATA"
AGENTS=$(python - "$PROMPT_DATA" <<'PY'
import json, os, sys
with open(sys.argv[1]) as stream:
    agents = {json.loads(line).get("metadata", {}).get("agent", os.environ["SWE_AGENT"]) for line in stream if line.strip()}
if not agents or agents - {"codex", "claude_code"}:
    raise ValueError(f"Invalid coding-agent selection: {agents}")
print(" ".join(sorted(agents)))
PY
)
for agent in $AGENTS; do
    if [[ $agent == claude_code ]]; then
        if [[ -n ${SLIME_AGENT_CC_NATIVE_TARBALL:-} ]]; then
            test -s "$SLIME_AGENT_CC_NATIVE_TARBALL"
        else
            test -s "${SLIME_AGENT_NODE_TARBALL:?Set the Node runtime archive or use a native CLI archive}"
            test -s "${SLIME_AGENT_CC_TARBALL:?Set the Claude Code npm archive or use a native CLI archive}"
        fi
    else
        if [[ -n ${SLIME_AGENT_CODEX_NATIVE_TARBALL:-} ]]; then
            test -s "$SLIME_AGENT_CODEX_NATIVE_TARBALL"
        else
            test -s "${SLIME_AGENT_NODE_TARBALL:?Set the Node runtime archive or use a native CLI archive}"
            test -s "${SLIME_AGENT_CODEX_TARBALL:?Set the Codex npm archive or use a native CLI archive}"
        fi
    fi
done
test -s "$SUNABAKO_CLUSTER"
test -s "$SUNABAKO_IMAGES"
if [[ -e "$LOG_ROOT/run.log" ]]; then
    echo 'LOG_ROOT/run.log already exists; choose a new attempt log directory.' >&2
    exit 1
fi
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

source "$SLIME_DIR/scripts/models/qwen3.5-27B.sh"
SGLANG_CAPACITY_ARGS=()
if [[ -n ${MAX_RUNNING_REQUESTS:-} ]]; then
    SGLANG_CAPACITY_ARGS+=(--sglang-max-running-requests "$MAX_RUNNING_REQUESTS")
fi
if [[ ${ENABLE_HICACHE:-1} == 1 ]]; then
    SGLANG_CAPACITY_ARGS+=(--sglang-enable-hierarchical-cache --sglang-hicache-ratio "${HICACHE_RATIO:-2}"
        --sglang-hicache-mem-layout page_first_direct --sglang-hicache-io-backend direct)
fi
TRACKING_ARGS=()
CHECKPOINT_ARGS=()
if (( SAVE_INTERVAL > 0 )); then
    CHECKPOINT_ARGS=(--save "$RUN_ROOT/checkpoints" --save-interval "$SAVE_INTERVAL")
fi
if [[ -n ${WANDB_PROJECT:-} ]]; then
    TRACKING_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT"
        --wandb-group "${WANDB_NAME:-qwen-agent-rl}" --disable-wandb-random-suffix --wandb-dir "$LOG_ROOT")
    if [[ -n ${WANDB_ENTITY:-} ]]; then
        TRACKING_ARGS+=(--wandb-team "$WANDB_ENTITY")
    fi
fi
STORAGE_ARGS=(--rollout-data-transport "${ROLLOUT_DATA_TRANSPORT:-straw}")
if [[ ${ROLLOUT_DATA_TRANSPORT:-straw} == straw ]]; then
    STORAGE_ARGS+=(--rollout-data-dir "$RUN_ROOT/rollout_data" --rollout-queue-online-gc
        --rollout-storage-profile "${ROLLOUT_STORAGE_PROFILE:-local}")
    if [[ -n ${ROLLOUT_STORAGE_DECLARATION:-} ]]; then
        STORAGE_ARGS+=(--rollout-storage-declaration "$ROLLOUT_STORAGE_DECLARATION")
    fi
fi

# Runtime credentials belong in a private file, never in argv or shell tracing.
umask 077
RUNTIME_ENV=$(mktemp "$RUN_ROOT/runtime-env.XXXXXX.json")
trap 'rm -f -- "$RUNTIME_ENV"' EXIT
python - "$RUNTIME_ENV" <<'PY'
import json, os, sys
keys = {
    "PYTHONPATH", "PYTHONUNBUFFERED", "CUDA_DEVICE_MAX_CONNECTIONS",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "GOMAXPROCS", "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
    "NCCL_NVLS_ENABLE", "SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS",
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY",
}
keys.update(k for k in os.environ if k.startswith(("SUNABAKO_", "SWE_", "SLIME_AGENT_", "ADAPTER_", "WANDB_")))
with open(sys.argv[1], "w") as stream:
    json.dump({"env_vars": {k: os.environ[k] for k in keys if k in os.environ}}, stream)
PY

cd "$SLIME_DIR"
ray job submit --address="$RAY_DASHBOARD_ADDRESS" --runtime-env="$RUNTIME_ENV" -- \
    python -u "$SLIME_DIR/train.py" \
    --actor-num-nodes "$ACTOR_NUM_NODES" --actor-num-gpus-per-node "$GPUS_PER_NODE" \
    --num-gpus-per-node "$GPUS_PER_NODE" --rollout-num-gpus "$ROLLOUT_NUM_GPUS" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" --load "${LOAD_CHECKPOINT:-$HF_CHECKPOINT}" \
    "${CHECKPOINT_ARGS[@]}" \
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async \
    --fully-async-group-prefetch-factor "${GROUP_PREFETCH_FACTOR:-2}" \
    --custom-generate-function-path examples.coding_agent_rl.generate.generate \
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std \
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata \
    --rollout-shuffle --apply-chat-template-kwargs '{"reasoning_effort":"medium"}' \
    --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" --num-steps-per-rollout 1 --micro-batch-size 1 \
    --rollout-max-context-len "$MAX_CONTEXT_LEN" --rollout-max-response-len "$MAX_RESPONSE_LEN" \
    --rollout-temperature 1 --rollout-top-p 0.95 --rollout-stop-token-ids 248046 248044 \
    --use-score-centering --use-tis --tis-clip 2 --tis-clip-low 0 \
    --partial-rollout --flush-cache-interval "$FLUSH_CACHE_INTERVAL" \
    --tensor-model-parallel-size "${TP_SIZE:-4}" --pipeline-model-parallel-size 3 \
    --decoder-first-pipeline-num-layers 22 --decoder-last-pipeline-num-layers 20 \
    --context-parallel-size "${CP_SIZE:-2}" --sequence-parallel \
    --qwen-gdn-implementation "${QWEN_GDN_IMPLEMENTATION:-megatron}" \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}" \
    --log-probs-chunk-size 1024 --balance-data --use-distributed-optimizer \
    --advantage-estimator grpo --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --optimizer-cpu-offload --optimizer-offload-fraction 1 --overlap-cpu-optimizer-d2h-h2d \
    --use-precision-aware-optimizer --use-stateless-adam --no-save-optim \
    --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE" \
    --router-policy "${ROUTER_POLICY:-consistent_hashing}" \
    --sglang-server-concurrency "$SERVER_CONCURRENCY" "${SGLANG_CAPACITY_ARGS[@]}" \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.8}" --sglang-context-length "$MAX_CONTEXT_LEN" \
    --sglang-speculative-algorithm EAGLE --sglang-speculative-num-steps "${SPECULATIVE_STEPS:-3}" \
    --sglang-speculative-eagle-topk 1 --sglang-speculative-num-draft-tokens "${SPECULATIVE_DRAFT_TOKENS:-4}" \
    --sglang-mamba-radix-cache-strategy extra_buffer --sglang-mamba-full-memory-ratio 0.5 \
    --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3 \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 --attention-backend flash \
    "${STORAGE_ARGS[@]}" "${TRACKING_ARGS[@]}" "$@" \
    2>&1 | tee "$LOG_ROOT/run.log"
