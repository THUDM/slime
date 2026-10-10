#!/usr/bin/env bash
# Qwen3.8-27B on Xiaomi MiMo SWE, two training nodes and two sunabako nodes.
# Run from a long-lived shell. Uses only this experiment's Ray cluster; never
# runs global pkill or stops unrelated jobs. Sandbox memory mode is explicit.
set -euo pipefail
SLIME_DIR=${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the prepared MiMo data, images, cluster configuration and toolchain directory}
PROMPT_DATA=${PROMPT_DATA:-$DATA_ROOT/train.jsonl}
RUN_ROOT=${RUN_ROOT:-$DATA_ROOT/run}
RUN_LOG=${RUN_LOG:-$RUN_ROOT/run.log}
NUM_ROLLOUT=${NUM_ROLLOUT:-1}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * SAMPLES_PER_PROMPT))}
HF_CHECKPOINT=${HF_CHECKPOINT:-/cloud/oss_checkpoints/Qwen3/Qwen3.8-27B}
TRAIN_HEAD=${TRAIN_HEAD:?Set the training head IP}
TRAIN_WORKER=${TRAIN_WORKER:?Set the second training node IP}
SSH_PORT=${SSH_PORT:-22}
RAY_PORT=${RAY_PORT:-16479}
DASHBOARD_PORT=${DASHBOARD_PORT:-18265}
export MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-65536}
export MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
CHAT_TEMPLATE_KWARGS=${CHAT_TEMPLATE_KWARGS:-'{"reasoning_effort":"medium"}'}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLIME_DIR:${MEGATRON_DIR:-/root/Megatron-LM}"
export MASTER_ADDR="$TRAIN_HEAD" MASTER_PORT=16480
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME} TP_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAY_ADDRESS="$TRAIN_HEAD:$RAY_PORT"
export no_proxy="localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,100.64.0.0/10,$TRAIN_HEAD,$TRAIN_WORKER${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
export SWE_SANDBOX_PROVIDER=sunabako
export SUNABAKO_CLUSTER=${SUNABAKO_CLUSTER:-$DATA_ROOT/cluster.json}
export SUNABAKO_IMAGES=${SUNABAKO_IMAGES:-$DATA_ROOT/images.json}
export SUNABAKO_ARTIFACTS="$RUN_ROOT/sandboxes"
export SUNABAKO_MEMORY_MB=2048
export SUNABAKO_ALLOW_TEST_MEMORY=${SUNABAKO_ALLOW_TEST_MEMORY:?Explicitly set 1 only for bounded functional testing without a hard memory cap, or 0 for production cgroups}
export SLIME_AGENT_NODE_TARBALL=${SLIME_AGENT_NODE_TARBALL:-$DATA_ROOT/toolchain/node-v22.23.3-linux-x64.tar.xz}
export SLIME_AGENT_CC_TARBALL=${SLIME_AGENT_CC_TARBALL:-$DATA_ROOT/toolchain/claude-code-2.1.295.tgz}
export SWE_AGENT=claude_code SWE_TRAIN_PROTOCOL=scaleswe
export SWE_AGENT_TIME_BUDGET_SEC=${SWE_AGENT_TIME_BUDGET_SEC:-1200}
export SWE_EVAL_TIMEOUT_SEC=${SWE_EVAL_TIMEOUT_SEC:-180}
export SWE_ROLLOUT_GUARD_SEC=${SWE_ROLLOUT_GUARD_SEC:-$((SWE_AGENT_TIME_BUDGET_SEC + SWE_EVAL_TIMEOUT_SEC + 300))}
export SWE_BOOT_CONCURRENCY=2
# Preserve every sampled turn when the agent re-renders earlier messages.
export SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS:-0}
export SWE_CC_PROMPT=${SWE_CC_PROMPT:-'Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. Edit source files only; do not modify tests or PROBLEM_STATEMENT.md. The .harness directory contains your own live execution logs, so exclude it from searches. Implement the fix, run the relevant tests, then print a one-line summary and exit. Do not commit.'}
export SLIME_AGENT_CC_EXTRA_ARGS=${SLIME_AGENT_CC_EXTRA_ARGS:-'--max-turns 40 --tools Bash,Read,Edit,Write,Glob,Grep --disable-slash-commands'}
SLIME_AGENT_CC_EXTRA_ENVS=$(python - <<'PYCLI'
import json, os
settings = {
    "MAX_THINKING_TOKENS": "4096",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": os.environ["MAX_RESPONSE_LEN"],
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": os.environ["MAX_CONTEXT_LEN"],
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "70",
}
settings.update(json.loads(os.environ.get("SLIME_AGENT_CC_EXTRA_ENVS", "{}")))
print(json.dumps(settings))
PYCLI
)
export SLIME_AGENT_CC_EXTRA_ENVS
export ADAPTER_BIND_HOST=0.0.0.0 ADAPTER_PUBLIC_HOST=${ADAPTER_PUBLIC_HOST:-auto} ADAPTER_PORT=18091
if [[ ! -s "$PROMPT_DATA" ]]; then
    echo 'PROMPT_DATA is empty or missing; run examples/coding_agent_rl/prepare_mimo.py first' >&2
    exit 1
fi
mkdir -p "$RUN_ROOT"
if [[ -e "$RUN_LOG" ]]; then
    echo 'RUN_LOG already exists; choose a new log path for an explicit retry' >&2
    exit 1
fi
# sunabako is a pip dependency of this Slime example, on both training nodes.
python -c 'import sunabako; print("sunabako client:", sunabako.__file__)'
ssh -p "$SSH_PORT" -o BatchMode=yes "root@$TRAIN_WORKER" \
    "python -c 'import sunabako; print(sunabako.__file__)'"
if [[ ${SUNABAKO_START_RAY:-1} == 1 ]]; then
    ray start --head --node-ip-address "$TRAIN_HEAD" --port "$RAY_PORT" --num-gpus 8 --num-cpus 48 \
        --object-store-memory 2147483648 --disable-usage-stats --dashboard-host=127.0.0.1 \
        --dashboard-port "$DASHBOARD_PORT" --temp-dir /tmp/sunabako-ray \
        --min-worker-port 24000 --max-worker-port 24999
    ssh -p "$SSH_PORT" -o BatchMode=yes "root@$TRAIN_WORKER" \
        "ray start --address=$TRAIN_HEAD:$RAY_PORT --node-ip-address=$TRAIN_WORKER --num-gpus=8 --num-cpus=48 --object-store-memory=2147483648 --disable-usage-stats --min-worker-port=24000 --max-worker-port=24999"
fi
python - <<'PY'
import os,time,ray
ray.init(address=os.environ['RAY_ADDRESS'])
end=time.monotonic()+90
while time.monotonic()<end:
    nodes=[n for n in ray.nodes() if n['Alive']]
    if len(nodes)==2 and sum(n['Resources'].get('GPU',0) for n in nodes)==16:
        print('Verified two-node 16-GPU Ray cluster',flush=True)
        break
    time.sleep(1)
else: raise RuntimeError('Expected exactly two training nodes and 16 GPUs')
ray.shutdown()
PY
# A Ray job runtime environment propagates these settings to remote actors.
RUNTIME_ENV_JSON=$(python - <<'PYENV'
import json, os
keys = {
    "PYTHONPATH", "PYTHONUNBUFFERED", "MASTER_ADDR", "MASTER_PORT",
    "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
    "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS",
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY",
}
keys.update(k for k in os.environ if k.startswith(("SUNABAKO_", "SWE_", "SLIME_AGENT_", "ADAPTER_")))
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if k in os.environ}}))
PYENV
)
source "$SLIME_DIR/scripts/models/qwen3.5-27B.sh"
SGLANG_GRAPH_ARGS=(--sglang-cuda-graph-max-bs-decode 4)
if [[ ${SGLANG_DISABLE_CUDA_GRAPH:-0} == 1 ]]; then
    SGLANG_GRAPH_ARGS+=(--sglang-disable-cuda-graph)
fi
cd "$SLIME_DIR"
ray job submit --address="http://127.0.0.1:$DASHBOARD_PORT" \
    --runtime-env-json="$RUNTIME_ENV_JSON" -- python -u train.py \
    --actor-num-nodes 2 --actor-num-gpus-per-node 8 --num-gpus-per-node 8 --colocate \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" \
    --save "$RUN_ROOT/checkpoints" --save-interval 1 \
    --custom-generate-function-path examples.coding_agent_rl.generate.generate \
    --prompt-data "$PROMPT_DATA" \
    --input-key prompt --label-key label --metadata-key metadata \
    --apply-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
    --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --rollout-max-context-len "$MAX_CONTEXT_LEN" --rollout-max-response-len "$MAX_RESPONSE_LEN" --rollout-temperature 1.0 \
    --rollout-top-p 0.95 \
    --rollout-stop-token-ids 248046 248044 --num-steps-per-rollout 1 \
    --global-batch-size "$GLOBAL_BATCH_SIZE" --micro-batch-size 1 \
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 8 --context-parallel-size 1 \
    --sequence-parallel --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --max-tokens-per-gpu "$MAX_CONTEXT_LEN" --log-probs-chunk-size 1024 --use-dynamic-batch-size \
    --advantage-estimator grpo --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0 --eps-clip 0.2 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer \
    --rollout-num-gpus-per-engine 4 --sglang-mem-fraction-static 0.45 --sglang-context-length "$MAX_CONTEXT_LEN" \
    --sglang-max-running-requests 4 "${SGLANG_GRAPH_ARGS[@]}" \
    --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3 \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 --attention-backend flash \
    --save-debug-rollout-data "$RUN_ROOT/rollout_{rollout_id}.pt" \
    --save-debug-train-data "$RUN_ROOT/train_{rollout_id}.pt" \
    "$@" \
    2>&1 | tee "$RUN_LOG"
