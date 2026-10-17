#!/usr/bin/env bash
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export NO_PROXY="${no_proxy}"

# Submit to an existing 8-node x 8-H100 Ray cluster. All paths must be shared.
# Assume the BF16 GLM-5.3-Flash checkpoint and dapo-math-17k are already
# downloaded under BASE_DIR, which must be mounted on every Ray node.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/models/glm5.3-flash-300B.sh"
BASE_DIR="${BASE_DIR:-/data}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${BASE_DIR}/GLM-5.3-Flash}"
PROMPT_DATA="${PROMPT_DATA:-${BASE_DIR}/dapo-math-17k/dapo-math-17k.jsonl}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/local/glm53/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${RUN_DIR}"
export REPO_ROOT HF_CHECKPOINT
# Let NCCL/Gloo choose their interfaces unless the cluster needs an override.
export SOCKET_IFNAME="${SOCKET_IFNAME:-}"
export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

RUNTIME_ENV_JSON="$(python - <<'PY'
import json, os
env = {
    'PYTHONPATH': os.environ['REPO_ROOT'] + ':' + os.environ['MEGATRON_PATH'],
    'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'NCCL_IB_TIMEOUT': '22', 'NCCL_IB_QPS_PER_CONNECTION': '2',
    'NCCL_NVLS_ENABLE': '0', 'NCCL_CUMEM_ENABLE': '0',
    # Ray workers inherit the head/worker daemon environment as well.
    'http_proxy': '', 'https_proxy': '', 'HTTP_PROXY': '', 'HTTPS_PROXY': '',
    'all_proxy': '', 'ALL_PROXY': '',
    'no_proxy': os.environ['no_proxy'],
    'NO_PROXY': os.environ['NO_PROXY'],
    'TOKENIZERS_PARALLELISM': 'false',
}
if os.environ['SOCKET_IFNAME']:
    env['NCCL_SOCKET_IFNAME'] = env['GLOO_SOCKET_IFNAME'] = os.environ['SOCKET_IFNAME']
print(json.dumps({'env_vars': env}))
PY
)"

cd "${REPO_ROOT}"
# Populate remote-code caches once per node before concurrent worker startup.
python - <<'PY'
import os
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ray.init(address="auto", log_to_driver=False)

@ray.remote
def prepare_processor(path):
    from transformers import AutoProcessor, AutoTokenizer
    AutoProcessor.from_pretrained(path, trust_remote_code=True)
    AutoTokenizer.from_pretrained(path, trust_remote_code=True)

ray.get([
    prepare_processor.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(os.environ["HF_CHECKPOINT"])
    for node in ray.nodes() if node["Alive"]
])
ray.shutdown()
PY

ray job submit --address="${RAY_DASHBOARD:-http://127.0.0.1:8265}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python train.py \
    --actor-num-nodes 8 --actor-num-gpus-per-node 8 --colocate \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" --load "${HF_CHECKPOINT}" \
    --megatron-to-hf-mode raw \
    --prompt-data "${PROMPT_DATA}" --input-key prompt --label-key label --apply-chat-template \
    --rm-type "${RM_TYPE:-deepscaler}" \
    --num-rollout "${NUM_ROLLOUT:-3}" --rollout-batch-size 16 --n-samples-per-prompt 8 \
    --global-batch-size 128 --micro-batch-size 1 \
    --rollout-max-response-len "${RESPONSE_LENGTH:-262144}" \
    --rollout-temperature 1.0 --rollout-top-p 0.95 \
    --use-rollout-routing-replay --use-score-centering --pg-loss-type reinforce \
    --advantage-estimator grpo --disable-grpo-std-normalization --calculate-per-token-loss \
    --use-tis --tis-clip-low 0.5 --tis-clip 2.0 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4 \
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11 \
    --context-parallel-size "${CP_SIZE:-8}" --expert-model-parallel-size 16 --expert-tensor-parallel-size 1 \
    --allgather-cp \
    --moe-token-dispatcher-type alltoall \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --seq-length 524288 --max-position-embeddings 524288 \
    --use-dynamic-batch-size --max-tokens-per-gpu 2048 --data-pad-size-multiplier 128 \
    --log-probs-chunk-size 256 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant \
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer \
    --no-load-optim --no-load-rng --no-save-optim \
    --freeze-params-name-list 'model.visual.*' \
    --rollout-num-gpus-per-engine 16 \
    --sglang-dp-size 16 --sglang-ep-size 16 --sglang-enable-dp-attention \
    --sglang-enable-dp-lm-head --sglang-moe-dense-tp-size 1 \
    --sglang-mm-enable-dp-encoder --sglang-disable-shared-experts-fusion \
    --sglang-dsa-prefill-backend fa3 --sglang-dsa-decode-backend fa3 \
    --sglang-mem-fraction-static 0.75 --sglang-context-length 524288 \
    --sglang-max-running-requests 32 --sglang-chunked-prefill-size 4096 \
    --sglang-disable-overlap-schedule --sglang-cuda-graph-max-bs-decode 8 \
    --sglang-mamba-ssm-dtype float32 --sglang-page-size 64 \
    --sglang-kv-cache-dtype bfloat16 \
    --sglang-watchdog-timeout 3600 \
    --attention-dropout 0 --hidden-dropout 0 --bf16 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --distributed-timeout-minutes 60 \
    --update-weight-buffer-size 1073741824 \
    --save-debug-train-data "${RUN_DIR}/train-{rollout_id}.pt" \
    "$@"
