#!/bin/bash
#
# Qwen3-30B-A3B deterministic train/rollout (EP8 colocate).
#
# Same workload shape as scripts/run-qwen3-30B-A3B.sh (GBS=256, n=8,
# resp=8192, CPU-offload Adam, GRPO), with the aligned Megatron↔SGLang stack
# from tests/test_qwen3_30b_a3b_deterministic_e2e.py:
#   DeepGEMM batch-invariant FP8 MoE, FA4, fp32 router, DeepEP LL,
#   qwen3_moe_aligned attention, --deterministic-mode.
#
# DeepGEMM alignment requires TP=1 / ETP=1, so this is pure EP8 (not the
# production TP=4 EP=8 layout).
#
# Checkpoint: expert-FP8 HF (attn BF16), e.g. Qwen3-30B-A3B-FP8-experts.
#
#   bash scripts/run-qwen3-30B-A3B-deterministic.sh
#   NUM_ROLLOUT=1 ENABLE_EVAL=0 bash scripts/run-qwen3-30B-A3B-deterministic.sh
#   SLIME_E2E_SHAPE=1 NUM_ROLLOUT=1 ENABLE_EVAL=0 bash scripts/run-qwen3-30B-A3B-deterministic.sh
#   CI_TEST=1 NUM_ROLLOUT=1 ENABLE_EVAL=0 bash scripts/run-qwen3-30B-A3B-deterministic.sh
#
# Env overrides: HF_MODEL REF_LOAD LOAD SAVE PROMPT_DATA EVAL_DATA NUM_ROLLOUT
#   CI_TEST ENABLE_EVAL USE_WANDB MAX_TRAIN_ROLLOUT_DIFF SGLANG_KV_CACHE_DTYPE
#   SLIME_E2E_SHAPE ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE
#   ROLLOUT_MAX_RESPONSE_LEN ROLLOUT_MAX_CONTEXT_LEN MAX_TOKENS_PER_GPU
#   SGLANG_ROOT MEGATRON_ROOT MLP_SOCKET_IFNAME MASTER_ADDR

# Shared-container safe cleanup (do not pkill python).
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3

set -ex

# Ray prestarts many workers; default soft nofile=1024 OOMs the raylet.
ulimit -n 524288

export PYTHONUNBUFFERED=1
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${SLIME_ROOT}"

# Architecture only. Do not enable fused MoE permute (DeepGEMM alignment path).
MODEL_ARGS=(
   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 32
   --num-query-groups 4
   --kv-channels 128
   --num-layers 48
   --hidden-size 2048
   --ffn-hidden-size 6144
   --normalization RMSNorm
   --position-embedding-type rope
   --no-position-embedding
   --norm-epsilon 1e-6
   --rotary-percent 1.0
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 151936
   --make-vocab-size-divisible-by 16
   --rotary-base 1000000
   --moe-ffn-hidden-size 768
   --moe-router-score-function softmax
   --moe-router-topk-scaling-factor 1.0
   --moe-router-topk 8
   --moe-layer-freq '[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'
   --num-experts 128
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-aux-loss-coeff 0
)

# Container defaults under /root; override with HF_MODEL / PROMPT_DATA / etc.
HF_MODEL="${HF_MODEL:-/root/Qwen3-30B-A3B-FP8-experts}"
REF_LOAD="${REF_LOAD:-${HF_MODEL}}"
LOAD="${LOAD:-${HF_MODEL}}"
SAVE="${SAVE:-/root/Qwen3-30B-A3B_slime_det/}"
PROMPT_DATA="${PROMPT_DATA:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-/root/datasets/aime-2024/aime-2024.jsonl}"
SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-bfloat16}"
CONTEXT_PARALLEL_SIZE="${SLIME_E2E_CONTEXT_PARALLEL_SIZE:-1}"
MAX_TRAIN_ROLLOUT_DIFF="${MAX_TRAIN_ROLLOUT_DIFF:-9.999e-7}"
CI_TEST="${CI_TEST:-0}"
ENABLE_EVAL="${ENABLE_EVAL:-0}"
USE_WANDB="${USE_WANDB:-0}"

if [ ! -f "${PROMPT_DATA}" ] && [ -f /root/dapo-math-17k/dapo-math-17k.jsonl ]; then
    PROMPT_DATA=/root/dapo-math-17k/dapo-math-17k.jsonl
fi
if [ ! -f "${EVAL_DATA}" ] && [ -f /root/aime-2024/aime-2024.jsonl ]; then
    EVAL_DATA=/root/aime-2024/aime-2024.jsonl
fi

mkdir -p "${SAVE}"

DEEPGEMM_LAYERS=( $(seq 0 47) )
JIT_KERNELS_DIR="${SLIME_ROOT}/slime/backends/sglang_utils/jit_kernels"
PYTHONPATH_VALUE="${SLIME_ROOT}:${MEGATRON_ROOT}:${SGLANG_ROOT}/python"
if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"
fi
export PYTHONPATH="${PYTHONPATH_VALUE}"

if [ "${KV_CACHE_DTYPE}" = "fp8_e4m3" ]; then
    DSA_KV_FP8_QAT=1
else
    DSA_KV_FP8_QAT=0
fi

# alignment_env() + Qwen3 overrides (expert ffn=768 → MASKED_GEMM_FAST_ACT=0)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_P2P_LEVEL=NVL
export NCCL_ALGO="^NVLS"
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_COMPILE_DISABLE=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export TE_DISABLE_FA3=TRUE
export NVSHMEM_DISABLE_NCCL=1
export SGLANG_DEEPGEMM_BATCH_INVARIANT=1
export SGLANG_DEEPGEMM_PAD_EXPERT_M=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
export SGLANG_JIT_KERNEL_EXTRA_PATH="${JIT_KERNELS_DIR}"
export SGLANG_MASKED_GEMM_FAST_ACT="${SGLANG_MASKED_GEMM_FAST_ACT:-0}"
export SGLANG_DEEPEP_LL_PREFILL_STAGING=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-64}"
export SGLANG_DSA_FUSE_TOPK=0
export SGLANG_DISABLE_DSA_INDEXER_FUSION=1
export SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0
export INDEXER_ROPE_NEOX_STYLE=0
export MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS=1
export MEGATRON_USE_SGLANG_FP8_INDEXER=1
export MEGATRON_USE_SGLANG_ROUTER_GEMM=1
export MEGATRON_USE_SGLANG_ROPE=1
export MEGATRON_USE_SGLANG_SPARSE_MLA=1
export DSA_KV_FP8_QAT
export DSA_KV_FP8_QAT_BLOCK_SIZE=128
# Only ENABLE=0. Do NOT set SGL*_DISABLE_*: SGLang deprecation shim copies
# DISABLE into ENABLE and would re-arm the unbalanced-memory RuntimeError.
unset SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK || true
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
export QWEN3_MOE_ALIGNED_SPEC="${QWEN3_MOE_ALIGNED_SPEC:-1}"
export QWEN3_ALIGNED_USE_FUSED_QK_ROPE="${QWEN3_ALIGNED_USE_FUSED_QK_ROPE:-1}"
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK:-1}"
export SGLANG_KV_CACHE_DTYPE="${KV_CACHE_DTYPE}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"

if [ -n "${MLP_SOCKET_IFNAME:-}" ]; then
    export GLOO_SOCKET_IFNAME="${MLP_SOCKET_IFNAME}"
    export TP_SOCKET_IFNAME="${MLP_SOCKET_IFNAME}"
    export NCCL_SOCKET_IFNAME="${MLP_SOCKET_IFNAME}"
    export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="${MLP_SOCKET_IFNAME}"
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_MODEL}"
   --ref-load "${REF_LOAD}"
   --load "${LOAD}"
   --save "${SAVE}"
   --save-interval 20
)

# SLIME_E2E_SHAPE=1: same alignment stack, gate-sized workload.
E2E_SHAPE="${SLIME_E2E_SHAPE:-0}"
if [ "${E2E_SHAPE}" = "1" ]; then
  ROLLOUT_BATCH_SIZE=8
  N_SAMPLES_PER_PROMPT=1
  GLOBAL_BATCH_SIZE=8
  ROLLOUT_MAX_CONTEXT_LEN=4096
  ROLLOUT_MAX_RESPONSE_LEN=32
  MAX_TOKENS_PER_GPU=8192
  SGLANG_CHUNKED_PREFILL_SIZE=4096
  SGLANG_CONTEXT_LENGTH=8192
  SGLANG_MAX_PREFILL_TOKENS=4096
  SGLANG_MEM_FRACTION_STATIC=0.50
else
  # Match run-qwen3-30B-A3B.sh defaults; override via env as needed.
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
  ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
  ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
  MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20480}"
  SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
  SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
  SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-8192}"
  SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.50}"
fi
echo "SHAPE: batch=${ROLLOUT_BATCH_SIZE} n=${N_SAMPLES_PER_PROMPT} gbs=${GLOBAL_BATCH_SIZE} resp=${ROLLOUT_MAX_RESPONSE_LEN} ctx=${ROLLOUT_MAX_CONTEXT_LEN}"
echo "CKPT: hf=${HF_MODEL} load=${LOAD} save=${SAVE}"

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1
   --rollout-top-p 1.0

   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

EVAL_ARGS=()
if [ "${ENABLE_EVAL}" != "0" ]; then
    EVAL_ARGS=(
       --eval-interval 20
       --eval-prompt-data aime "${EVAL_DATA}"
       --n-samples-per-eval-prompt 16
       --eval-max-response-len 16384
       --eval-top-p 1
    )
fi

# TP=1 EP=8: DeepGEMM alignment forbids production TP=4
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --data-pad-size-multiplier 512
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --disable-grpo-std-normalization
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [ "${USE_WANDB}" != "0" ]; then
    WANDB_ARGS=(
       --use-wandb
       --wandb-project "${WANDB_PROJECT:-slime-deterministic}"
       --wandb-group "${WANDB_GROUP:-qwen3-30B-A3B-deterministic}"
       --wandb-mode "${WANDB_MODE:-offline}"
       --wandb-dir "${WANDB_DIR:-/tmp/slime_det_wandb}"
    )
fi

SGLANG_ARGS=(
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine 8
   --sglang-server-concurrency 128
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-ep-size 8
   --sglang-dp-size 8
   --sglang-moe-dp-size 1
   --sglang-moe-dense-tp-size 1
   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode low_latency
   --sglang-moe-runner-backend deep_gemm
   --sglang-fp8-gemm-runner-backend deep_gemm
   --sglang-page-size 64
   --sglang-kv-cache-dtype "${KV_CACHE_DTYPE}"
   --sglang-attention-backend fa4
   --sglang-enable-fused-qk-norm-rope
   --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
   --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
   --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
   --sglang-enable-fp32-moe-router
   --sglang-enable-deterministic-inference
   --sglang-disable-prefill-cuda-graph
   --sglang-cuda-graph-max-bs-decode 64
   --sglang-watchdog-timeout 7200
   --sglang-dist-timeout 1800
   --sglang-trust-remote-code
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
   --update-weight-mode full
   --update-weight-transport nccl
   --update-weight-buffer-size 2147483648
   --no-check-for-nan-in-loss-and-grad
   --train-env-vars "{\"PYTORCH_CUDA_ALLOC_CONF\":\"${SLIME_ALLOC_CONF:-expandable_segments:True}\",\"CUDA_LAUNCH_BLOCKING\":\"${ALIGNED_LAUNCH_BLOCKING:-0}\"}"
   --custom-megatron-before-log-prob-hook-path
     slime.backends.megatron_utils.alignment.deepgemm_forward.enable_deepgemm_all_forward
   --custom-megatron-before-train-step-hook-path
     slime.backends.megatron_utils.alignment.deepgemm_forward.enable_deepgemm_all_forward_before_train_step
   --megatron-deepgemm-forward-layers "${DEEPGEMM_LAYERS[@]}"
   --megatron-deepgemm-moe-forward-layers "${DEEPGEMM_LAYERS[@]}"
   --deterministic-mode
   --skip-eval-before-train
)

if [ "${QWEN3_MOE_ALIGNED_SPEC}" = "1" ]; then
    MISC_ARGS+=(
       --spec slime_plugins.models.qwen3_moe_aligned get_qwen3_moe_aligned_spec
    )
fi

CI_ARGS=()
if [ "${CI_TEST}" != "0" ]; then
    CI_ARGS=(
       --ci-test
       --ci-disable-kl-checker
       --ci-train-rollout-logprob-abs-diff-threshold "${MAX_TRAIN_ROLLOUT_DIFF}"
    )
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_ADDRESS="${MASTER_ADDR}:6379"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --num-cpus 16 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH_VALUE}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"RAY_ADDRESS\": \"${MASTER_ADDR}:6379\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NO_PROXY\": \"*\",
    \"no_proxy\": \"*\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_P2P_LEVEL\": \"NVL\",
    \"NCCL_ALGO\": \"^NVLS\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"CUBLAS_WORKSPACE_CONFIG\": \":4096:8\",
    \"TORCH_COMPILE_DISABLE\": \"1\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"0\",
    \"TE_DISABLE_FA3\": \"TRUE\",
    \"NVSHMEM_DISABLE_NCCL\": \"1\",
    \"SGLANG_DEEPGEMM_BATCH_INVARIANT\": \"1\",
    \"SGLANG_DEEPGEMM_PAD_EXPERT_M\": \"1\",
    \"SGLANG_JIT_DEEPGEMM_PRECOMPILE\": \"false\",
    \"SGLANG_JIT_KERNEL_EXTRA_PATH\": \"${JIT_KERNELS_DIR}\",
    \"SGLANG_MASKED_GEMM_FAST_ACT\": \"${SGLANG_MASKED_GEMM_FAST_ACT}\",
    \"SGLANG_DEEPEP_LL_PREFILL_STAGING\": \"1\",
    \"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK\": \"${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK}\",
    \"SGLANG_DSA_FUSE_TOPK\": \"0\",
    \"SGLANG_DISABLE_DSA_INDEXER_FUSION\": \"1\",
    \"SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD\": \"0\",
    \"INDEXER_ROPE_NEOX_STYLE\": \"0\",
    \"MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS\": \"1\",
    \"MEGATRON_USE_SGLANG_FP8_INDEXER\": \"1\",
    \"MEGATRON_USE_SGLANG_ROUTER_GEMM\": \"1\",
    \"MEGATRON_USE_SGLANG_ROPE\": \"1\",
    \"MEGATRON_USE_SGLANG_SPARSE_MLA\": \"1\",
    \"DSA_KV_FP8_QAT\": \"${DSA_KV_FP8_QAT}\",
    \"DSA_KV_FP8_QAT_BLOCK_SIZE\": \"128\",
    \"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK\": \"0\",
    \"QWEN3_MOE_ALIGNED_SPEC\": \"${QWEN3_MOE_ALIGNED_SPEC}\",
    \"QWEN3_ALIGNED_USE_FUSED_QK_ROPE\": \"${QWEN3_ALIGNED_USE_FUSED_QK_ROPE}\",
    \"SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK\": \"${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK}\",
    \"SGLANG_KV_CACHE_DTYPE\": \"${KV_CACHE_DTYPE}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CI_ARGS[@]}
