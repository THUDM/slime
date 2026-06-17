#!/bin/bash
#
# Qwen3.5-VL RL training on geo3k dataset.
#
# Supports both Dense (Qwen3.5-9B / Qwen3.5-27B) and MoE (Qwen3.5-35B-A3B,
# Qwen3.5-397B-A17B, ...) variants via the official NVIDIA Megatron-Bridge
# package (>= 0.4.0). Selection is by env var:
#
#     # Dense — default
#     MODEL_NAME=Qwen3.5-9B  ./run_geo3k_qwen35_vl.sh
#     MODEL_NAME=Qwen3.5-27B ./run_geo3k_qwen35_vl.sh
#
#     # MoE
#     MODEL_NAME=Qwen3.5-35B-A3B ./run_geo3k_qwen35_vl.sh
#
# The Megatron-side provider is built by megatron-bridge directly from the
# HuggingFace config of the checkpoint at $HF_CHECKPOINT (no fork required).

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_BACKEND="megatron"
MODEL_NAME=${MODEL_NAME:-"Qwen3.5-9B"}
DATASET_NAME=${SLIME_SCRIPT_DATASET_NAME:-"chenhegu/geo3k_imgurl"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")
MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
BASE_FOLDED=${SLIME_BASE_FOLDED:-"/root"}

# Heuristic: any "*A<digits>B*" suffix denotes a MoE variant (A3B / A17B / ...)
if [[ "$MODEL_NAME" == *A[0-9]*B* ]]; then
    IS_MOE=1
else
    IS_MOE=0
fi

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# ---------------------------------------------------------------------------
# Cleanup (no set -e here: pkill returns non-zero when no process matches)
# ---------------------------------------------------------------------------
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

set -ex

export PYTHONUNBUFFERED=1

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "MODEL_NAME: $MODEL_NAME (IS_MOE=$IS_MOE)"

# ---------------------------------------------------------------------------
# Download model and dataset
# ---------------------------------------------------------------------------
mkdir -p ${BASE_FOLDED}/models ${BASE_FOLDED}/datasets
if [ ! -d "${BASE_FOLDED}/models/${MODEL_NAME}" ]; then
   hf download Qwen/${MODEL_NAME} --local-dir ${BASE_FOLDED}/models/${MODEL_NAME}
fi
if [ ! -d "${BASE_FOLDED}/datasets/${DATASET_LOCAL_NAME}" ]; then
   hf download --repo-type dataset ${DATASET_NAME} --local-dir ${BASE_FOLDED}/datasets/${DATASET_LOCAL_NAME}
fi

# ---------------------------------------------------------------------------
# Args common to dense and MoE
# ---------------------------------------------------------------------------
CKPT_ARGS=(
   --hf-checkpoint ${BASE_FOLDED}/models/${MODEL_NAME}
   --load ${BASE_FOLDED}/models/${MODEL_NAME}
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data ${BASE_FOLDED}/datasets/${DATASET_LOCAL_NAME}/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 512
)

# Required for VLM datasets — geo3k stores image URLs under "images"
MULTIMODAL_KEYS='{"image": "images"}'

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data ${DATASET_LOCAL_NAME} ${BASE_FOLDED}/datasets/${DATASET_LOCAL_NAME}/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# Wandb args (only if WANDB_API_KEY is set)
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime-geo3k-vlm
      --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   --colocate
)

# ---------------------------------------------------------------------------
# Variant-specific args (Dense vs MoE)
# ---------------------------------------------------------------------------
if [ "$IS_MOE" = "1" ]; then
   # MoE branch — Qwen3.5-35B-A3B / 397B-A17B follow the same SGLang recipe
   # as Qwen3-Next (LMSYS cookbook):
   #   https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3-Next
   # i.e. NEXTN speculative decoding (uses the model's built-in MTP head,
   # which Qwen3.5 ships with) + extra_buffer mamba scheduler + page-size=64
   # so radix cache stays enabled. SGLANG_ENABLE_SPEC_V2=1 is exported in
   # the Ray runtime_env below.
   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 8
      --sglang-mem-fraction-static 0.7
      --sglang-ep-size 8
      --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256

      # NEXTN speculative decoding (MTP-based, native to Qwen3.5).
      --sglang-speculative-algorithm NEXTN
      --sglang-speculative-num-steps 3
      --sglang-speculative-eagle-topk 1
      --sglang-speculative-num-draft-tokens 4

      # Hybrid (mamba-style) scheduler tuned for Qwen3-Next / Qwen3.5 MoE.
      # extra_buffer + page-size=64 lets radix cache coexist with
      # speculative decoding. page-size must satisfy
      # FLA_CHUNK_SIZE % page_size == 0 (cookbook value: 64).
      --sglang-mamba-scheduler-strategy extra_buffer
      --sglang-page-size 64

      --sglang-max-running-requests 512
      
      # Workaround: SGLang's symmetric-memory custom all-reduce trips
      # `cudaIpcOpenMemHandle` / `share_graph_inputs` on some driver / IPC
      # configs ("CUDA error: invalid argument" inside custom_all_reduce.cuh).
      # NCCL all-reduce is plenty fast for the rollout workers.
      --sglang-disable-custom-all-reduce
   )

   BACKEND_ARGS=(
      --train-backend megatron
      # MoE Qwen3.5-35B-A3B has num_query_groups = 2 (gated attention)
      --tensor-model-parallel-size 2
      --sequence-parallel
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --expert-model-parallel-size 8
      --expert-tensor-parallel-size 1
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash

      # GDN (Gated DeltaNet, Qwen3.5's linear-attention branch) does not
      # support packed sequences in megatron-core today
      # (`gated_delta_net.py:300 NotImplementedError`). Force the padded
      # BSHD layout — slime's data pipeline then sets
      # packed_seq_params=None and the GDN guard is not tripped.
      --qkv-format bshd
      --micro-batch-size 1
   )
else
   # Dense branch — Qwen3.5-9B / 27B. The bridge derives the full provider
   # config (hidden size, GDN heads, mRoPE sections, etc.) from the HF
   # checkpoint's config.json, so no model-specific Megatron flags are
   # required. We only set parallelism + memory knobs here.
   SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 8
      --sglang-mem-fraction-static 0.7
      --sglang-max-running-requests 512

      # Workaround: SGLang's symmetric-memory custom all-reduce trips
      # `cudaIpcOpenMemHandle` / `share_graph_inputs` on some driver / IPC
      # configs ("CUDA error: invalid argument" inside custom_all_reduce.cuh).
      # NCCL all-reduce is plenty fast for the rollout workers.
      --sglang-disable-custom-all-reduce
   )

   BACKEND_ARGS=(
      --train-backend megatron
      --tensor-model-parallel-size 2
      --sequence-parallel
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash

      # GDN (Gated DeltaNet, Qwen3.5's linear-attention branch) does not
      # support packed sequences in megatron-core today
      # (`gated_delta_net.py:300 NotImplementedError`). Force the padded
      # BSHD layout — slime's data pipeline then sets
      # packed_seq_params=None and the GDN guard is not tripped.
      --qkv-format bshd
      --micro-batch-size 1
   )
fi

# ---------------------------------------------------------------------------
# Optional legacy text-spec MODEL_ARGS (only sourced for the matching variant)
# ---------------------------------------------------------------------------
SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# Source the matching scripts/models/qwen3.5-*.sh if present. In bridge mode
# (--megatron-to-hf-mode bridge) the provider comes from the HF config, but
# slime's CLI parser still needs MODEL_ARGS to be defined. We fall back to an
# empty array if no matching file exists for the requested variant.
MODEL_ARGS=()
case "$MODEL_NAME" in
   *0.8B*)      CANDIDATE_MODEL_SH="${SLIME_DIR}/scripts/models/qwen3.5-0.8B.sh" ;;
   *4B*)        CANDIDATE_MODEL_SH="${SLIME_DIR}/scripts/models/qwen3.5-4B.sh" ;;
   *9B*)        CANDIDATE_MODEL_SH="${SLIME_DIR}/scripts/models/qwen3.5-9B.sh" ;;
   *27B*)       CANDIDATE_MODEL_SH="${SLIME_DIR}/scripts/models/qwen3.5-27B.sh" ;;
   *35B-A3B*)   CANDIDATE_MODEL_SH="${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh" ;;
   *)           CANDIDATE_MODEL_SH="" ;;
esac
if [ -n "$CANDIDATE_MODEL_SH" ] && [ -f "$CANDIDATE_MODEL_SH" ]; then
   # shellcheck disable=SC1090
   source "$CANDIDATE_MODEL_SH"
fi

# ---------------------------------------------------------------------------
# Start Ray (if not external) and submit the job
# ---------------------------------------------------------------------------
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_ENABLE_SPEC_V2\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}
