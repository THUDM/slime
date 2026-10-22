#!/bin/bash

# NVIDIA-Nemotron-3.5-Super-EA-09112026 VL SFT on geo3k (4 nodes x 8 H100 80 GB, TP2 / EP32 / PP1).
#
# scripts/run-qwen3.5-35B-A3B-sft.sh with the model swapped, and the parallelism, DeepEP and MTP
# settings of examples/geo3k_vlm/run_geo3k_nemotron35.sh (CP2, DeepEP and MTP training on by
# default; CONTEXT_PARALLEL_SIZE=1, USE_DEEPEP=0, USE_MTP=0 turn them off). The chat template is
# ChatML-shaped (`<|im_start|>assistant\n ... <|im_end|>`), so the qwen3_5 loss mask
# applies unchanged. Samples carry images, so sft_rollout runs the processor.
#
# Multi-node needs an external Ray cluster with the head on this node (:8265):
# start it first, then run this script on the head node.
#
# Optional environment:
#   SFT_DATA              parquet with `messages` + `images` (default /root/datasets/geo3k_sft/train.parquet)
#   SFT_NUM_ROLLOUT       number of SFT steps (default 47)
#   SFT_SAVE_DIR          checkpoint dir (default /root/ckpt/nemotron35_vl_sft); empty disables saving
#   SFT_SAVE_INTERVAL     default = SFT_NUM_ROLLOUT, i.e. one checkpoint at the end
#   USE_DEEPEP=0          native all-to-all dispatcher instead of DeepEP (default 1)
#   CONTEXT_PARALLEL_SIZE (default 2), TENSOR_MODEL_PARALLEL_SIZE (default 2),
#   EXPERT_MODEL_PARALLEL_SIZE (default 32), EXPERT_TENSOR_PARALLEL_SIZE (default 1)
#   USE_MTP=0             do not train the MTP head (default 1); MTP_NUM_STEPS (2), MTP_LOSS_SCALING_FACTOR (0.2)
#   SLIME_SCRIPT_EXTRA_ARGS  extra train flags, appended last

pkill -9 sglang
sleep 3
pkill -9 slime

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

MODEL_NAME="NVIDIA-Nemotron-3.5-Super-EA-09112026"
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
NUM_NODES=${SLIME_SCRIPT_NUM_NODES:-4}
SFT_NUM_ROLLOUT=${SFT_NUM_ROLLOUT:-47}
SFT_SAVE_DIR=${SFT_SAVE_DIR-/root/ckpt/nemotron35_vl_sft}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/nemotron3.5-super-vl.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/models/${MODEL_NAME}
   --load /root/models/${MODEL_NAME}
)
if [ -n "${SFT_SAVE_DIR}" ]; then
   CKPT_ARGS+=(--save "${SFT_SAVE_DIR}" --save-interval ${SFT_SAVE_INTERVAL:-${SFT_NUM_ROLLOUT}})
fi

MULTIMODAL_KEYS='{"image": "images"}'

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${SFT_DATA:-/root/datasets/geo3k_sft/train.parquet}
   --input-key messages
   --rollout-shuffle
   --num-rollout ${SFT_NUM_ROLLOUT}
   --rollout-batch-size 64
   --global-batch-size 64

   --loss-type sft_loss
   --loss-mask-type qwen3_5
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE:-2}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size ${CONTEXT_PARALLEL_SIZE:-2}
   --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE:-32}
   --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE:-1}

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # No dynamic batching (packing), as in run_geo3k_nemotron35.sh: one sample per micro-batch.
   # With --use-dynamic-batch-size at CP2 all of a rank's samples land in one packed micro-batch
   # and the first backward pass runs out of memory on 80 GB unless --max-tokens-per-gpu is 1024.
   --micro-batch-size 1
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# Same flag-spelling resolution as run_geo3k_nemotron35.sh: slime drops unknown
# Megatron flags silently, so check the Megatron that is actually mounted.
MEGATRON_ARGUMENTS_PY=/root/Megatron-LM/megatron/training/arguments.py
# --optimizer-cpu-offload is checked first: recent Megatron dev has both spellings, and
# there --offload-optimizer-states is a different mechanism (RL-time offload of the optimizer
# state, not CPU-resident Adam), so picking it leaves the optimizer on the GPU and runs out of
# memory when DeepEP allocates its buffers.
if grep -q "optimizer-cpu-offload" "${MEGATRON_ARGUMENTS_PY}" 2>/dev/null; then
   OPTIMIZER_ARGS+=(--optimizer-cpu-offload --use-precision-aware-optimizer)
elif grep -q "offload_optimizer_states" "${MEGATRON_ARGUMENTS_PY}" 2>/dev/null; then
   OPTIMIZER_ARGS+=(--offload-optimizer-states)
else
   echo "${MEGATRON_ARGUMENTS_PY} has neither --offload-optimizer-states nor --optimizer-cpu-offload" >&2
   exit 2
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# DeepEP's flex dispatcher fails with expert-TP > 1 (CUDA "unspecified launch failure" on
# every rank), so fall back to the native all-to-all there.
if [ "${EXPERT_TENSOR_PARALLEL_SIZE:-1}" -gt 1 ] && [ "${USE_DEEPEP:-1}" = "1" ]; then
   echo "EXPERT_TENSOR_PARALLEL_SIZE>1: DeepEP is not supported, using the native all-to-all dispatcher" >&2
   USE_DEEPEP=0
fi
if [ "${USE_DEEPEP:-1}" = "1" ]; then
   MISC_ARGS+=(
      --moe-token-dispatcher-type flex
      --moe-flex-dispatcher-backend deepep
   )
fi

# MTP training, as in run_geo3k_nemotron35.sh: one `/*E` pattern segment per depth, one shared layer.
if [ "${USE_MTP:-1}" = "1" ]; then
   MTP_NUM_STEPS=${MTP_NUM_STEPS:-2}
   MTP_PATTERN_SUFFIX=""
   for (( _depth = 0; _depth < MTP_NUM_STEPS; _depth++ )); do
      MTP_PATTERN_SUFFIX+="/*E"
   done
   MODEL_ARGS+=(
      --hybrid-override-pattern "${NEMOTRON_H_PATTERN}${MTP_PATTERN_SUFFIX}"
      --mtp-num-layers ${MTP_NUM_STEPS}
      --mtp-use-repeated-layer
      --enable-mtp-training
      --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}
   )
fi

read -r -a EXTRA_ARGS <<< "${SLIME_SCRIPT_EXTRA_ARGS:-}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# Quoted expansions: --hybrid-override-pattern contains `*` (see run_geo3k_nemotron35.sh).
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
