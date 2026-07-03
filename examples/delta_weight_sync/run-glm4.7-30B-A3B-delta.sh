#!/bin/bash
# Disk delta weight-sync demo on GLM-4.7-Flash (30B-A3B), non-colocated, 2 nodes x 8 GPU.
# The trainer publishes per-tensor deltas to --update-weight-disk-dir as a canonical HF directory;
# each rollout host applies them into --update-weight-local-checkpoint-dir and reloads via the
# vanilla update_weights_from_disk path.
#
# Prerequisites:
#   - A 2-node (16-GPU) Ray cluster, this script run on the head node.
#   - GLM-4.7-Flash HF checkpoint + its torch_dist conversion (tools/convert_hf_to_torch_dist.py).
#   - dapo-math-17k.jsonl.
#   - --update-weight-disk-dir on a filesystem both nodes share. On an object-store-backed volume
#     that needs an explicit commit/refresh to surface writes across hosts, also pass
#     --custom-delta-pre-push-path / --custom-delta-pre-read-path (see the doc).

set -ex
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/glm4.7-30B-A3B.sh"

MODEL_DIR=${MODEL_DIR:-/root/models/GLM-4.7-Flash}
DATA_PATH=${DATA_PATH:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${MODEL_DIR}_torch_dist"
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-response-len 8192
   --global-batch-size 128
)

# Disk delta weight sync (the point of this example).
WEIGHT_SYNC_ARGS=(
   --update-weight-mode delta
   --update-weight-transport disk
   --update-weight-disk-dir /shared/fs/glm47-delta-updates
   --update-weight-local-checkpoint-dir /local/nvme/glm47-rollout-ckpt
   --update-weight-delta-encoding xor
   --update-weight-delta-checksum xxh3-128
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 2
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.8
   --sglang-enable-dp-attention
   --sglang-dp-size 8
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

# Non-colocated: 16 actor GPUs (2 x 8) train while a 16-GPU rollout pool generates (delta mode
# requires non-colocation).
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 16 \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${WEIGHT_SYNC_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${SGLANG_ARGS[@]}"
