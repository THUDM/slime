#!/bin/bash

# Non-colocated GLM-4.7-355B-A32B with delta weight sync.
# 8 actor nodes (TP=8, PP=4, EP=16) + 64 rollout GPUs (8 H100 nodes worth), 16 nodes total.
# Disk transport is active by default; the NCCL block below it is commented out.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/slime/scripts/models/glm4.5-355B-A32B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/GLM-4.7-355B-A32B
   --ref-load /root/GLM-4.7-355B-A32B_torch_dist/
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --num-steps-per-rollout 4
   --balance-data
   --rollout-stop-token-ids 151329 151336 151338
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 2
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 1e-4
   --eps-clip-high 2e-4
   --use-tis
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-delta
   # --wandb-group glm4.7-355B-delta
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-dp-size 4
   --sglang-ep-size 32
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1

   # Receiver batches up to this many bytes per model.load_weights call. Bigger
   # amortizes per-call cost (name resolution, MoE expert remap) but raises peak HBM.
   --sglang-update-weight-delta-chunk-bytes $((2 * 1024 * 1024 * 1024))

   # Max parallel I/O threads for reading delta files from disk (disk transport only).
   --sglang-update-weight-delta-read-workers 4

   # mtp
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
)

# Delta weight sync. Pick one of the two blocks below.

# ── Disk (default) — for training/inference disaggregation across datacenters ────
# `deltas_zstd` is the right pick when shared-FS bandwidth is ≤ ~300 MB/s.
DELTA_ARGS=(
   --update-weight-mode delta
   --update-weight-transport disk
   --update-weight-encoding deltas_zstd
   --update-weight-delta-dir /shared/fs/delta-updates
)

# ── NCCL (baseline) — intra-datacenter, no shared FS ────────────────────────────
# DELTA_ARGS=(
#    --update-weight-mode delta
#    --update-weight-transport nccl
#    --update-weight-encoding indices
# )

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
   --update-weight-buffer-size $((2 * 1024 * 1024 * 1024))
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 8 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 64 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${DELTA_ARGS[@]} \
   ${MISC_ARGS[@]}
