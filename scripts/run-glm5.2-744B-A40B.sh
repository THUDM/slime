#!/bin/bash

# GLM-5.2 744B-A40B RL training on 32 nodes / 256 H100 GPUs with PD disaggregation.

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm5.2-744B-A40B.sh"

if [ -z "${BASE_DIR:-}" ]; then
  echo "BASE_DIR is not set. Please set it to a shared path visible from every node."
  exit 1
fi

SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/GLM-5.2-FP8
   --ref-load $BASE_DIR/GLM-5.2_torch_dist
   --load $BASE_DIR/GLM-5.2_slime
   --save $BASE_DIR/GLM-5.2_slime
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 65536
   --rollout-temperature 1.0

   --global-batch-size 64
)

# TP=4, PP=8, CP=8 consumes all 256 GPUs (32 nodes) for one training group; DP=1.
# Experts use EP=32: expert_tp(1) * ep(32) * pp(8) = 256 = world_size (expert_dp=1).
#
# DSA cross-layer index sharing requires every pipeline stage to START on a
# "computing" layer (index_topk_freq=4, index_skip_topk_offset=3 -> computing
# layers are 1,2,3,7,11,...,75). A uniform 78/8 split would start stages on skip
# layers and fail. We instead use first=14, last=16, leaving 6 middle stages of
# (78-14-16)/6 = 8 layers each. Stage starts land on global layers
# 1,15,23,31,39,47,55,63 -- all computing layers.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 8
   --decoder-first-pipeline-num-layers 14
   --decoder-last-pipeline-num-layers 16
   --context-parallel-size 8
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --data-pad-size-multiplier 1024
   --log-probs-chunk-size 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
   --tis-clip-low 0.5
   --tis-clip 2.0
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
   # --wandb-project slime-dev
   # --wandb-group glm5.2-744B-A40B
)

SGLANG_CONFIG_FILE=$(mktemp /tmp/sglang_glm52_744B_A40B_XXXXXX.yaml)
# PD disaggregation: 1 prefill engine (64 GPU) + 3 decode engines (192 GPU) = 256.
# Each engine spans 64 GPUs (EP=64, within DeepEP's supported rank set). Prefill
# uses the auto DeepEP path; decode uses low_latency + deep_gemm for throughput.
cat > "${SGLANG_CONFIG_FILE}" <<CFG
sglang:
  - name: default
    server_groups:
      - worker_type: prefill
        num_gpus: 64
        num_gpus_per_engine: 64
        overrides:
          dp_size: 64
          ep_size: 64
          enable_dp_attention: true
          enable_dp_lm_head: true
          moe_dense_tp_size: 1
          load_balance_method: follow_bootstrap_room
          chunked_prefill_size: 131072
          max_running_requests: 512
          deepep_mode: auto
      - worker_type: decode
        num_gpus: 192
        num_gpus_per_engine: 64
        overrides:
          dp_size: 64
          ep_size: 64
          enable_dp_attention: true
          enable_dp_lm_head: true
          moe_dense_tp_size: 1
          load_balance_method: round_robin
          max_running_requests: 768
          cuda_graph_max_bs: 12
          deepep_mode: low_latency
          moe_runner_backend: deep_gemm
          disable_overlap_schedule: true
CFG

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 64
   --sglang-mem-fraction-static 0.70

   --sglang-enable-dp-attention
   --sglang-ep-size 64
   --sglang-dp-size 64
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head

   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode auto

   --sglang-page-size 64
   --sglang-kv-cache-dtype fp8_e4m3
   --sglang-nsa-decode-backend flashmla_kv
   --sglang-nsa-prefill-backend flashmla_sparse
   --sglang-attention-backend nsa
   --sglang-cuda-graph-max-bs 8
   --sglang-disable-overlap-schedule

   --sglang-max-running-requests 512
   --sglang-watchdog-timeout 3600

   # PD transport over RDMA/IB.
   --sglang-disaggregation-transfer-backend mooncake
   --sglang-disaggregation-ib-device "mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107"
   --sglang-config "${SGLANG_CONFIG_FILE}"

   # MTP / EAGLE speculative decoding using the model's own next-token-prediction
   # layer (the GLM-5.2 checkpoint ships an MTP layer), so no separate draft model
   # is needed.
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 4
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 5
   --sglang-speculative-draft-attention-backend nsa
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --moe-token-dispatcher-type alltoall
)

if [ -z "${MASTER_ADDR:-}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi

NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},10.0.0.0/8,100.64.0.0/10"
export no_proxy="${NO_PROXY_LIST}"
export NO_PROXY="${NO_PROXY_LIST}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [ -n "${HOSTFILE:-}" ]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
      continue
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "PYTHONPATH": "/root/slime:/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTHONUNBUFFERED": "1",
    "no_proxy": "${NO_PROXY_LIST}",
    "NO_PROXY": "${NO_PROXY_LIST}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_NET_GDR_LEVEL": "2",
    "NCCL_IB_QPS_PER_CONNECTION": "2",
    "NCCL_IB_TC": "160",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_PXN_DISABLE": "0",
    "NCCL_MIN_CTAS": "4",
    "NVTE_FWD_LAYERNORM_SM_MARGIN": "8",
    "NVTE_BWD_LAYERNORM_SM_MARGIN": "8",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "MC_IB_PCI_RELAXED_ORDERING": "1",
    "MLP_SKIP_SORT_RDMA": "true",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "true",
    "NVSHMEM_DISABLE_NCCL": "1"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 32 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --no-check-for-nan-in-loss-and-grad \
   --update-weight-buffer-size $(( 1024 * 1024 * 1024 * 2 )) \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
