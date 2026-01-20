#!/bin/bash

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

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../scripts/models/glm4.7-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/GLM-4.7-Flash
   --ref-load $BASE_DIR/GLM-4.7-Flash_torch_dist/
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 128
   #--over-sampling-batch-size 256
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1.0

   --global-batch-size 1024
   #--balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime24 $BASE_DIR/rl_data/aime-2024.jsonl
   --n-samples-per-eval-prompt 2
   --eval-max-response-len 16384
   --eval-temperature 0.6
   --eval-top-p 0.95
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --decoder-last-pipeline-num-layers 23

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
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
   # --wandb-group glm4.7-flash
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.8
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1

   # mtp
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3

   --sglang-cuda-graph-max-bs 64

   --sglang-max-running-requests 512
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash

   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)

# launch the master node of ray in container
export MASTER_ADDR=${MLP_WORKER_0_HOST}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
  if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
    continue
  fi
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
done
wait

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "MASTER_ADDR": "${MLP_WORKER_0_HOST}",
        "PYTHONPATH": "/root/Megatron-LM/",
        "NCCL_CUMEM_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
        "NCCL_IB_TC": "160",
        "NCCL_PXN_DISABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_NET_GDR_LEVEL": "4",
        "NCCL_IB_RETRY_CNT": "7",
        "NCCL_IB_TIMEOUT": "32",
        "NCCL_IB_QPS_PER_CONNECTION": "8",
        "NCCL_P2P_LEVEL": "NVL",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_MIN_CTAS": "4",
        "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_btl": "^openib",
        "OMPI_MCA_routed": "direct",
        "OMPI_MCA_routed_radix": "1024",
        "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
        "OMPI_MCA_oob_tcp_if_include": "${MLP_SOCKET_IFNAME}",
        "OMPI_MCA_btl_tcp_if_include": "${MLP_SOCKET_IFNAME}"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --save-debug-rollout-data "data.pt" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
