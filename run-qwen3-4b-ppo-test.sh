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
export TP_SIZE=2
export PP_SIZE=1
export CP_SIZE=2

export MASTER_ADDR=${MLP_WORKER_0_HOST}
export MASTER_PORT=${MLP_WORKER_0_PORT}
export GLOO_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}
export NCCL_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}
export no_proxy=localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint /cloud/oss_checkpoints/Qwen3/Qwen3-4B
   --ref-load /mnt/o1_alicloud/personal/zzl/mcore_checkpoint/Qwen3-4B
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/o1_alicloud/personal/zzl/rl_data/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0

   --global-batch-size 32
   --balance-data
)

EVAL_ARGS=(
   #--eval-interval 10
   --eval-prompt-data aime24 /mnt/o1_alicloud/personal/lvxin/rl_data/aime2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-temperature 0.6
   --eval-top-p 0.95
)


PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator ppo
   --kl-loss-coef 0.00
   --kl-loss-type k1
   --kl-coef 0.00
   --entropy-coef 0.00
   --critic-lr 1e-5
   --num-critic-only-steps 1
   --normalize-advantages
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.999
   --lr-warmup-iters 10
   --lr-decay-iters 1000
   --optimizer-cpu-offload 
   --overlap-cpu-optimizer-d2h-h2d 
   --use-precision-aware-optimizer 
   --use-rollout-logprobs
)

WANDB_ARGS=(
   # --use-wandb
   --wandb-host https://wandb.glm.ai/
   --wandb-team sanshi333
   --wandb-project slime-qwen3-4b
   --wandb-group qwen2.5-32B-dapo
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
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
)

# launch the master node of ray in container
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "MASTER_ADDR": "${MLP_WORKER_0_HOST}",
        "PYTHONPATH": "/root/Megatron-LM",
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
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --calculate-per-token-loss \
   --rollout-num-gpus 8 \
   --load-debug-rollout-data "/mnt/lilei/data/data_1119.pkl" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
#   --load-debug-rollout-data "/mnt/lilei/data/data_1119.pkl" \
