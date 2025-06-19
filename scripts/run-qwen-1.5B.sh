#!/bin/bash

# 调试时用来重启任务
pkill -9 sglang
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python


export PYTHONBUFFERED=16

# network
export MASTER_ADDR=${MASTER_PORT:-"127.0.0.1"}
export MASTER_PORT=${MLP_WORKER_0_PORT:-"12345"}
export no_proxy=localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}

export TP_SIZE=1
export PP_SIZE=1
export CP_SIZE=1

# qwen2.5 1.5B
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 1536
   --ffn-hidden-size 8960
   --num-attention-heads 12
   --max-position-embeddings 32768
   --seq-length 4096
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 10000
   --attention-backend auto
   --group-query-attention
   --num-query-groups 2
   --vocab-size 151936
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --untie-embeddings-and-output-weights
   --attention-dropout 0.0
   --hidden-dropout 0.0
)

CKPT_ARGS=(
   --hf-checkpoint /root/DeepSeek-R1-Distill-Qwen-1.5B/
   --ref-load /root/DeepSeek-R1-Distill-Qwen-1.5B_torch_dist
)

ROLLOUT_ARGS=(
   --rm-type deepscaler
   --prompt-data /root/deepscaler/deepscaler.jsonl
   --apply-chat-template
   --input-key prompt
   --label-key label
   --num-rollout 3000
   --rollout-batch-size 128
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   --rollout-shuffle
   --n-samples-per-prompt 8
   --global-batch-size 1024
   --micro-batch-size 1
   --ref-micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
   --balance-data
   --sampling-batch-size 128
   # --partial-rollout
   # --partial-rollout-min-response-length 20
   # --partial-rollout-min-tokens 8
   # --partial-rollout-mix-ratio 0.5 
   # --over-sampling-filter-path slime.rollout.filter_hub.over_sampling_filters.sort_by_reward_std
   # --over-sampling-filter-input-size 160
   # --diversity-sampling-filter-path slime.rollout.filter_hub.diversity_sampling_filters.check_reward_nonzero_std
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data /root/aime-2024/aime-2024.jsonl
   --eval-max-response-len 16500
   --n-samples-per-eval-prompt 16
)

DISTRIBUTED_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --sequence-parallel
)

PERF_ARGS=(
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-key ${WANDB_API_KEY}
   --wandb-project slime-guagua-1.5B
   --wandb-group slime-guagua-non-partial
)

# launch the master node of ray in container
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "NCCL_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "MASTER_ADDR": "${MASTER_ADDR}",
        "MASTER_PORT": "${MASTER_PORT}",
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_CUDA_INCLUDE_DIR": "/usr/local/cuda/include"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --rollout-num-gpus-per-engine 1 \
   --sglang-mem-fraction-static 0.5 \
   --colocate \
   --offload \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]}