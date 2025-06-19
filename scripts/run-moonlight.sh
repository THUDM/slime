#!/bin/bash

# 调试时用来重启任务
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

# env
export MASTER_ADDR=${MASTER_PORT:-"127.0.0.1"}
export MASTER_PORT=${MLP_WORKER_0_PORT:-"12345"}
export no_proxy=localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}

export TP_SIZE=4
export PP_SIZE=1
export CP_SIZE=1
export EP_SIZE=4
export ETP_SIZE=2

TARGET_VOCAB=163840

EXP_NAME="moonlight"

MOE_ROUTED_EXPERTS=64
MOE_ACTIVE_ROUTED_EXPERTS=6
MOE_SHARED_EXPERTS=2

NHIDDEN=2048
MOE_FFN_HIDDEN=1408
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
MOE_ROUTER_GROUP_TOPK=1
MOE_ROUTER_NUM_GROUPS=1
MOE_ROUTER_TOPK_SCALING_FACTOR=2.446
FFN_HIDDEN=11264
NLAYERS=27
NHEADS=16
FIRST_K_DENSE_REPLACE=1

SEQ_LEN=8192

# 1) 构造数组
arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

# 2) 将数组用逗号和空格分隔，拼接到方括号里
#    IFS 临时设为 ', '，然后 echo "${arr[*]}"
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# moonlight
MODEL_ARGS=(
    --disable-bias-linear
    --seq-length $SEQ_LEN
    --max-position-embeddings $SEQ_LEN
    --num-layers $NLAYERS
    --hidden-size $NHIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NHEADS
    --kv-channels 128
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-5
    --rotary-percent 1.0
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --vocab-size $TARGET_VOCAB

    --multi-latent-attention
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm
    --rotary-scaling-factor 1
    --rotary-base 50000
    --mscale 1.0
    --mscale-all-dim 1.0
    --attention-softmax-in-fp32
    --no-rope-fusion  # disable rope fusion which is unhappy with rotary-interleaved*

    # moe
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 1e-3
    --moe-router-group-topk $MOE_ROUTER_GROUP_TOPK
    --moe-router-num-groups $MOE_ROUTER_NUM_GROUPS
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor $MOE_ROUTER_TOPK_SCALING_FACTOR
    --moe-token-drop-policy probs
    --moe-router-dtype fp32
    --moe-permute-fusion
)

CKPT_ARGS=(
   --hf-checkpoint /root/Moonlight-16B-A3B-Instruct
   --ref-load /root/Moonlight-16B-A3B-Instruct_torch_dist
)

ROLLOUT_ARGS=(
   --rm-type deepscaler
   --prompt-data /root/deepscaler/deepscaler.jsonl
   --apply-chat-template
   --input-key prompt
   --label-key label
   --num-rollout 3000
   --rollout-batch-size 16
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --rollout-shuffle
   --n-samples-per-prompt 8
   --global-batch-size 128
   --micro-batch-size 1
   --ref-micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9120
   --balance-data
   --partial-rollout
   --sampling-batch-size 64
   --partial-rollout-min-response-length 20
   --partial-rollout-min-tokens 8
   --partial-rollout-mix-ratio 0.25
)

DISTRIBUTED_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size ${ETP_SIZE}
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
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --eps-clip-c 10.0
)

OPTIMIZER_ARGS=(
   --lr 1e-6
   --lr-warmup-iters 320
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
)

WANDB_ARGS=(
   --use-wandb
   --wandb-key ${WANDB_API_KEY}
   --wandb-project slime-dev
   --wandb-group slime-dev
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
   --rollout-num-gpus-per-engine 8 \
   --offload \
   --colocate \
   --sglang-mem-fraction-static 0.5 \
   --sglang-context-length 16384 \
   --sglang-enable-ep-moe \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]}
