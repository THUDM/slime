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

export PYTHONBUFFERED=16

# DeepSeek-R1-Distill-Qwen-7B
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 3584
   --ffn-hidden-size 18944
   --num-attention-heads 28
   --group-query-attention
   --num-query-groups 4
   --max-position-embeddings 131072
   --seq-length 4096
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-06
   --rotary-base 10000
   --vocab-size 152064
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --untie-embeddings-and-output-weights
   --attention-dropout 0.0
   --hidden-dropout 0.0
)

CKPT_ARGS=(
   --hf-checkpoint /root/DeepSeek-R1-Distill-Qwen-7B
   --ref-load /root/DeepSeek-R1-Distill-Qwen-7B_torch_dist
   --save-interval 100
   --save /root/DeepSeek-R1-Distill-Qwen-7B_slime
)

ROLLOUT_ARGS=(
   --rollout-function-path slime_plugins.rollout_buffer.rollout_buffer_example.generate_rollout
   --rm-type deepscaler
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --num-rollout 3000
   --rollout-batch-size 128
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   --rollout-shuffle
   --n-samples-per-prompt 8
   --global-batch-size 1024
   --micro-batch-size 8
   --ref-micro-batch-size 8
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
   --balance-data
)

DISTRIBUTED_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
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
   # --use-wandb
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "0"
     }
   }' \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --rollout-num-gpus-per-engine 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   --rollout-buffer-url http://${MASTER_ADDR}:8889 \
   --keep-old-actor \
   --disable-rewards-normalization \
   --loss-mask-type distill_qwen \
   --log-passrate
