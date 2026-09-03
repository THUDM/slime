# Copyright (c) Kanana LLM Team - Kakao Corp.

EP=4
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash}

FIRST_K_DENSE_REPLACE=1
N_MOE_LAYERS=47

MOE_ROUTED_EXPERTS=128
MOE_ACTIVE_ROUTED_EXPERTS=6
MOE_SHARED_EXPERTS=2

MOE_FFN_HIDDEN=768
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$((MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS))

TOTAL_LAYERS=$((FIRST_K_DENSE_REPLACE + N_MOE_LAYERS))
arr=()
for ((i=0; i<TOTAL_LAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

MODEL_ARGS=(
    --disable-bias-linear
    --num-layers $TOTAL_LAYERS
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 32
    --kv-channels 192
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 128256
    --seq-length 8192
    --max-position-embeddings 8192

    --multi-latent-attention
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm
    --rotary-base 1000000
    --no-rope-fusion

    # moe
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-router-score-function sigmoid
    --moe-router-topk-scaling-factor 2.448
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --moe-grouped-gemm
    # --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
)
