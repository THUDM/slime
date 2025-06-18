MOE_SHARED_EXPERTS=2
MOE_FFN_HIDDEN=1408
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
MOE_ROUTER_TOPK_SCALING_FACTOR=2.446
NLAYERS=27
FIRST_K_DENSE_REPLACE=1

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# moonlight
MODEL_ARGS=(
    --disable-bias-linear
    --num-layers 27
    --hidden-size 2048
    --ffn-hidden-size 11264
    --num-attention-heads 16
    --kv-channels 128
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-5
    --rotary-percent 1.0
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --vocab-size 163840

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
    --no-rope-fusion

    # moe
    --num-experts 64
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-topk 6
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --moe-router-group-topk 1
    --moe-router-num-groups 1
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor $MOE_ROUTER_TOPK_SCALING_FACTOR
    --moe-token-drop-policy probs
    --moe-router-dtype fp32
    --moe-permute-fusion
)