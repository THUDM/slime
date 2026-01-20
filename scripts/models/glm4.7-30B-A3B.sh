MOE_SHARED_EXPERTS=1

MOE_FFN_HIDDEN=1536
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$((MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS))
N_DENSE_LAYERS=1
N_MOE_LAYERS=46

MODEL_ARGS=(
    --moe-layer-freq [0]*$N_DENSE_LAYERS+[1]*$N_MOE_LAYERS
    --num-experts 64
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-topk 4
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk-scaling-factor 1.8
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --num-layers $((N_DENSE_LAYERS + N_MOE_LAYERS))
    --hidden-size 2048
    --ffn-hidden-size 10240
    --num-attention-heads 20
    --disable-bias-linear
    --add-qkv-bias
    --swiglu
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --no-position-embedding
    --normalization RMSNorm
    --qk-layernorm
    --multi-latent-attention
    --q-lora-rank 768
    --kv-lora-rank 512
    --qk-head-dim 192
    --v-head-dim 256
    --kv-channels 192
    --qk-pos-emb-head-dim 64
    --vocab-size 154880
    --rotary-base 1000000
    --enable-experimental
)
