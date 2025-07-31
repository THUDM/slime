N_DENSE_LAYERS=1
N_MOE_LAYERS=45

# glm4.5-106B-A12B
MODEL_ARGS=(
    --disable-bias-linear
    --group-query-attention
    --num-attention-heads 96
    --num-query-groups 8
    --kv-channels 128
    --num-layers $((N_DENSE_LAYERS + N_MOE_LAYERS))
    --hidden-size 4096
    --ffn-hidden-size 10944

    --add-qkv-bias
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-percent 0.5
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 151552
    --rotary-base 1000000

    # moe
    --moe-ffn-hidden-size 1408
    --moe-shared-expert-intermediate-size 1408
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-router-topk 8
    --moe-layer-freq [0]*$N_DENSE_LAYERS+[1]*$N_MOE_LAYERS
    --num-experts 128
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-aux-loss-coeff 0
)