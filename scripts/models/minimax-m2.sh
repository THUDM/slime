# MiniMax-M2.5 (229B, 62 layers, 256 experts, top-8)
MODEL_ARGS=(
    --spec "slime_plugins.models.minimax_m2" "get_minimax_m2_layer_spec"
    --disable-bias-linear
    --num-layers 62
    --hidden-size 3072
    --ffn-hidden-size 1536
    --num-attention-heads 48
    --kv-channels 128
    --num-query-groups 8
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 200064
    --group-query-attention

    --rotary-percent 0.5
    --rotary-base 5000000
    --qk-layernorm
    --no-rope-fusion
    --attention-softmax-in-fp32

    # MoE
    --num-experts 256
    --moe-ffn-hidden-size 1536
    --moe-router-topk 8
    --moe-layer-freq "[1]*62"
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-load-balancing-type none
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 0
    --moe-grouped-gemm
    --moe-permute-fusion
)
