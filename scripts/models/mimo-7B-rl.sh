MODEL_ARGS=(
    --swiglu
    --num-layers 36
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --use-rotary-position-embeddings
    --disable-bias-linear
    --add-qkv-bias
    --normalization "RMSNorm"
    --norm-epsilon 1e-05
    --rotary-base 640000
    --vocab-size 151680
    --untie-embeddings-and-output-weights
    --max-position-embeddings 32768
    --mtp-num-layers 1
)
