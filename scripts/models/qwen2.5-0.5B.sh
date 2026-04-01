MODEL_ARGS=(
   --swiglu
   --num-layers 24
   --hidden-size 896
   --ffn-hidden-size 4864
   --num-attention-heads 14
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --group-query-attention
   --num-query-groups 2
   --vocab-size 151936
)