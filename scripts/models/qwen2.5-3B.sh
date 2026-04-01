MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2048
   --ffn-hidden-size 11008
   --num-attention-heads 16
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
