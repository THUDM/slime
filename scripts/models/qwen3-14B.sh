MODEL_ARGS=(
   --swiglu
   --num-layers 40
   --hidden-size 5120
   --ffn-hidden-size 17408
   --num-attention-heads 40
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
