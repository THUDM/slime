MODEL_ARGS=(
   --swiglu
   --num-layers 64
   --hidden-size 5120
   --ffn-hidden-size 25600
   --num-attention-heads 64
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
   --untie-embeddings-and-output-weights
)
