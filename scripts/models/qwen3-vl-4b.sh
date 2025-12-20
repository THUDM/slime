MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --num-query-groups 8
   --init-method-std 0.02
   --norm-epsilon 1e-06
   --rotary-base 5000000
   --vocab-size 151936
   --seq-length 262144
   --use-rotary-position-embeddings
   --normalization "RMSNorm"
   --qk-layernorm
   --group-query-attention
   --disable-bias-linear
   --kv-channels 128
)