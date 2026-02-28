MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 24
   --num-query-groups 4
   --kv-channels 256
   --num-layers 64
   --hidden-size 5120
   --ffn-hidden-size 17408
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # qwen3.5 specific
   --attention-output-gate
)
