MODEL_ARGS=(
   --spec "slime_plugins.models.gemma4" "get_gemma4_spec"
   --custom-model-provider-path "slime_plugins.models.gemma4_provider.model_provider"
   --num-layers 48
   --hidden-size 3840
   --ffn-hidden-size 15360
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 8
   --kv-channels 256
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 10000
   --rotary-percent 1.0
   --vocab-size 262144
   --qk-layernorm
)
