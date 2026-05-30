# Gemma4 26B-A4B MoE model configuration
# Based on google/gemma-4-26B-A4B-it
# 30 layers, 2816 hidden, 16 heads (8 kv), 128 experts top-8
# Features: SWA (window=1024, every 6th layer full attention), gelu_pytorch_tanh

MODEL_ARGS=(
   --spec "slime_plugins.models.gemma4" "get_gemma4_spec"
   # Gemma4 uses GeGLU (gated GELU-tanh), not SwiGLU. Activation is set by
   # get_gemma4_spec; --swiglu is intentionally omitted.
   --num-layers 30
   --hidden-size 2816
   --ffn-hidden-size 2112
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
   # MoE
   --num-experts 128
   --moe-ffn-hidden-size 704
   --moe-router-topk 8
   --moe-router-dtype fp32
   --moe-router-score-function softmax
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
)
