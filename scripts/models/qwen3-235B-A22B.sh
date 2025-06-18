# qwen3-235B-a22B
MODEL_ARGS=(
   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 64
   --num-query-groups 4
   --kv-channels 128
   --num-layers 94
   --hidden-size 4096
   --ffn-hidden-size 12288

   --normalization RMSNorm
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 1.0
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 151936

   --rotary-base 1000000

   # moe
   --moe-ffn-hidden-size 1536
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "'([1]*94)'"
   --num-experts 128
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0
)