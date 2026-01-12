# qwen3-235B-a22B
NLAYERS=94
FIRST_K_DENSE_REPLACE=0

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

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

   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"

   # moe
   --moe-ffn-hidden-size 1536
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq $MOE_LAYER_FREQ
   --num-experts 128
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0
)