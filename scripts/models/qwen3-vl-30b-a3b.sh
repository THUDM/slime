NLAYERS=48
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
   --swiglu
   --num-layers 48
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 32
   --num-query-groups 4
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
   --untie-embeddings-and-output-weights

   # moe
   --moe-ffn-hidden-size 768
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