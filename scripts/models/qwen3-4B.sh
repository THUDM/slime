MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   # Original: --vocab-size 151936
   # Fixed: Qwen3-4B tokenizer max token ID is 151668 (</think>)
   # Megatron best practice: round up to multiple of 128 for performance
   # 151668 -> 151936 (original) -> 152064 (rounded to 128)
   # --vocab-size 152064
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)