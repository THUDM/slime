#!/bin/bash
# Nemotron Nano-30B-A3B (NemotronH hybrid Mamba2+Attn+MoE) — Megatron MODEL_ARGS.
# Authoritative values dumped from mb-nano's nemotron_3_nano_finetune_config provider
# (the exact config our SFT trained with). 52 layers, pattern MEMEM*E..., 128 experts.
MODEL_ARGS=(
   --num-layers 52
   --hidden-size 2688
   --ffn-hidden-size 1856
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 2
   --kv-channels 128
   --normalization RMSNorm
   --position-embedding-type none
   --disable-bias-linear
   --squared-relu
   --untie-embeddings-and-output-weights
   --make-vocab-size-divisible-by 128
   # --- hybrid Mamba/Attn/MLP allocation ---
   --hybrid-override-pattern "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
  
   --mamba-num-heads 64
   --mamba-head-dim 64
   --mamba-state-dim 128
   --mamba-num-groups 8
   # --- MoE ---
   --num-experts 128
   --moe-ffn-hidden-size 1856
   --moe-router-topk 6
   --moe-shared-expert-intermediate-size 3712
   --moe-grouped-gemm
   --moe-router-load-balancing-type seq_aux_loss
   --moe-token-dispatcher-type alltoall
   --transformer-impl local
   --no-persist-layer-norm
   --no-gradient-accumulation-fusion
)
