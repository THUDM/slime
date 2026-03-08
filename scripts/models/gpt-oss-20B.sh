#!/bin/bash

# GPT-OSS 20B model configuration
# Based on openai/gpt-oss-20b
# 24 layers, 2880 hidden, 64 heads (8 kv), 32 experts top-4, all MoE
# Features: learnable softmax, SWA (window=128, skip_freq=2), quick GeGLU

NLAYERS=24

MODEL_ARGS=(
    --spec "slime_plugins.models.gpt_oss" "get_gpt_oss_spec"
    --num-layers ${NLAYERS}
    --hidden-size 2880
    --ffn-hidden-size 2880
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --kv-channels 64
    --seq-length 4096
    --max-position-embeddings 131072
    --padded-vocab-size 201088
    --make-vocab-size-divisible-by 128
    --tokenizer-type HuggingFaceTokenizer
    --bf16
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-rope-fusion
    --no-bias-gelu-fusion
    --no-bias-dropout-fusion
    --use-mcore-models
    --rotary-percent 1.0
    --rotary-base 150000
    --position-embedding-type rope
    --use-rope-scaling
    --rope-scaling-factor 32
    --sequence-parallel
    # MoE
    --num-experts 32
    --moe-ffn-hidden-size 2880
    --moe-router-topk 4
    --moe-router-dtype fp32
    --moe-router-score-function softmax
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    # GPT-OSS specific
    --quick-geglu
    --glu-linear-offset 1.0
    --softmax-type learnable
    --window-attn-skip-freq 2
    --window-size 128,0
    --activation-func-clamp-value 7.0
)
