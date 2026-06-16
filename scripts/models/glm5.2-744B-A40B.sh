MOE_ROUTED_EXPERTS=256
MOE_ACTIVE_ROUTED_EXPERTS=8
MOE_SHARED_EXPERTS=1

NHIDDEN=6144
MOE_FFN_HIDDEN=2048
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))
FFN_HIDDEN=12288
N_DENSE_LAYERS=3
N_MOE_LAYERS=75
NHEADS=64

# GLM-5.2 744B-A40B with DSA *cross-layer index sharing*.
# Only the computing layers (layer 1,2,3,7,11,...,75 in Megatron 1-indexing)
# carry indexer weights and compute the sparse top-k; the remaining layers reuse
# the most recent computing layer's indices. The schedule (index_topk_freq=4,
# index_skip_topk_offset=3) is read from the HF config by the shared glm5 provider
# (cross-layer sharing activates automatically when index_topk_freq > 1).
MODEL_ARGS=(
   --spec "slime_plugins.models.glm5.glm5" "get_glm5_spec"
    --moe-layer-freq [0]*$N_DENSE_LAYERS+[1]*$N_MOE_LAYERS
    --num-experts $MOE_ROUTED_EXPERTS
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-topk $MOE_ACTIVE_ROUTED_EXPERTS
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-score-function sigmoid
    --moe-router-pre-softmax
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk-scaling-factor 2.5
    --moe-aux-loss-coeff 0
    --moe-router-dtype fp32
    --make-vocab-size-divisible-by 16
    --num-layers $((N_DENSE_LAYERS + N_MOE_LAYERS))
    --hidden-size $NHIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NHEADS
    --disable-bias-linear
    --swiglu
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --no-position-embedding
    --normalization RMSNorm
    --qk-layernorm
    --multi-latent-attention
    --q-lora-rank 2048
    --kv-lora-rank 512
    --qk-head-dim 192
    --v-head-dim 256
    --kv-channels 192
    --qk-pos-emb-head-dim 64
    --vocab-size 154880
    --rotary-base 8000000
    --enable-experimental

    # DSA + context parallel uses the sequential allgather-CP layout (not zigzag);
    # the index-share provider gathers index_k/kv across the CP group to match.
    --allgather-cp
)
