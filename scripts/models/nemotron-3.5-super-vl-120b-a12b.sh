#!/bin/bash
# MODEL_ARGS for nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B-SourceOfTruth.
#
# This is nemotron-3-super-120b-a12b.sh with the --spec line retargeted and two
# flags added. Nothing else moves, and that is a measurement rather than a hope:
#
#   * the 88-layer hybrid pattern below is character-for-character equal to
#     N3S_HYBRID_PATTERN in the Nemotron-3 bundle. It is NOT read from
#     `hybrid_override_pattern`, which this config.json does not have -- it is
#     `llm_config.layers_block_type`, an 88-element list of "mamba"/"moe"/
#     "attention", mapped M/E/*. Corroborated by the safetensors index the same
#     way the Nemotron-3 one was: 40 x mixer.A_log, 40 x mixer.gate.weight,
#     8 x mixer.q_proj.weight;
#   * of every key the two language configs share, exactly one value differs,
#     and it is `eos_token_id` (11 here, 2 there). No flag below reads it;
#   * all 42,683 of Nemotron-3's weight names appear verbatim in this
#     checkpoint under one `language_model.` prefix. The other 395 tensors are
#     the vision half.
#
# Re-check all three with §11 of
# Scripts-Slime/sft_vlm_geo3k_nemotron3.5/docs/sft_vlm_geo3k_nemotron3.5_development_plan.md.
#
# ---------------------------------------------------------------------------
# THE TWO DELTAS
#
#   --spec ... nemotron_35_super_vl    builds the same MambaModel *inside* a
#                       wrapper that owns the RADIO tower and the mlp1
#                       projector, and scatters image features onto the
#                       <image> placeholders. The language half is not
#                       reimplemented: that plugin calls the Nemotron-3 one.
#
#   --trust-remote-code the Nemotron-3 bundle switches this off and says why
#                       ("nothing loads the HF modelling code"). Here the
#                       vision tower IS the HF modelling code: modeling_radio.py
#                       and modeling_nemotron_h_omni.py ship in the checkpoint
#                       directory, and the processor is remote code too.
#
# There is deliberately no vision argument group. Every vision shape --
# 32 layers, d=1280, patch 16, 3 cls + 7 register tokens, downsample_ratio 0.5,
# projector 5120 -> 20480 -> 4096 -- is read by the plugin from the HF config at
# build time. Restating any of it here would create a second source of truth for
# numbers that must agree with the checkpoint, which is the one kind of drift a
# bundle file cannot catch.
# ---------------------------------------------------------------------------

# Identical to the Nemotron-3 bundle's; see the header for how that was checked.
N35VL_HYBRID_PATTERN="MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"

MODEL_ARGS=(
   # Two tokens, not a dotted path: --spec is nargs='*' and import_module()
   # unpacks `base_path, name = module_path`.
   --spec "slime_plugins.models.nemotron_35_super_vl" "get_nemotron_35_super_vl_model_provider"

   # The vision tower, the projector and the processor are all remote code.
   --trust-remote-code

   # -- hybrid stack --------------------------------------------------------
   --is-hybrid-model
   --hybrid-override-pattern "${N35VL_HYBRID_PATTERN}"
   # d_inner = mamba_num_heads x mamba_head_dim = 128 x 64 = 8192 = 2 x hidden,
   # i.e. HF's expand=2. mamba_state_dim is HF ssm_state_size, mamba_num_groups
   # is HF n_groups (the *mamba* one -- llm_config also has an `n_group: 1`,
   # which is the MoE routing group count and unrelated).
   --mamba-num-heads 128
   --mamba-head-dim 64
   --mamba-state-dim 128
   --mamba-num-groups 8

   # -- shape ---------------------------------------------------------------
   # 88 = len(llm_config.layers_block_type). This config has no
   # `num_hidden_layers`, unlike Nemotron-3's.
   --num-layers 88
   --hidden-size 4096
   --ffn-hidden-size 2688
   --num-attention-heads 32
   --group-query-attention
   # 2 KV heads. NOT a cap of TP=2: megatron requires num_query_groups to be a
   # multiple *or a divisor* of TP, so TP4/TP8 are legal too and replicate the
   # KV heads. TP2 is what this recipe runs.
   --num-query-groups 2
   --kv-channels 128

   --normalization RMSNorm
   # `layer_norm_epsilon` here; Nemotron-3 spells the same 1e-5 `norm_eps`.
   --norm-epsilon 1e-5
   # relu^2, not SwiGLU -- which is why the HF checkpoint stores only
   # up_proj/down_proj per expert and no gate_proj, and why linear_fc1 is NOT
   # the doubled gate+up tensor every other MoE bundle in this tree has.
   --squared-relu
   --disable-bias-linear
   --untie-embeddings-and-output-weights
   # NoPE, and here it is a statement of the config rather than an override of
   # it: unlike Nemotron-3's, this llm_config carries no rope_theta and no
   # partial_rotary_factor at all.
   --position-embedding-type none
   --init-method-std 0.014
   --vocab-size 131072
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash

   # -- latent MoE ----------------------------------------------------------
   # Each MoE layer projects 4096 -> 1024 (fc1_latent_proj), routes 22 of 512
   # experts in the 1024-wide latent space, then projects back
   # (fc2_latent_proj). moe_layer.py builds exactly that pair when
   # moe_latent_size is set, under exactly those two names, which is what makes
   # the HF mapping in hf_to_megatron/nemotron_h.py a rename rather than a
   # reshape.
   --num-experts 512
   --moe-latent-size 1024
   --moe-ffn-hidden-size 2688
   --moe-router-topk 22
   --moe-router-score-function sigmoid
   --moe-router-enable-expert-bias
   --moe-router-load-balancing-type seq_aux_loss
   --moe-router-topk-scaling-factor 5.0
   --moe-router-dtype fp32
   --moe-shared-expert-intermediate-size 5376
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
   --moe-permute-fusion
   # Frozen routing statistics for the segment: a few hundred SFT steps cannot
   # rebalance 512 experts, so letting the balancer move them only adds a
   # gradient the run cannot evaluate. Same defaults as the Nemotron-3 bundle.
   --moe-aux-loss-coeff "${N35VL_AUX_LOSS_COEFF:-0}"
   --moe-router-bias-update-rate "${N35VL_BIAS_UPDATE_RATE:-0}"
)

# MTP is deliberately absent. `mtp_layers_block_type: ["attention", "moe"]` and
# `num_nextn_predict_layers: 1` are in the config and the 1,024 MTP expert
# tensors are in the checkpoint, but MambaModel has no `mtp_block_spec` -- it is
# GPTModel that has one -- so the block is not built, its weights are never
# requested, and hf_to_megatron/nemotron_h.py never reads them. Adding it means
# porting get_gpt_mtp_block_spec onto the hybrid stack and handling
# "1 MTP depth = 2 layers", which is where Megatron-Bridge hit an aux-loss
# tracker out-of-bounds crash. For SFT the head contributes nothing.

# Megatron-LM's conf file for the Nemotron-3 sibling carries four throughput
# flags this leaves off by default. They are pure fusions -- no numerical intent
# -- and none has been exercised against the vision path. Same reasoning, and
# the same default, as USE_DEEPEP elsewhere: a first failure should be a
# training problem rather than a kernel-selection problem.
if [[ "${USE_FUSION_ARGS:-0}" == "1" ]]; then
   MODEL_ARGS+=(
      --enable-experimental
      --use-fused-weighted-squared-relu
      --cross-entropy-loss-fusion
      --cross-entropy-fusion-impl native
   )
   echo "[model] fusion args enabled (Megatron-LM's conf defaults)"
fi
