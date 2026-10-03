#!/bin/bash
# MODEL_ARGS for nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.
#
# Nemotron-3 Super is a hybrid stack -- 88 layers of Mamba-2, attention and
# *latent* MoE -- so it needs the model provider in
# slime_plugins/models/nemotron_h.py rather than slime's GPTModel branch. That is
# what the --spec line below selects; everything after it is ordinary Megatron
# configuration.
#
# ---------------------------------------------------------------------------
# WHERE EVERY VALUE COMES FROM
#
# Two sources, and they agree:
#
#   1. the HF `config.json` of the checkpoint itself;
#   2. Megatron-LM's own argument bundle for this exact model,
#      examples/post_training/modelopt/conf/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.sh.
#      Nemotron-3 Super is the only Nemotron with release-grade functional test
#      coverage there -- tests/functional_tests/test_cases/nemotron/
#      nemotron3_super_release_* -- so these flags are not a guess.
#
# Deltas from that conf file, all deliberate, all of them because this is an SFT
# bundle rather than a modelopt PTQ run:
#
#   --export-model-type / --save-interval / --micro-batch-size
#                        modelopt / launcher concerns; the launcher owns them.
#   --use-fused-weighted-squared-relu, --cross-entropy-loss-fusion,
#   --cross-entropy-fusion-impl, --enable-experimental
#                        throughput-only. Set USE_FUSION_ARGS=1 to add them back.
#   --moe-aux-loss-coeff  the conf trains with 1e-4. Here it is 0, and
#                        --moe-router-bias-update-rate is 0 too, so the routing
#                        statistics are frozen. A few hundred SFT steps cannot
#                        rebalance 512 experts, so letting the balancer move them
#                        only adds a gradient the run cannot evaluate.
#                        N3S_AUX_LOSS_COEFF / N3S_BIAS_UPDATE_RATE restore it.
#   --trust-remote-code   not needed: nothing loads the HF modelling code. The
#                        tokenizer is a PreTrainedTokenizerFast and the network is
#                        megatron.core's, not `modeling_nemotron_h`.
#
# Every flag here exists in megatron.core 0.16.0rc0. The transformer argument
# group is generated from TransformerConfig's dataclass fields
# (megatron/training/argument_utils.py::ArgumentGroupFactory), so `moe_latent_size`
# and the four `mamba_*` fields are real CLI flags even though they appear
# nowhere in arguments.py.
# ---------------------------------------------------------------------------

# The layer stack. 88 layers: 40 Mamba-2 (M), 40 latent-MoE (E), 8 attention
# (*), read straight off `hybrid_override_pattern` in config.json. The counts
# are corroborated by the safetensors index -- 40 x mixer.A_log, 40 x
# mixer.gate.weight, 8 x mixer.q_proj.weight -- which is how a transcription
# error in this string would have been caught.
#
# `E` is only a legal symbol because megatron.core 0.16 added it to
# ssm/mamba_hybrid_layer_allocation.Symbols; on an older image this pattern is
# rejected outright rather than silently mis-built.
N3S_HYBRID_PATTERN="MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"

MODEL_ARGS=(
   # Two tokens, not a dotted path: --spec is nargs='*' and import_module()
   # unpacks `base_path, name = module_path`.
   --spec "slime_plugins.models.nemotron_h" "get_nemotron_h_spec"

   # -- hybrid stack --------------------------------------------------------
   --is-hybrid-model
   --hybrid-override-pattern "${N3S_HYBRID_PATTERN}"
   # d_inner = mamba_num_heads x mamba_head_dim = 128 x 64 = 8192 = 2 x hidden,
   # i.e. HF's expand=2. mamba_state_dim is HF ssm_state_size, mamba_num_groups
   # is HF n_groups (the *mamba* one -- config.json also has an `n_group: 1`,
   # which is the MoE routing group count and unrelated).
   --mamba-num-heads 128
   --mamba-head-dim 64
   --mamba-state-dim 128
   --mamba-num-groups 8

   # -- shape ---------------------------------------------------------------
   --num-layers 88
   --hidden-size 4096
   --ffn-hidden-size 2688
   --num-attention-heads 32
   --group-query-attention
   # 2 KV heads. NOT a cap of TP=2: megatron requires num_query_groups to be a
   # multiple *or a divisor* of TP (transformer_config.py, "must be a multiple or
   # divisor of"), so TP4/TP8 are legal too and replicate the KV heads. Upstream
   # exercises both on this checkpoint -- the release functional test runs
   # TP2/EP64, Megatron-Bridge's mcore-only SFT runs TP8/EP16.
   --num-query-groups 2
   --kv-channels 128

   --normalization RMSNorm
   --norm-epsilon 1e-5
   # relu^2, not SwiGLU -- which is why the HF checkpoint stores only
   # up_proj/down_proj per expert and no gate_proj, and why linear_fc1 here is
   # NOT the doubled gate+up tensor every other MoE bundle in this tree has.
   --squared-relu
   --disable-bias-linear
   --untie-embeddings-and-output-weights
   # NoPE. Nemotron-3 Super carries no positional embedding at all; config.json
   # still has rope_theta/partial_rotary_factor, but the model does not use
   # them, and Megatron-LM's own bundle says `none`.
   --position-embedding-type none
   --init-method-std 0.014
   --vocab-size 131072
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash

   # -- latent MoE ----------------------------------------------------------
   # The part that only works on a recent megatron.core: each MoE layer projects
   # 4096 -> 1024 (fc1_latent_proj), routes 22 of 512 experts in the 1024-wide
   # latent space, then projects back (fc2_latent_proj). moe_layer.py builds
   # exactly that pair when moe_latent_size is set, under exactly those two
   # names, which is what makes the HF mapping in
   # slime/backends/megatron_utils/hf_to_megatron/nemotron_h.py a rename rather
   # than a reshape.
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
   # Frozen routing statistics for the segment -- see the header.
   --moe-aux-loss-coeff "${N3S_AUX_LOSS_COEFF:-0}"
   --moe-router-bias-update-rate "${N3S_BIAS_UPDATE_RATE:-0}"
)

# Megatron-LM's conf file carries four throughput flags this leaves off by
# default. They are pure fusions -- no numerical intent -- and none of them has
# been exercised on this image. Same reasoning, and the same
# default, as USE_DEEPEP elsewhere: a first failure should be a training problem
# rather than a kernel-selection problem.
if [[ "${USE_FUSION_ARGS:-0}" == "1" ]]; then
   MODEL_ARGS+=(
      --enable-experimental
      --use-fused-weighted-squared-relu
      --cross-entropy-loss-fusion
      --cross-entropy-fusion-impl native
   )
   echo "[model] fusion args enabled (Megatron-LM's conf defaults)"
fi
