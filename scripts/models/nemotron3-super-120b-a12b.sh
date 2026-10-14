# NVIDIA Nemotron 3 Super 120B-A12B (BF16), for MCore's HybridModel.
#
# Every value below is config.json from nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16,
# translated into the MCore flag that carries it. The HF field name is in the
# comment wherever the two differ.
#
# Unlike the other files in this directory this model is not a GPTModel: the 88
# layers are a hybrid of Mamba2, MoE and attention, chosen per layer by
# --hybrid-override-pattern. --spec therefore points at a *model provider*
# rather than a layer spec, and slime calls it directly
# (backends/megatron_utils/model_provider.py:146-153).
#
# Most of the flags below are not declared in megatron/training/arguments.py --
# they are generated from TransformerConfig dataclass fields by
# megatron/training/argument_utils.py, the same mechanism that produces
# --moe-flex-dispatcher-backend. Verified present on mcore 0.20 (f6c33bde4).

# 'M' Mamba2 x40, 'E' MoE x40, '*' attention x8, in this order.
NEMOTRON_H_PATTERN="MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"

MODEL_ARGS=(
   --spec "slime_plugins.models.nemotron_h" "get_nemotron_h_model_provider"
   --hybrid-override-pattern "${NEMOTRON_H_PATTERN}"

   # ---------------------------------------------------------------- shape --
   --num-layers 88                         # num_hidden_layers
   --hidden-size 4096                      # hidden_size
   --num-attention-heads 32                # num_attention_heads
   --num-query-groups 2                    # num_key_value_heads (GQA)
   --kv-channels 128                       # head_dim
   --group-query-attention
   --vocab-size 131072                     # vocab_size
   --untie-embeddings-and-output-weights   # tie_word_embeddings: false
   --disable-bias-linear                   # use_bias / attention_bias / mlp_bias: false
   --normalization RMSNorm
   --norm-epsilon 1e-5                     # norm_eps / layer_norm_epsilon
   # NoPE. config.json still carries rope_theta 10000 and
   # partial_rotary_factor 1.0, but nothing reads them: SGLang's nemotron_h.py
   # has no rotary embedding at all -- NemotronHAttention.forward goes
   # qkv_proj -> RadixAttention -> o_proj and never sees positions -- and
   # neither does the checkpoint's own modeling_nemotron_h.py. Configuring
   # `rope` here made the trainer a different model from the engine and put
   # train_rollout_logprob_abs_diff at 2.3 against slime's 0.1 bound, with
   # entropy_loss at 0.75 against 0.16.
   --position-embedding-type none

   # ---------------------------------------------------------------- mamba --
   # in_proj packs [z, x, B, C, dt]: 2*d_inner + 2*n_groups*d_state + nheads
   #   d_inner = expand(2) x hidden(4096)        = 8192
   #           = mamba_num_heads(128) x head_dim(64)
   #   conv_dim = d_inner + 2*n_groups*d_state   = 10240
   --mamba-num-heads 128                   # mamba_num_heads
   --mamba-head-dim 64                     # mamba_head_dim
   --mamba-state-dim 128                   # ssm_state_size
   --mamba-num-groups 8                    # n_groups

   # ------------------------------------------------------------------ moe --
   # The experts run inside a 1024-wide latent space, not on the 4096 residual:
   # fc1_latent_proj 4096->1024, expert 1024->2688->1024, fc2_latent_proj
   # 1024->4096. MoELayer builds those two projections under exactly the names
   # the HF checkpoint uses.
   --num-experts 512                       # n_routed_experts
   --moe-latent-size 1024                  # moe_latent_size
   --moe-ffn-hidden-size 2688              # moe_intermediate_size
   --moe-router-topk 22                    # num_experts_per_tok
   # DeepSeek-V3-style aux-loss-free routing, NOT softmax. Three independent
   # witnesses agree on sigmoid, so this is the model's routing rather than the
   # value that happens to silence the check:
   #
   #   * the checkpoint carries 41 gate.e_score_correction_bias tensors (40 MoE
   #     layers + the MTP layer); that tensor only exists for aux-loss-free
   #     routing;
   #   * SGLang scores this architecture with scoring_func="sigmoid"
   #     (sglang/srt/models/nemotron_h.py:213), alongside
   #     routing_method_type=DeepSeekV3 and the same correction bias;
   #   * MCore refuses the combination outright at transformer_config.py:2886 --
   #       ValueError: Expert bias for aux-loss-free routing only supports
   #                   'sigmoid' and 'sqrtsoftplus' score functions
   #     which is exactly what a softmax configuration trips on launch.
   #
   # norm_topk_prob: true needs no flag of its own -- MCore renormalises the
   # top-k scores whenever topk > 1 (moe_utils.py:932), and topk is 22 here,
   # which matches SGLang's renormalize=config.norm_topk_prob.
   --moe-router-score-function sigmoid
   --moe-router-enable-expert-bias         # gate.e_score_correction_bias exists
   --moe-router-topk-scaling-factor 5.0    # routed_scaling_factor
   # n_group and topk_group are both 1, so group-limited routing is a no-op.
   # Leaving --moe-router-num-groups / --moe-router-group-topk unset is the
   # equivalent configuration, not an omission.
   --moe-shared-expert-intermediate-size 5376   # moe_shared_expert_intermediate_size
   --moe-grouped-gemm
   --moe-token-dispatcher-type alltoall
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0

   # ------------------------------------------------------------- non-gated --
   # mlp_hidden_act: relu2. The checkpoint has up_proj and down_proj and no
   # gate_proj, so linear_fc1 is up_proj alone rather than cat(gate, up), and
   # the activation is squared ReLU. MCore asserts these two agree
   # (transformer_config.py:1856-1860).
   --squared-relu
   # (gated_linear_unit stays at its False default; --swiglu is what would set it)

   # --------------------------------------------------------------- context --
   # max_position_embeddings is 262144. That is the model's capability, not a
   # per-run cost here -- SFT sequences are far shorter -- but it sizes RoPE.
   --max-position-embeddings 262144
)
