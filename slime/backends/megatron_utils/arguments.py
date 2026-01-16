from megatron.training.arguments import parse_args, validate_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]


def set_default_megatron_args(args):
    # always use zero optimizer
    args.use_distributed_optimizer = True
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        print(f"--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"

    # === Off-Policy GRPO: M2PO Filtering Defaults ===
    # M2PO (Multi-step Off-Policy Optimization) filters tokens with low importance weights
    # to reduce gradient variance from stale samples
    if not hasattr(args, "enable_m2po_filtering"):
        args.enable_m2po_filtering = False  # Default: disabled for backward compatibility

    if not hasattr(args, "m2po_threshold"):
        args.m2po_threshold = 0.1  # Filter tokens where π_θ/π_prox < 0.1

    # === Off-Policy GRPO: Proximal Logprob Approximation Defaults ===
    # Approximation methods can skip expensive forward passes through proximal policy
    if not hasattr(args, "use_proximal_logp_approximation"):
        args.use_proximal_logp_approximation = False  # Default: use full recomputation

    if not hasattr(args, "prox_logp_method"):
        # Options: "recompute" (full forward), "old_actor" (use old_actor as proxy),
        #          "linear" (interpolate between rollout and old_actor)
        args.prox_logp_method = "recompute"

    if not hasattr(args, "prox_logp_alpha"):
        args.prox_logp_alpha = 0.5  # Interpolation weight for linear approximation

    return args
