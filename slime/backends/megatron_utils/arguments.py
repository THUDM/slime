import ast
import logging

from megatron.training.arguments import parse_args as _megatron_parse_args
from megatron.training.arguments import validate_args as _megatron_validate_args
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from transformers import AutoConfig

__all__ = ["validate_args", "megatron_parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


_ALLGATHER_CP_DSA_ARCHITECTURES = {
    "DeepseekV32ForCausalLM",
    "GlmMoeDsaForCausalLM",
}


def _is_allgather_cp_dsa_model(hf_config):
    if hf_config is None:
        return False

    architecture_names = getattr(hf_config, "architectures", None) or []
    return any(name in _ALLGATHER_CP_DSA_ARCHITECTURES for name in architecture_names)


def _validate_allgather_cp_supported(args, hf_config=None):
    if not getattr(args, "allgather_cp", False) or getattr(args, "context_parallel_size", 1) <= 1:
        return

    if _is_allgather_cp_dsa_model(hf_config):
        return

    raise ValueError(
        "--allgather-cp with --context-parallel-size > 1 is currently only supported for "
        "DSA attention models (DeepSeek-V3.2 and GLM-5.1). Non-DSA models still use the "
        "zigzag CP layout and would silently scramble token order under allgather CP. "
        "Please remove --allgather-cp, set --context-parallel-size 1, or use a supported DSA model."
    )


def _has_dense_moe_layers(args):
    moe_layer_freq = getattr(args, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True

    if isinstance(moe_layer_freq, str):
        try:
            moe_layer_freq = ast.literal_eval(moe_layer_freq)
        except (SyntaxError, ValueError):
            return "0" in moe_layer_freq

    try:
        return any(int(layer_freq) == 0 for layer_freq in moe_layer_freq)
    except TypeError:
        return int(moe_layer_freq) == 0


def _is_moe_config(hf_config):
    return any(
        hasattr(hf_config, attr)
        for attr in (
            "moe_intermediate_size",
            "num_experts",
            "n_routed_experts",
            "num_local_experts",
        )
    )


def validate_args(args):
    """Run megatron's own validate_args plus slime-specific megatron validations."""

    _megatron_validate_args(args)

    # always use varlen
    args.variable_seq_lengths = True
    if getattr(args, "moe_token_dispatcher_type", None) == "allgather":
        logger.info(
            "--moe-token-dispatcher-type allgather does not support variable sequence length, "
            "please use alltoall dispatcher instead."
        )
        args.moe_token_dispatcher_type = "alltoall"

    if args.pipeline_model_parallel_size == 1:
        assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, (
            "decoder_first_pipeline_num_layers and decoder_last_pipeline_num_layers should be None when "
            "pipeline_model_parallel_size is 1."
        )


def _hf_validate_args(args, hf_config):
    def equal(x, y):
        return x == y

    errors = []

    # multimodal models have different config structure
    if hasattr(hf_config, "text_config"):
        hf_config = hf_config.text_config

    # Some models store rope_theta inside rope_parameters dict rather than
    # as a top-level attribute.  Prefer the dict value when available so
    # the validation doesn't compare against a stale class default.
    rope_params = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        _hf_rope_theta = rope_params["rope_theta"]
    else:
        _hf_rope_theta = getattr(hf_config, "rope_theta", None)

    validate_dense_ffn = not _is_moe_config(hf_config) or _has_dense_moe_layers(args)

    for hf_config_name, megatron_config_name, compare_fn in [
        ("hidden_size", "hidden_size", equal),
        ("num_attention_heads", "num_attention_heads", equal),
        ("num_hidden_layers", "num_layers", equal),
        ("intermediate_size", "ffn_hidden_size", equal),
        ("moe_intermediate_size", "moe_ffn_hidden_size", equal),
        ("shared_expert_intermediate_size", "moe_shared_expert_intermediate_size", equal),
        ("tie_word_embeddings", "untie_embeddings_and_output_weights", lambda x, y: not x == y),
        ("rms_norm_eps", "norm_epsilon", equal),
        ("rms_norm_eps", "layernorm_epsilon", equal),
    ]:
        if hf_config_name == "intermediate_size" and not validate_dense_ffn:
            continue

        if hasattr(hf_config, hf_config_name) and hasattr(args, megatron_config_name):
            if not compare_fn(getattr(hf_config, hf_config_name), getattr(args, megatron_config_name)):
                errors.append(
                    f"{hf_config_name} in hf config {getattr(hf_config, hf_config_name)} is not equal to "
                    f"{megatron_config_name} {getattr(args, megatron_config_name)}, please check the config."
                )

    # Validate rope_theta separately using the resolved value
    if _hf_rope_theta is not None:
        if not equal(_hf_rope_theta, getattr(args, "rotary_base", None)):
            errors.append(
                f"rope_theta in hf config {_hf_rope_theta} is not equal to "
                f"rotary_base {getattr(args, 'rotary_base', None)}, please check the config."
            )

    if len(errors) > 0:
        raise AssertionError("hf_validate_args failed: " + "; ".join(errors))


def _set_default_megatron_args(args):
    # always use zero optimizer
    args.use_distributed_optimizer = True
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # TODO: revisit this when megatron(dev) have solved the optimizer-cpu-offload ckpt saving bug
    args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    elif not args.tokenizer_model:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
    return args


# Public alias for external tools (e.g. convert_hf_to_torch_dist.py)
set_default_megatron_args = _set_default_megatron_args


def megatron_parse_args(extra_args_provider, skip_hf_validate=False):
    """Parse megatron args, validate HF config, and set defaults."""
    args = _megatron_parse_args(extra_args_provider=extra_args_provider, ignore_unknown_args=True)

    hf_config = None
    if args.hf_checkpoint and not skip_hf_validate:
        hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        _hf_validate_args(args, hf_config)

    if not skip_hf_validate:
        _validate_allgather_cp_supported(args, hf_config)

    args.rank = 0
    args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    args = _set_default_megatron_args(args)
    return args
