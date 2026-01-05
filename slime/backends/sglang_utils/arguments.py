import sglang
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from slime.utils.http_utils import _wrap_ipv6


# TODO: use all sglang router arguments with `--sglang-router` prefix
def add_sglang_router_arguments(parser):
    """
    Add arguments to the parser for the SGLang router.
    """
    parser.add_argument(
        "--sglang-router-ip",
        type=str,
        default=None,
        help="IP address of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-port",
        type=int,
        default=None,
        help="Port of the SGLang router",
    )
    parser.add_argument(
        "--sglang-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the SGLang router in seconds",
    )
    return parser


def add_sglang_arguments(parser):
    """
    Add arguments to the parser for the SGLang server.
    """
    parser = add_sglang_router_arguments(parser)
    parser.add_argument("--sglang-server-concurrency", type=int, default=512)

    old_add_argument = parser.add_argument

    skipped_args = [
        "model_path",
        "trust_remote_code",
        "random_seed",
        # memory
        "enable_memory_saver",
        # distributed
        "tp_size",
        "port",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "gpu_id_step",
        "base_gpu_id",
        "nccl_port",
        "skip_server_warmup",
        "enable_return_routed_experts",
    ]

    def new_add_argument_wrapper(*name_or_flags, **kwargs):
        """
        Add arguments to the parser, ensuring that the server arguments are prefixed and skippable.
        """
        # Determine the canonical name for skip check (e.g., "model_path")
        canonical_name_for_skip_check = None
        if "dest" in kwargs:
            canonical_name_for_skip_check = kwargs["dest"]
        else:
            for flag_name_candidate in name_or_flags:
                if isinstance(flag_name_candidate, str) and flag_name_candidate.startswith("--"):
                    # Derive from first long flag: --foo-bar -> foo_bar
                    stem = flag_name_candidate[2:]
                    canonical_name_for_skip_check = stem.replace("-", "_")
                    break
            # If no long flag and no dest, skip logic might not catch it unless short flags imply a dest.

        if canonical_name_for_skip_check and canonical_name_for_skip_check in skipped_args:
            return  # Skip this entire argument definition

        # If not skipped, proceed to prefix flags and dest
        new_name_or_flags_list = []
        for item_flag in name_or_flags:
            if isinstance(item_flag, str) and item_flag.startswith("-"):
                original_flag_stem = item_flag.lstrip("-")  # "foo-bar" from "--foo-bar", or "f" from "-f"
                prefixed_item = f"--sglang-{original_flag_stem}"
                new_name_or_flags_list.append(prefixed_item)
            else:
                # Positional arguments or non-string items
                new_name_or_flags_list.append(item_flag)

        # Prepare kwargs for the actual add_argument call.
        # Make a copy to avoid modifying the original kwargs dict.
        final_kwargs = kwargs.copy()

        # If 'dest' is explicitly provided and is a string, prefix it.
        # This ensures the attribute on the args namespace becomes, e.g., args.sglang_dest_name.
        if "dest" in final_kwargs and isinstance(final_kwargs["dest"], str):
            original_dest = final_kwargs["dest"]
            # Avoid double prefixing if dest somehow already starts with sglang_
            if not original_dest.startswith("sglang_"):
                final_kwargs["dest"] = f"sglang_{original_dest}"
        # If 'dest' is not explicitly provided (or is None/not a string),
        # argparse will derive 'dest' from the (now prefixed) flag names.
        # E.g., if the first flag is "--sglang-foo-bar", argparse sets dest to "sglang_foo_bar".

        old_add_argument(*new_name_or_flags_list, **final_kwargs)

    parser.add_argument = new_add_argument_wrapper
    ServerArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument

    return parser


def validate_args(args):
    if parse(sglang.__version__) == parse("0.4.10") and getattr(args, "sglang_enable_ep_moe", False):
        args.sglang_expert_parallel_size = args.rollout_num_gpus_per_engine

    args.sglang_tp_size = args.rollout_num_gpus_per_engine
    args.sglang_dp_size = args.sglang_data_parallel_size
    args.sglang_pp_size = args.sglang_pipeline_parallel_size
    args.sglang_ep_size = args.sglang_expert_parallel_size

    if args.sglang_dp_size > 1:
        assert args.sglang_enable_dp_attention

    if getattr(args, "sglang_router_ip", None):
        args.sglang_router_ip = _wrap_ipv6(args.sglang_router_ip)
