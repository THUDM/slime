import argparse
import os
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"true", "1", "yes", "y", "on"}


def _non_negative_int(value: Any) -> int:
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _positive_int(value: Any) -> int:
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _positive_float(value: Any) -> float:
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def add_megatron_server_arguments(parser):
    group = parser.add_argument_group("megatron server")
    group.add_argument(
        "--teacher-port",
        type=_positive_int,
        default=_positive_int(os.getenv("TEACHER_PORT", "7999")),
        help="HTTP port for the Megatron teacher server.",
    )
    group.add_argument(
        "--teacher-warmup-port",
        type=_positive_int,
        default=_positive_int(os.getenv("TEACHER_WARMUP_PORT", "7999")),
        help="Temporary HTTP port used by the Megatron teacher warmup server.",
    )
    group.add_argument(
        "--teacher-warmup-timeout-s",
        type=_positive_int,
        default=_positive_int(os.getenv("TEACHER_WARMUP_TIMEOUT_S", "3000")),
        help="Timeout in seconds for the Megatron teacher warmup request.",
    )
    group.add_argument(
        "--teacher-sample-reduction-chunk-size",
        type=_positive_int,
        default=4096,
        help="Row chunk size used while sampling from TP-sharded teacher logits.",
    )
    group.add_argument(
        "--teacher-label-reduction-chunk-size",
        type=_positive_int,
        default=4096,
        help="Row chunk size used while gathering label-token logprobs from TP-sharded teacher logits.",
    )
    group.add_argument(
        "--megatron-server-max-length",
        type=_non_negative_int,
        default=_non_negative_int(os.getenv("MEGATRON_SERVER_MAX_LENGTH", "0")),
        help="Reject teacher requests longer than this many tokens. Set 0 to disable.",
    )
    group.add_argument(
        "--megatron-server-update-timeout-s",
        type=_positive_float,
        default=_positive_float(os.getenv("MEGATRON_SERVER_UPDATE_TIMEOUT_S", "3600")),
        help="Default timeout in seconds for /update_from_disk when the request does not override it.",
    )
    group.add_argument(
        "--megatron-server-warmup",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("MEGATRON_SERVER_WARMUP", True),
        help="Whether to run a local warmup request before serving traffic.",
    )
    return parser


def configure_megatron_server_args(args):
    args.debug_train_only = True
    args.use_kl_loss = False
    args.offload_train = False
    args.use_dynamic_batch_size = False
    args.use_wandb = False
    args.kl_coef = 0
    args.use_opd = False
    args.use_critic = False
    args.keep_old_actor = False
    args.no_load_optim = True
    args.no_load_rng = True
    # Keep this as a list (not str), otherwise freeze logic iterates over characters.
    args.only_train_params_name_list = ["nothing_to_train"]
    return args


def validate_megatron_server_args(args):
    positive_fields = [
        "teacher_port",
        "teacher_warmup_port",
        "teacher_warmup_timeout_s",
        "teacher_sample_reduction_chunk_size",
        "teacher_label_reduction_chunk_size",
        "megatron_server_update_timeout_s",
    ]
    non_negative_fields = [
        "megatron_server_max_length",
    ]

    for name in positive_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    for name in non_negative_fields:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be >= 0")

    if args.only_train_params_name_list != ["nothing_to_train"]:
        raise ValueError("Megatron server must not train any parameters.")
    if not args.debug_train_only:
        raise ValueError("Megatron server requires debug_train_only=True.")
    if args.use_kl_loss or args.use_opd or args.use_critic:
        raise ValueError("Megatron server only supports teacher logprob prefill mode.")

    return args
