import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class FSDPArgs:
    # Optim
    optimizer: str = "adam"
    lr: float = 2e-5
    lr_decay_style: str = "constant"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.03

    # FSDP specific
    fsdp_wrap: str = "transformer_blocks"  # future use: auto wrap policy
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False
    fsdp_limit_all_gathers: bool = False
    fsdp_sync_module_states: bool = True
    fsdp_forward_prefetch: bool = True
    fsdp_backward_prefetch: bool = True

    # Logging
    wandb_project: str = "slime-fsdp"
    wandb_run_name: Optional[str] = None

    # Precision
    gradient_checkpointing: bool = False

    # YAML bookkeeping
    config: Optional[str] = None


def parse_fsdp_cli(extra_args_provider=None):
    parser = argparse.ArgumentParser("FSDP SFT Training (slime)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    for f in dataclasses.fields(FSDPArgs):
        if f.name == "config":
            continue
        arg_type = f.type if f.type != Optional[str] else str
        if arg_type is bool:
            parser.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=arg_type, default=f.default)

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    args = parser.parse_args()
    return args


def load_fsdp_args(extra_args_provider=None):
    args = parse_fsdp_cli(extra_args_provider)
    if args.config:
        with open(args.config, "r") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    return args
