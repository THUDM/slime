import argparse
import dataclasses
from dataclasses import dataclass

import yaml


@dataclass
class FSDPArgs:
    # Optim
    optimizer: str = "adam"  # Options: "adam" (GPU-based AdamW), "deepspeed_cpu_adam" (CPU-offloaded optimizer states)
    lr: float = 2e-5
    lr_decay_style: str = "constant"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.03

    attn_implementation: str = "flash_attention_2"

    # Logging
    wandb_project: str = "slime-fsdp"
    wandb_run_name: str | None = None

    # Precision
    gradient_checkpointing: bool = False
    fp16: bool = False

    # FSDP configuration
    fsdp_full_params: bool = False  # If True, use full_tensor; if False, use shard_tensor
    fsdp_state_dict_cpu_offload: bool = True  # If True, offload full state dict to CPU during collection.

    deterministic_mode: bool = False  # This name must be the same as Megatron's
    # Profile
    record_memory_history: bool = False
    memory_snapshot_path: str = "snapshot.pickle"
    use_pytorch_profiler: bool = False
    profile_step_start: int = 10
    profile_step_end: int = 12
    tensorboard_dir: str | None = None

    # YAML bookkeeping
    config: str | None = None


def parse_fsdp_cli(extra_args_provider=None):
    parser = argparse.ArgumentParser("FSDP SFT Training (slime)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    for f in dataclasses.fields(FSDPArgs):
        if f.name == "config":
            continue

        arg_type = str if f.type == (str | None) else f.type

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
