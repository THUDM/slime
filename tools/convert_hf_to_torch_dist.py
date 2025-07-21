import os
import shutil

import torch
from mbridge import AutoBridge
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.global_vars import set_args

import slime_plugins.mbridge  # noqa: F401


def init_distributed():
    """Initialize distributed environment"""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)


def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    return parser


def main():
    # Parse command line arguments
    args = parse_args(add_convertion_args)
    args.use_dist_ckpt = args.ckpt_format != "torch"
    set_args(args)

    # Initialize distributed environment
    init_distributed()

    # Load model
    hf_model_path = args.hf_checkpoint
    bridge = AutoBridge.from_pretrained(hf_model_path)
    model = bridge.get_model()
    bridge.load_weights(model, hf_model_path)
    print(f"Model loaded: {hf_model_path}")

    save_checkpoint(1, model, None, None, 0)
    # change to release ckpt
    tracker_filename = get_checkpoint_tracker_filename(args.save)
    with open(tracker_filename, "w") as f:
        f.write("release")
    source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
    target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
    shutil.move(source_dir, target_dir)


if __name__ == "__main__":
    main()
