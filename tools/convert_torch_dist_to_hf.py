import argparse
import os
import pickle

from transformers import AutoConfig

from slime.utils.torch_dist_to_hf import (
    _UnpicklerWrapper,
    copy_assets,
    load_torch_dist,
    save_tensors,
)

# Global monkey-patch so torch.load of checkpoint metadata can deserialise
# Megatron classes without importing them.  Only safe in standalone scripts.
pickle.Unpickler = _UnpicklerWrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        default=None,
        help="use the origin hf dir to copy files like tokenizer, config.json, etc.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "-a", "--add-missing-from-origin-hf", action="store_true", help="Add missing weights from origin hf checkpoint"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5 * 1024**3,
        help="Chunk size for saving tensors, default is 2GB.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocab size for removing padding, if applicable. If not provided, no padding will be removed.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError(
            "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
        )

    if args.model_name is None:
        hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
        args.model_name = type(hf_config).__name__.lower()

    megatron_args, state_dict = load_torch_dist(args.input_dir)

    save_tensors(
        megatron_args,
        args.model_name,
        state_dict,
        args.output_dir,
        args.chunk_size,
        args.vocab_size,
        args.origin_hf_dir if args.add_missing_from_origin_hf else None,
    )

    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)
