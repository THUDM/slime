#!/usr/bin/env python3
"""
Convert InternVL checkpoint between HuggingFace and SGLang formats.

Usage:
    python scripts/convert_internvl_checkpoint.py \
        --hf-checkpoint /path/to/hf/checkpoint \
        --sglang-model /path/to/original/sglang/model \
        --output /path/to/output

This script converts HuggingFace format checkpoints (from training) to
SGLang format (for inference), handling key naming differences and QKV
weight concatenation.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add slime to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from slime.utils.model_converter import convert_hf_checkpoint_to_sglang

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert InternVL checkpoint from HuggingFace to SGLang format"
    )
    parser.add_argument(
        "--hf-checkpoint",
        type=str,
        required=True,
        help="Path to HuggingFace format checkpoint directory (contains .safetensors files)"
    )
    parser.add_argument(
        "--sglang-model",
        type=str,
        required=True,
        help="Path to original SGLang model directory (for config and tokenizer)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted SGLang format checkpoint"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("InternVL Checkpoint Conversion: HuggingFace → SGLang")
    logger.info("=" * 60)
    logger.info(f"HF Checkpoint:  {args.hf_checkpoint}")
    logger.info(f"SGLang Model:   {args.sglang_model}")
    logger.info(f"Output:         {args.output}")
    logger.info("=" * 60)

    try:
        convert_hf_checkpoint_to_sglang(
            hf_checkpoint_dir=args.hf_checkpoint,
            sglang_save_dir=args.output,
            original_sglang_model_dir=args.sglang_model
        )
        logger.info("=" * 60)
        logger.info("✅ Conversion completed successfully!")
        logger.info(f"✅ SGLang checkpoint saved to: {args.output}")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info(f"1. Start SGLang server with converted checkpoint:")
        logger.info(f"   bash scripts/start_sglang_internvl.sh --model {args.output}")
        logger.info("2. Continue training with SLIME")

    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
