import argparse
from dataclasses import dataclass
import io
import os
from typing import Optional

import torch

from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config


def _patch_hybrid_pattern_compat():
    """Patch Megatron-LM forks that predate unified hybrid/MTP pattern helpers.

    The Slime v0.3.0 image pairs a newer Megatron-Bridge with a Megatron-LM fork
    whose mamba_hybrid_layer_allocation module does not expose
    parse_hybrid_pattern. Bridge imports the symbol unconditionally while loading
    MLM checkpoints, even for non-Mamba models such as Qwen3.6-VL.
    """
    import megatron.core.ssm.mamba_hybrid_layer_allocation as hybrid_alloc

    symbols = hybrid_alloc.Symbols
    if not hasattr(symbols, "MTP_SEPARATOR"):
        symbols.MTP_SEPARATOR = "/"
    if not hasattr(symbols, "PIPE"):
        symbols.PIPE = "|"
    if not hasattr(symbols, "VALID_LAYERS"):
        symbols.VALID_LAYERS = getattr(symbols, "VALID", None)

    if hasattr(hybrid_alloc, "parse_hybrid_pattern"):
        return

    @dataclass
    class ParsedHybridPattern:
        main_pattern: Optional[str]
        mtp_pattern: Optional[str]
        mtp_num_depths: int

    def _valid_layer_symbols():
        valid = getattr(symbols, "VALID_LAYERS", None) or getattr(symbols, "VALID", None)
        if valid is None:
            valid = {symbols.MAMBA, symbols.ATTENTION, symbols.MLP, symbols.MOE}
        return set(valid)

    def _validate_pattern(pattern: Optional[str], pattern_name: str) -> None:
        if not pattern:
            return
        valid = _valid_layer_symbols()
        for symbol in pattern:
            if symbol == symbols.PIPE:
                continue
            if symbol not in valid:
                raise ValueError(f"In {pattern_name} hybrid pattern, '{symbol}' is not one of {valid}")

    def parse_hybrid_pattern(pattern: Optional[str]) -> ParsedHybridPattern:
        if not pattern:
            return ParsedHybridPattern(main_pattern=None, mtp_pattern=None, mtp_num_depths=0)

        parts = pattern.split(symbols.MTP_SEPARATOR)
        main_pattern = parts[0] or None
        _validate_pattern(main_pattern, "main")

        if len(parts) == 1:
            return ParsedHybridPattern(main_pattern=main_pattern, mtp_pattern=None, mtp_num_depths=0)

        mtp_patterns = parts[1:]
        first_mtp_pattern = mtp_patterns[0]
        if not first_mtp_pattern:
            raise ValueError(f"Empty MTP hybrid pattern in '{pattern}'")
        for mtp_pattern in mtp_patterns:
            if mtp_pattern != first_mtp_pattern:
                raise ValueError(f"Inconsistent MTP hybrid patterns in '{pattern}'")
            _validate_pattern(mtp_pattern, "MTP")

        return ParsedHybridPattern(
            main_pattern=main_pattern,
            mtp_pattern=first_mtp_pattern,
            mtp_num_depths=len(mtp_patterns),
        )

    hybrid_alloc.ParsedHybridPattern = ParsedHybridPattern
    hybrid_alloc.parse_hybrid_pattern = parse_hybrid_pattern


_patch_hybrid_pattern_compat()

import megatron.bridge.training.model_load_save as _model_load_save_module
from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion import model_bridge as _model_bridge_module
from megatron.core.dist_checkpointing.strategies import torch as _mcore_torch_strategy


# Here we need to patch Megatron Bridge's `load_model_config`, since the checkpoint is saved
# by Megatron and lack of provider information.
_provider_override = {}
_original_load_model_config = _model_load_save_module.load_model_config


def _patched_load_model_config(checkpoint_path):
    model_cfg, mlm_args = _original_load_model_config(checkpoint_path)
    provider = _provider_override.get("provider")
    if provider is not None:
        from megatron.bridge.models.model_provider import ModelProviderMixin

        if not isinstance(model_cfg, ModelProviderMixin):
            print(f"[convert] Overriding MLM TransformerConfig with Bridge provider: " f"{type(provider).__name__}")
            return provider, mlm_args
    return model_cfg, mlm_args


_model_load_save_module.load_model_config = _patched_load_model_config


_bridge_class = getattr(_model_bridge_module, "ModelBridge", None)
if _bridge_class is None:
    _bridge_class = getattr(_model_bridge_module, "MegatronModelBridge")
_original_build_conversion_tasks = _bridge_class.build_conversion_tasks


def _patched_build_conversion_tasks(self, *args, **kwargs):
    tasks = list(_original_build_conversion_tasks(self, *args, **kwargs))
    filtered = [task for task in tasks if task is not None]
    skipped = len(tasks) - len(filtered)
    if skipped:
        print(f"[convert] Skipping {skipped} unmapped Megatron->HF conversion tasks.")
    return filtered


_bridge_class.build_conversion_tasks = _patched_build_conversion_tasks


_original_replace_sharded_keys = _mcore_torch_strategy._replace_sharded_keys_with_state_dict_keys


def _patched_replace_sharded_keys_with_state_dict_keys(state_dict, flat_mapping, rename_mapping):
    """Handle ShardedObject BytesIO values when loading Slime torch_dist ckpts.

    This Megatron-LM fork unwraps tensor shards to lists, but leaves sharded
    object payloads as a single BytesIO. The original restore function then
    calls len(BytesIO) and fails. Decode those payloads back into the list that
    mcore_to_pyt_state_dict serialized, and leave tensor shard lists unchanged.
    """
    recovered_sd = {}
    for key, tensors in state_dict.items():
        if isinstance(tensors, io.BytesIO):
            tensors.seek(0)
            tensors = torch.load(tensors, map_location="cpu", weights_only=False)
        elif not isinstance(tensors, list):
            tensors = [tensors]

        assert len(tensors) == len(rename_mapping[key]), (
            key,
            len(tensors),
            len(rename_mapping[key]),
        )
        for tensor, recovered_key in zip(tensors, rename_mapping[key]):
            recovered_sd[recovered_key] = tensor

    return _mcore_torch_strategy.unflatten_state_dict(recovered_sd, flat_mapping)


_mcore_torch_strategy._replace_sharded_keys_with_state_dict_keys = _patched_replace_sharded_keys_with_state_dict_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert torch distributed checkpoint to HuggingFace format using Megatron Bridge"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to the torch distributed checkpoint directory"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the HuggingFace checkpoint")
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="Path to the original HuggingFace model directory (for config)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    print(f"Loading config from {args.origin_hf_dir}")
    bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True))

    # Use Bridge's provider so the correct model class is created (e.g., Qwen3VLModel
    # instead of GPTModel). This is needed because MLM checkpoints lack run_config.yaml.
    provider = bridge.to_megatron_provider(load_weights=False)
    _provider_override["provider"] = provider
    print(f"[convert] Using Bridge provider: {type(provider).__name__}")

    print(f"Exporting checkpoint from {args.input_dir} to {args.output_dir}")
    bridge.export_ckpt(args.input_dir, args.output_dir)

    print("Done!")
