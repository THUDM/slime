"""
Model format converter between HuggingFace and SGLang formats.

This module provides utilities to convert InternVL model weights between
HuggingFace format (used for training) and SGLang format (used for inference).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def convert_hf_to_sglang_keys(hf_state_dict: dict) -> dict:
    """
    Convert HuggingFace InternVL weight keys to SGLang format.

    Key differences:
    1. multi_modal_projector.* -> mlp1.*
    2. vision_tower.* -> vision_model.*
    3. embeddings naming (cls_token, patch_embeddings, position_embeddings)
    4. encoder.layer.N -> encoder.layers.N
    5. QKV weights: separate q/k/v_proj -> concatenated qkv
    """
    new_state_dict = {}
    qkv_buffer = {}  # Buffer for QKV concatenation

    for key, value in hf_state_dict.items():
        # === 1. multi_modal_projector → mlp1.* ===
        if key.startswith('multi_modal_projector.layer_norm.'):
            new_key = key.replace('multi_modal_projector.layer_norm.', 'mlp1.0.')
        elif key.startswith('multi_modal_projector.linear_1.'):
            new_key = key.replace('multi_modal_projector.linear_1.', 'mlp1.1.')
        elif key.startswith('multi_modal_projector.linear_2.'):
            new_key = key.replace('multi_modal_projector.linear_2.', 'mlp1.3.')

        # === 2. embeddings ===
        elif key == 'vision_tower.embeddings.cls_token':
            new_key = 'vision_model.embeddings.class_embedding'
        elif key.startswith('vision_tower.embeddings.patch_embeddings.projection.'):
            new_key = key.replace(
                'vision_tower.embeddings.patch_embeddings.projection',
                'vision_model.embeddings.patch_embedding'
            )
        elif key == 'vision_tower.embeddings.position_embeddings':
            new_key = 'vision_model.embeddings.position_embedding'

        # === 3. encoder.layer.X → encoder.layers.X ===
        elif key.startswith('vision_tower.encoder.layer.'):
            parts = key.split('.')
            layer_id = parts[3]
            suffix = '.'.join(parts[4:])
            base = f"vision_model.encoder.layers.{layer_id}."

            # Handle QKV weight and bias separately for concatenation
            if suffix in {
                'attention.q_proj.weight', 'attention.k_proj.weight', 'attention.v_proj.weight',
                'attention.q_proj.bias', 'attention.k_proj.bias', 'attention.v_proj.bias'
            }:
                if layer_id not in qkv_buffer:
                    qkv_buffer[layer_id] = {'weight': {}, 'bias': {}}

                if suffix.endswith('.weight'):
                    if 'q_proj' in suffix:
                        qkv_buffer[layer_id]['weight']['q_proj'] = value
                    elif 'k_proj' in suffix:
                        qkv_buffer[layer_id]['weight']['k_proj'] = value
                    elif 'v_proj' in suffix:
                        qkv_buffer[layer_id]['weight']['v_proj'] = value

                elif suffix.endswith('.bias'):
                    if 'q_proj' in suffix:
                        qkv_buffer[layer_id]['bias']['q_proj'] = value
                    elif 'k_proj' in suffix:
                        qkv_buffer[layer_id]['bias']['k_proj'] = value
                    elif 'v_proj' in suffix:
                        qkv_buffer[layer_id]['bias']['v_proj'] = value

                continue  # Skip adding to new_state_dict, will concatenate later

            elif suffix.startswith('attention.projection_layer.'):
                new_key = base + 'attn.proj.' + suffix.split('.')[-1]
            elif suffix.startswith('layernorm_before.'):
                new_key = base + 'norm1.' + suffix.split('.')[-1]
            elif suffix.startswith('layernorm_after.'):
                new_key = base + 'norm2.' + suffix.split('.')[-1]
            elif suffix == 'lambda_1':
                new_key = base + 'ls1'
            elif suffix == 'lambda_2':
                new_key = base + 'ls2'
            else:
                new_key = base + suffix

        else:
            new_key = key

        new_state_dict[new_key] = value

    # === 4. Concatenate QKV weights and biases ===
    for layer_id, qkv_parts in qkv_buffer.items():
        base = f"vision_model.encoder.layers.{layer_id}.attn.qkv"

        # Concatenate weights
        if all(k in qkv_parts['weight'] for k in ('q_proj', 'k_proj', 'v_proj')):
            qkv_weight = torch.cat([
                qkv_parts['weight']['q_proj'],
                qkv_parts['weight']['k_proj'],
                qkv_parts['weight']['v_proj']
            ], dim=0)
            new_state_dict[base + '.weight'] = qkv_weight

        # Concatenate biases
        if all(k in qkv_parts['bias'] for k in ('q_proj', 'k_proj', 'v_proj')):
            qkv_bias = torch.cat([
                qkv_parts['bias']['q_proj'],
                qkv_parts['bias']['k_proj'],
                qkv_parts['bias']['v_proj']
            ], dim=0)
            new_state_dict[base + '.bias'] = qkv_bias

    return new_state_dict


def convert_hf_checkpoint_to_sglang(
    hf_checkpoint_dir: str,
    sglang_save_dir: str,
    original_sglang_model_dir: Optional[str] = None,
) -> None:
    """
    Convert a HuggingFace checkpoint to SGLang format.

    Args:
        hf_checkpoint_dir: Directory containing HuggingFace format weights (safetensors)
        sglang_save_dir: Directory to save converted SGLang format weights
        original_sglang_model_dir: Optional path to original SGLang model for config/tokenizer
    """
    hf_path = Path(hf_checkpoint_dir)
    save_path = Path(sglang_save_dir)

    logger.info(f"Converting HF checkpoint from {hf_path} to SGLang format at {save_path}")

    # Load HF safetensor weights
    checkpoint_files = list(hf_path.glob("*.safetensors"))
    if not checkpoint_files:
        logger.warning(f"No safetensors files found in {hf_path}")
        return

    logger.info(f"Found {len(checkpoint_files)} safetensor files")

    hf_state_dict = {}
    for checkpoint_file in checkpoint_files:
        with safe_open(checkpoint_file, framework='pt') as f:
            for key in f.keys():
                hf_state_dict[key] = f.get_tensor(key)

    # Convert keys
    sglang_state_dict = convert_hf_to_sglang_keys(hf_state_dict)

    # Save converted weights
    save_path.mkdir(parents=True, exist_ok=True)

    # Save as safetensors (SGLang can load this)
    output_file = save_path / "model.safetensors"
    save_file(sglang_state_dict, str(output_file))
    logger.info(f"Saved converted weights to {output_file}")

    # Copy config and tokenizer if original model dir is provided
    if original_sglang_model_dir:
        original_path = Path(original_sglang_model_dir)
        for file_name in ['config.json', 'tokenizer_config.json', 'tokenizer.json',
                          'special_tokens_map.json', 'vocab.json', 'merges.txt']:
            src_file = original_path / file_name
            if src_file.exists():
                import shutil
                shutil.copy(src_file, save_path / file_name)
                logger.info(f"Copied {file_name}")

    logger.info(f"Conversion complete: {save_path}")
