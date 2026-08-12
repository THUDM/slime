from pathlib import Path

from transformers import AutoConfig

from .common import load_model_hf_weights
from .deepseek import deepseek_hf_tensor
from .glm import glm4_hf_tensor, glm4_moe_hf_tensor
from .qwen import mimo_hf_tensor, minimax_m2_hf_tensor, qwen_hf_tensor, qwen_moe_hf_tensor
from .qwen3_5 import qwen3_5_hf_tensor
from .qwen3_next import qwen3_next_hf_tensor

_LOADERS = {
    "deepseek_v3": deepseek_hf_tensor,
    "deepseek_v32": deepseek_hf_tensor,
    "glm4": glm4_hf_tensor,
    "glm4_moe": glm4_moe_hf_tensor,
    "glm4_moe_lite": deepseek_hf_tensor,
    "glm_moe_dsa": deepseek_hf_tensor,
    "kimi_k2": deepseek_hf_tensor,
    "llama": qwen_hf_tensor,
    "mimo": mimo_hf_tensor,
    "minimax_m2": minimax_m2_hf_tensor,
    "qwen2": qwen_hf_tensor,
    "qwen2_moe": qwen_moe_hf_tensor,
    "qwen3": qwen_hf_tensor,
    "qwen3_5": qwen3_5_hf_tensor,
    "qwen3_5_moe": qwen3_5_hf_tensor,
    "qwen3_moe": qwen_moe_hf_tensor,
    "qwen3_next": qwen3_next_hf_tensor,
}


def supports_hf_weight_loading(path: str | Path) -> bool:
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config.model_type in _LOADERS


def load_hf_weights(args, model, path: str | Path) -> None:
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    try:
        get_hf_tensor = _LOADERS[config.model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported HuggingFace model type: {config.model_type}") from exc
    load_model_hf_weights(args, model, path, config, get_hf_tensor)
