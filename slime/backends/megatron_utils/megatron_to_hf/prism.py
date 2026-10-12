"""Prism's physical Megatron parameters to Omni's HF weight-loading names.

QKV is group-major ``[Q_group, K_group, V_group]``; gathered SwiGLU FC1 is
``[all gate rows, all up rows]``. Repeated execution units share physical layer
parameters and must never be expanded during publication. These mappings match
mossLite's Prism native-to-HF converter and Omni's ``load_weights`` contract.
"""

import re

import torch

from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix


def is_prism_model(model_name: str) -> bool:
    family = model_name.lower().replace("_", "").replace("-", "")
    return family in {"prism", "mossttsprism", "mossttsprismconfig", "mossttsprismmodel"}


def _dimension(args, name: str) -> int:
    value = getattr(args, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Prism weight conversion requires positive {name}")
    return value


def _inference_dtype(args, source_dtype: torch.dtype) -> torch.dtype:
    value = getattr(args, "sglang_dtype", "auto")
    if value in (None, "auto"):
        return source_dtype
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    dtype = aliases.get(value) if isinstance(value, str) else value
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported Prism inference dtype: {value}")
    return dtype


def convert_prism_to_hf(args, name: str, param: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Convert one complete, TP-gathered parameter without altering vocab padding.

    ``sglang_dtype=auto`` preserves the trainable dtype. An explicit existing
    ``--sglang-dtype bfloat16`` casts every published parameter, including norms
    and stop, to BF16. The engine retains its FP32 stop projection computation.
    """
    if param.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported Prism trainable dtype: {param.dtype}")
    name = strip_param_name_prefix(name)
    hidden = _dimension(args, "hidden_size")

    def checked(shape):
        if tuple(param.shape) != shape:
            raise ValueError(f"Prism shape mismatch for {name}: got {tuple(param.shape)}, expected {shape}")
        return param.to(dtype=_inference_dtype(args, param.dtype)).contiguous()

    if name == "text_embedding.word_embeddings.weight":
        # The HF artifact owns the padded text vocabulary. Tokenizer vocab_size
        # is smaller and must not truncate this matrix, unlike text-only models.
        return [("transformer.embed_tokens.weight", checked((_dimension(args, "padded_vocab_size"), hidden)))]
    if name == "stop_head.weight":
        return [(name, checked((2, hidden)))]

    match = re.fullmatch(r"(audio_embeddings|audio_lm_heads)\.(\d+)\.(word_embeddings\.)?weight", name)
    if match:
        module, channel, embedding = match.groups()
        if (module == "audio_embeddings") != (embedding is not None) or int(channel) >= _dimension(args, "n_vq"):
            raise ValueError(f"unknown Prism parameter: {name}")
        return [(f"{module}.{channel}.weight", checked((_dimension(args, "speech_vocab_size"), hidden)))]

    match = re.fullmatch(r"prediction_norms\.(\d+)\.weight", name)
    if match:
        counts = getattr(args, "prism_rvq_heads_per_execution_unit", ())
        if int(match[1]) >= sum(count > 0 for count in counts):
            raise ValueError(f"unknown Prism prediction norm: {name}")
        return [(name, checked((hidden,)))]

    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if match is None or int(match[1]) >= _dimension(args, "num_layers"):
        raise ValueError(f"unknown Prism parameter: {name}")
    layer, suffix = match.groups()
    target = f"transformer.layers.{layer}"
    mlp_only = int(layer) in getattr(args, "prism_mlp_only_layers", ())
    ffn = _dimension(args, "ffn_hidden_size")

    if suffix == "mlp.linear_fc1.layer_norm_weight":
        return [(f"{target}.post_attention_layernorm.weight", checked((hidden,)))]
    if suffix == "mlp.linear_fc1.weight":
        gate, up = checked((2 * ffn, hidden)).chunk(2, dim=0)
        return [(f"{target}.mlp.gate_proj.weight", gate), (f"{target}.mlp.up_proj.weight", up)]
    if suffix == "mlp.linear_fc2.weight":
        return [(f"{target}.mlp.down_proj.weight", checked((hidden, ffn)))]
    if mlp_only:
        raise ValueError(f"MLP-only Prism layer cannot have attention parameter: {name}")

    heads, groups = _dimension(args, "num_attention_heads"), _dimension(args, "num_query_groups")
    head_dim = _dimension(args, "kv_channels")
    if heads % groups:
        raise ValueError("Prism query heads must be divisible by query groups")
    q_per_group = heads // groups
    if suffix == "self_attention.linear_qkv.weight":
        qkv = checked(((heads + 2 * groups) * head_dim, hidden)).view(groups, q_per_group + 2, head_dim, hidden)
        return [
            (f"{target}.self_attn.q_proj.weight", qkv[:, :q_per_group].reshape(-1, hidden).contiguous()),
            (f"{target}.self_attn.k_proj.weight", qkv[:, q_per_group].reshape(-1, hidden).contiguous()),
            (f"{target}.self_attn.v_proj.weight", qkv[:, q_per_group + 1].reshape(-1, hidden).contiguous()),
        ]
    if suffix == "self_attention.linear_proj.weight":
        return [(f"{target}.self_attn.o_proj.weight", checked((hidden, heads * head_dim)))]
    if suffix == "self_attention.linear_qkv.layer_norm_weight":
        return [(f"{target}.input_layernorm.weight", checked((hidden,)))]
    if suffix in ("self_attention.q_layernorm.weight", "self_attention.k_layernorm.weight"):
        axis = "q" if suffix == "self_attention.q_layernorm.weight" else "k"
        return [(f"{target}.self_attn.{axis}_norm.weight", checked((head_dim,)))]
    raise ValueError(f"unknown Prism parameter: {name}")
