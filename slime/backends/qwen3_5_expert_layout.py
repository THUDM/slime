"""Qwen3.5 fused routed-expert layout helpers.

The native slime bridge sees the same logical routed experts in two physical
layouts:

* Hugging Face stores one 3D ``gate_up_proj`` / ``down_proj`` tensor per layer.
* compressed-tensors and SGLang exchange one 2D weight per expert/projection.

This module owns that layout boundary so offline checkpoint conversion and
online Megatron-to-HF updates cannot silently drift apart.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any


QWEN3_5_MOE_MODEL_TYPE = "qwen3_5_moe"
QWEN3_5_MOE_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"
GATE_UP_PROJ = "gate_up_proj"
DOWN_PROJ = "down_proj"

_HF_FUSED_EXPERT_RE = re.compile(r"(?P<prefix>.*\.mlp\.experts)\.(?P<projection>gate_up_proj|down_proj)$")
_MEGATRON_FUSED_EXPERT_RE = re.compile(r"mlp\.experts\.linear_fc(?P<projection>[12])$")
_EP_OFFSET_RE = re.compile(r"\.__ep_offset(?P<offset>\d+)$")


def is_qwen3_5_moe_config(config: Any) -> bool:
    """Match the exact Qwen3.5 MoE identity used by the supported checkpoint."""
    if config is None:
        return False
    if isinstance(config, Mapping):
        model_type = config.get("model_type")
        architectures = config.get("architectures") or []
    else:
        model_type = getattr(config, "model_type", None)
        architectures = getattr(config, "architectures", None) or []
    return model_type == QWEN3_5_MOE_MODEL_TYPE and QWEN3_5_MOE_ARCHITECTURE in architectures


def match_hf_fused_expert(name: str) -> tuple[str, str] | None:
    """Return ``(expert_prefix, projection)`` for a fused Qwen3.5 HF key."""
    match = _HF_FUSED_EXPERT_RE.fullmatch(name)
    if match is None:
        return None
    return match.group("prefix"), match.group("projection")


def megatron_fused_expert_projection(name: str) -> str | None:
    """Return the canonical projection for a fused Qwen3.5 MCore name."""
    match = _MEGATRON_FUSED_EXPERT_RE.fullmatch(name)
    if match is None:
        return None
    return GATE_UP_PROJ if match.group("projection") == "1" else DOWN_PROJ


def encode_expert_parallel_offset(name: str, expert_offset: int) -> str:
    """Carry an EP rank's first global expert id across the name-only bridge seam."""
    return f"{name}.__ep_offset{expert_offset}"


def decode_expert_parallel_offset(name: str) -> tuple[str, int]:
    """Strip the internal EP offset suffix, defaulting to rank-zero offset."""
    match = _EP_OFFSET_RE.search(name)
    if match is None:
        return name, 0
    return name[: match.start()], int(match.group("offset"))


def iter_fused_expert_projections(
    weight: Any,
    projection: str,
    *,
    first_expert_id: int = 0,
) -> Iterator[tuple[int, str, Any]]:
    """Split one 3D fused tensor into ``(global_id, projection, 2D weight)``."""
    if weight.dim() != 3:
        raise ValueError(f"fused expert weight must be 3D, got shape {tuple(weight.shape)}")
    if projection not in {GATE_UP_PROJ, DOWN_PROJ}:
        raise ValueError(f"unsupported fused expert projection: {projection}")
    if projection == GATE_UP_PROJ and weight.shape[1] % 2 != 0:
        raise ValueError(f"gate_up_proj output dimension must be even, got shape {tuple(weight.shape)}")

    for local_expert_id in range(weight.shape[0]):
        expert_id = first_expert_id + local_expert_id
        expert_weight = weight[local_expert_id]
        if projection == GATE_UP_PROJ:
            gate, up = expert_weight.chunk(2, dim=0)
            yield expert_id, "gate_proj", gate.contiguous()
            yield expert_id, "up_proj", up.contiguous()
        else:
            yield expert_id, "down_proj", expert_weight.contiguous()
