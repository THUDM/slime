"""Default-off diagnostic switch for QwenVL deepstack visual injection."""

from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, os.environ.get(name.lower(), "0")).lower() in {"1", "true", "yes", "on"}


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _tensor_summary(value) -> str:
    if torch.is_tensor(value):
        return f"shape={tuple(value.shape)} dtype={str(value.dtype).replace('torch.', '')}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_tensor_summary(item) for item in value[:4]) + (", ..." if len(value) > 4 else "") + "]"
    return type(value).__name__


def apply_qwen_vl_disable_deepstack_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
    except Exception as exc:
        logger.debug("QwenVL disable-deepstack patch unavailable: %r", exc)
        return

    original_forward = Qwen3VLGPTModel.forward

    @functools.wraps(original_forward)
    def _patched_forward(self, *args, **kwargs):
        if not _env_flag("SLIME_QWENVL_DISABLE_DEEPSTACK"):
            return original_forward(self, *args, **kwargs)

        deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
        if deepstack_visual_embeds is None:
            return original_forward(self, *args, **kwargs)

        if _rank() == 0 and not getattr(self, "_slime_qwenvl_disable_deepstack_logged", False):
            logger.info(
                "SLIME_QWENVL_DISABLE_DEEPSTACK enabled: dropping decoder deepstack visual embeds %s",
                _tensor_summary(deepstack_visual_embeds),
            )
            self._slime_qwenvl_disable_deepstack_logged = True

        patched_kwargs = dict(kwargs)
        patched_kwargs["deepstack_visual_embeds"] = None
        patched_kwargs["visual_pos_masks"] = None
        return original_forward(self, *args, **patched_kwargs)

    Qwen3VLGPTModel.forward = _patched_forward
    _PATCHED = True
    logger.info("Installed QwenVL disable-deepstack diagnostic patch")


apply_qwen_vl_disable_deepstack_patch()
