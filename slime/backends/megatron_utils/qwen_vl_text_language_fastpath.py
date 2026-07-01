"""Opt-in text-only fastpath for Qwen3VL Bridge models.

Qwen3VLModel owns the vision embedding + packed-sequence workaround.  For
microbatches without multimodal tensors, Slime can already provide the normal
CP-local THD inputs used by the language model.  This patch lets those batches
bypass the Qwen3VL wrapper while keeping the root model/DDP forward call.
"""

import logging
import os

import torch
from megatron.core import mpu

logger = logging.getLogger(__name__)
_PATCHED = False


def _enabled() -> bool:
    return os.environ.get("SLIME_QWENVL_TEXT_LANGUAGE_FASTPATH", "0").lower() in {"1", "true", "yes", "on"}


def _mrope_cp_thd_unsafe(kwargs: dict) -> bool:
    packed_seq_params = kwargs.get("packed_seq_params")
    if getattr(packed_seq_params, "qkv_format", None) != "thd":
        return False
    if kwargs.get("attention_mask") is not None:
        return True
    if kwargs.get("position_ids") is None:
        return True
    if os.environ.get("SLIME_QWENVL_TEXT_FASTPATH_LOCAL_MROPE", "1").lower() in {"1", "true", "yes", "on"}:
        return False
    try:
        return mpu.get_context_parallel_world_size() > 1
    except Exception:
        return True


def _has_tensor(value) -> bool:
    if torch.is_tensor(value):
        return value.numel() > 0
    if isinstance(value, dict):
        return any(_has_tensor(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_tensor(v) for v in value)
    return value is not None


def _has_mm_kwargs(kwargs: dict) -> bool:
    for name in (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "image_input_mask",
        "video_input_mask",
        "cp_img_num",
        "images_padded",
    ):
        if _has_tensor(kwargs.get(name)):
            return True
    return False


def apply_qwen_vl_text_language_fastpath_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    except Exception as exc:
        logger.debug("QwenVL text language fastpath unavailable: %r", exc)
        return

    original_forward = Qwen3VLModel.forward

    def _patched_forward(self, *args, **kwargs):
        if not _enabled() or args or _has_mm_kwargs(kwargs):
            return original_forward(self, *args, **kwargs)

        input_ids = kwargs.get("input_ids")
        packed_seq_params = kwargs.get("packed_seq_params")
        language_model = getattr(self, "language_model", None)
        if language_model is None or input_ids is None or packed_seq_params is None:
            return original_forward(self, *args, **kwargs)

        if getattr(packed_seq_params, "qkv_format", None) != "thd":
            return original_forward(self, *args, **kwargs)

        if _mrope_cp_thd_unsafe(kwargs):
            if not getattr(self, "_slime_qwenvl_text_fastpath_skip_logged", False):
                logger.info(
                    "SLIME_QWENVL_TEXT_LANGUAGE_FASTPATH skipped for packed THD batch; "
                    "using Qwen3VL wrapper to preserve mRoPE/CP preprocessing"
                )
                self._slime_qwenvl_text_fastpath_skip_logged = True
            return original_forward(self, *args, **kwargs)

        if not getattr(self, "_slime_qwenvl_text_fastpath_logged", False):
            logger.info("SLIME_QWENVL_TEXT_LANGUAGE_FASTPATH enabled: bypassing Qwen3VL wrapper for text-only THD batch")
            self._slime_qwenvl_text_fastpath_logged = True

        rotary_pos_emb = getattr(language_model, "rotary_pos_emb", None)
        if getattr(rotary_pos_emb, "is_thd_format", None) is not None:
            rotary_pos_emb.is_thd_format = True

        return language_model(
            input_ids=input_ids,
            position_ids=kwargs.get("position_ids"),
            attention_mask=kwargs.get("attention_mask"),
            labels=kwargs.get("labels"),
            inference_params=kwargs.get("inference_params"),
            inference_context=kwargs.get("inference_context"),
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=kwargs.get("extra_block_kwargs"),
            runtime_gather_output=kwargs.get("runtime_gather_output"),
            loss_mask=kwargs.get("loss_mask"),
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
        )

    Qwen3VLModel.forward = _patched_forward
    _PATCHED = True
    logger.info("Installed QwenVL text language fastpath patch")


apply_qwen_vl_text_language_fastpath_patch()
