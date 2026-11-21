import base64
import io
import json
import logging
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # return_tensors=None for text (input_ids as lists), "pt" for modality-specific outputs
    result["text_kwargs"] = {
        **result.get("text_kwargs", {}),
        "return_tensors": None,
        "return_mm_token_type_ids": False,
    }
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def _try_load_glm4v_processor(name_or_path: str, **kwargs):
    """Fallback: manually construct a Glm4vProcessor for GLM-4.6V / GLM-4.5V models.

    AutoProcessor fails for these models on transformers < 5.0 because
    the Glm46VProcessor / Glm4vMoeProcessor classes are not registered.
    The underlying Glm4vProcessor (non-MoE) works for both variants since
    they share the same vision architecture.
    """
    try:
        from transformers.models.glm4v.image_processing_glm4v import Glm4vImageProcessor
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
        from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor
    except ImportError:
        return None

    pp_path = Path(name_or_path) / "preprocessor_config.json"
    vp_path = Path(name_or_path) / "video_preprocessor_config.json"
    if not pp_path.exists():
        return None

    skip_keys = {"image_processor_type", "processor_class", "video_processor_type"}
    with open(pp_path) as f:
        pp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
    image_processor = Glm4vImageProcessor(**pp_cfg)

    video_processor = None
    if vp_path.exists():
        with open(vp_path) as f:
            vp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
        video_processor = Glm4vVideoProcessor(**vp_cfg)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    proc = Glm4vProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )
    logger.info(f"Loaded Glm4vProcessor manually for {name_or_path}")
    return proc


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer instead of a proper processor, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        # Fallback: try to construct a GLM-4.6V / GLM-4.5V processor manually.
        proc = _try_load_glm4v_processor(name_or_path, **kwargs)

    return proc


def _extract_images_from_messages(messages):
    """Extract PIL images from chat messages containing multimodal content.

    Handles base64 strings (with or without data: URI prefix), file paths,
    and PIL Image objects embedded in message content dicts.
    """
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image_data = item.get("image")
            if image_data is None:
                continue
            if isinstance(image_data, Image.Image):
                images.append(image_data)
            elif isinstance(image_data, str):
                if image_data.startswith("data:"):
                    _, encoded = image_data.split(",", 1)
                    images.append(Image.open(io.BytesIO(base64.b64decode(encoded))))
                else:
                    try:
                        raw = base64.b64decode(image_data)
                        images.append(Image.open(io.BytesIO(raw)))
                    except Exception:
                        # Not base64 — try as file path
                        images.append(Image.open(image_data))
    return images


# Image processors whose own sizing rule must see the original image. For these
# the qwen_vl_utils pre-resize below is skipped, and each image is resized once,
# here, to the size the processor itself picks.
#
# Why not just skip qwen_vl_utils: the image then reaches two different sizing
# rules -- the HF processor builds the training prompt, SGLang's own processor the
# rollout prompt -- and for Nemotron 3.5 they disagree (HF rounds patches with
# round(x + 0.5), SGLang with round(x)). On geo3k that is 1570 of 2702 images
# with different image-token counts. Qwen's resize happens to hide it by snapping
# every image to a multiple of 32 first. The HF target size is a fixed point of
# both rules, so resizing to it makes both sides build the same grid -- and, since
# neither then resizes again, the same pixels.
_NATIVE_RESIZE_IMAGE_PROCESSORS = frozenset({"NemotronH_Omni_Reasoning_V3ImageProcessor"})


def uses_native_resize(processor) -> bool:
    """Whether process_vision_info resizes to the processor's own target size."""
    image_processor = getattr(processor, "image_processor", None)
    return type(image_processor).__name__ in _NATIVE_RESIZE_IMAGE_PROCESSORS


def _resize_to_native_target(images, image_processor):
    """Resize each image to the size `image_processor` would resize it to.

    The size comes from the processor itself (`imgs_sizes`, one (h, w) per image),
    so this does not restate its sizing rule. One call over all of a prompt's
    images, so the processor's per-prompt budget applies as it would at train time.
    The resize uses that processor's own kernel (torch bicubic, antialias), rounded
    back to uint8 so the same image can be handed to the rollout engine as a PNG.
    """
    import numpy as np
    import torch

    images = [image.convert("RGB") for image in images]
    sizes = image_processor(images=images, return_tensors=None)["imgs_sizes"]
    resized = []
    for image, (height, width) in zip(images, sizes, strict=True):
        if image.size != (width, height):
            pixels = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)[None].float()
            pixels = torch.nn.functional.interpolate(
                pixels, size=(height, width), mode="bicubic", align_corners=False, antialias=True
            )
            image = Image.fromarray(pixels.round().clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy())
        resized.append(image)
    return resized


def process_vision_info(prompt, processor):
    """Extract PIL images (and videos) from the message list for training.

    Processors in _NATIVE_RESIZE_IMAGE_PROCESSORS get their images resized to their
    own target size (see there). Otherwise tries qwen_vl_utils first (Qwen VL
    family), falls back to generic extraction for other models (e.g. GLM-4.6V).
    """
    if uses_native_resize(processor):
        images = _extract_images_from_messages(prompt) or None
        if images:
            images = _resize_to_native_target(images, processor.image_processor)
        return {"images": images, "videos": None}

    try:
        from qwen_vl_utils import process_vision_info as qwen_process_vision_info

        if hasattr(processor.image_processor, "patch_size"):
            image_patch_size = processor.image_processor.patch_size
        else:
            image_patch_size = DEFAULT_PATCH_SIZE
        images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    except Exception:
        # Fallback: generic extraction for non-Qwen models
        images = _extract_images_from_messages(prompt) or None
        videos = None

    return {"images": images, "videos": videos}


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"
