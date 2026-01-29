import base64
import io
import logging
import types

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14

_qwen_process_vision_info = None


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def _patch_processor_call(processor: ProcessorMixin):
    """Patch processor.__call__ to inject default kwargs for modality-specific processors."""
    original_call = processor.__call__

    def patched_call(self, *args, **kwargs):
        # force return_tensors to None for input_ids
        kwargs["return_tensors"] = None
        # have been resized by qwen_vl_utils, update this when supporting other models
        kwargs["do_resize"] = False

        # set return_tensors="pt" for modality-specific outputs
        for modality_kwargs_key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
            if modality_kwargs_key not in kwargs:
                kwargs[modality_kwargs_key] = {"return_tensors": "pt"}
            else:
                kwargs[modality_kwargs_key].setdefault("return_tensors", "pt")

        return original_call(*args, **kwargs)

    processor.__call__ = types.MethodType(patched_call, processor)


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    # Patch processor __call__ to add default kwargs
    if proc is not None:
        _patch_processor_call(proc)

        global _qwen_process_vision_info
        if _qwen_process_vision_info is None:
            try:
                from qwen_vl_utils import process_vision_info as _fn

                _qwen_process_vision_info = _fn
            except ImportError:
                logger.warning("qwen_vl_utils not installed, process_vision_info will not work")

    return proc


def process_vision_info(prompt, processor):
    # TODO: temporary solution, will write image utils for slime later
    if _qwen_process_vision_info is None:
        raise ImportError("qwen_vl_utils is not installed. Install it with: pip install qwen-vl-utils")

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = _qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"
