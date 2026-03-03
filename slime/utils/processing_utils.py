import base64
import io
import logging
import os

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def is_internvl_model(processor) -> bool:
    """Check if the processor belongs to an InternVL model."""
    if processor is None:
        return False
    # InternVL models use InternVLChatConfig which has specific attributes
    processor_class_name = processor.__class__.__name__
    return "InternVL" in processor_class_name or "InternLM" in processor_class_name


def load_image(image_input) -> Image.Image:
    """Load image from various input types (path, URL, PIL Image, etc.)."""
    if isinstance(image_input, Image.Image):
        return image_input
    elif isinstance(image_input, str):
        if image_input.startswith(("http://", "https://")):
            import requests
            from io import BytesIO

            response = requests.get(image_input, timeout=10)
            return Image.open(BytesIO(response.content))
        elif os.path.exists(image_input):
            return Image.open(image_input)
        elif image_input.startswith("data:image"):
            # Base64 encoded image
            import base64

            header, data = image_input.split(",", 1)
            image_data = base64.b64decode(data)
            return Image.open(io.BytesIO(image_data))
    raise ValueError(f"Cannot load image from: {type(image_input)}")


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    forced = {
        # force return_tensors to None for input_ids
        "return_tensors": None,
    }
    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    result.update(forced)

    # set return_tensors="pt" for modality-specific outputs
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def process_vision_info(prompt, processor):
    # TODO: temporary solution, will write image utils for slime later
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def process_vision_info_internvl(prompt, processor=None):
    """Process vision info for InternVL models.

    InternVL uses a different format than Qwen-VL:
    - Images are loaded as PIL Images
    - No special patch size processing needed
    - Returns images list directly
    """
    images = []

    if isinstance(prompt, list):
        # Conversation format: list of message dicts
        for message in prompt:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_data = item.get("image")
                        if image_data:
                            images.append(load_image(image_data))
            elif isinstance(content, str):
                # Check for image placeholders in string content
                pass

    return {"images": images, "videos": []}


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"
