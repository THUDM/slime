import base64
import io
import json
import logging
from pathlib import Path
import time
from urllib.parse import urlparse

from PIL import Image, ImageFile
import requests
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Some generated PPT screenshots are mildly truncated but still decodable.
# Enable this before qwen_vl_utils opens image paths and returns lazy PIL objects.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14
IMAGE_DOWNLOAD_TIMEOUT = (5, 30)
IMAGE_DOWNLOAD_RETRIES = 3
IMAGE_DOWNLOAD_HEADERS = {
    "User-Agent": "slime-vlm-data-loader/1.0",
}


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


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _load_image_from_bytes(raw: bytes, source: str) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to decode image from {source}") from e


def _download_image(url: str) -> Image.Image:
    urls = [url]
    if url.startswith("https://"):
        urls.append("http://" + url[len("https://") :])

    last_error = None
    for candidate in urls:
        for attempt in range(1, IMAGE_DOWNLOAD_RETRIES + 1):
            try:
                response = requests.get(
                    candidate,
                    headers=IMAGE_DOWNLOAD_HEADERS,
                    timeout=IMAGE_DOWNLOAD_TIMEOUT,
                )
                response.raise_for_status()
                return _load_image_from_bytes(response.content, candidate)
            except Exception as e:
                last_error = e
                if attempt < IMAGE_DOWNLOAD_RETRIES:
                    time.sleep(0.5 * attempt)
                    continue
                logger.warning(
                    "Failed to load image URL %s after %d attempts: %s",
                    candidate,
                    IMAGE_DOWNLOAD_RETRIES,
                    e,
                )

    raise RuntimeError(f"Failed to load image URL {url}") from last_error


def _load_image_from_file(path: str) -> Image.Image:
    try:
        image = Image.open(path)
        image.load()
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to decode image from file {path}") from e


def _load_image_from_string(image_data: str) -> Image.Image:
    if image_data.startswith("data:"):
        _, encoded = image_data.split(",", 1)
        return _load_image_from_bytes(base64.b64decode(encoded), "data URI")

    if _is_url(image_data):
        return _download_image(image_data)

    try:
        raw = base64.b64decode(image_data)
        return _load_image_from_bytes(raw, "base64 string")
    except Exception:
        return _load_image_from_file(image_data)


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
                images.append(_load_image_from_string(image_data))
    return images


def process_vision_info(prompt, processor):
    """Extract PIL images (and videos) from the message list for training.

    Tries qwen_vl_utils first (Qwen VL family), falls back to generic
    extraction for other models (e.g. GLM-4.6V).
    """
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
