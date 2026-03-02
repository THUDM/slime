import base64
import io
import logging
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# InternVL image processing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


# ============== InternVL Image Processing (from official example) ==============

def _build_internvl_transform(input_size):
    """Build transform for InternVL image processing."""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio for dynamic preprocessing."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic preprocess for InternVL - split image into patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_internvl(image_file, input_size=448, max_num=12):
    """Load and preprocess image for InternVL model."""
    if isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    transform = _build_internvl_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLProcessorWrapper:
    """
    A simple processor wrapper for InternVL models.
    Mimics the interface of HuggingFace processors but uses InternVL's native image processing.
    """

    def __init__(self, tokenizer, input_size=448, max_num=12):
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.max_num = max_num
        # For compatibility checks
        self._is_internvl = True

    def __call__(self, text=None, images=None, **kwargs):
        """Process text and images for InternVL."""
        result = {}

        # Process text
        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
            result.update(text_inputs)

        # Process images
        if images is not None:
            if not isinstance(images, list):
                images = [images]

            all_pixel_values = []
            num_patches_list = []

            for img in images:
                pixel_values = load_image_internvl(img, self.input_size, self.max_num)
                num_patches_list.append(pixel_values.shape[0])
                all_pixel_values.append(pixel_values)

            if all_pixel_values:
                result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
                result["num_patches_list"] = num_patches_list

        return result

    @property
    def image_processor(self):
        """Return self as image processor for compatibility."""
        return self

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


def is_internvl_model(processor) -> bool:
    """Check if the processor belongs to an InternVL model."""
    if processor is None:
        return False
    # Check for our custom wrapper
    if hasattr(processor, "_is_internvl") and processor._is_internvl:
        return True
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

    # Remove empty lists to avoid processor errors (e.g., empty videos list causes IndexError)
    for key in ("images", "videos"):
        if key in result and (result[key] is None or len(result[key]) == 0):
            del result[key]

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
