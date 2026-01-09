import base64
import io
import logging

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


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
    # temporary solution, will write image utils for slime later
    from qwen_vl_utils import process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def prepare_model_inputs(
    messages,
    tokenizer,
    processor,
    metadata: dict | None = None,
    apply_chat_template: bool = True,
    apply_chat_template_kwargs: dict | None = None,
):
    """Build model input ids and processed multimodal inputs from chat messages."""
    chat_kwargs = apply_chat_template_kwargs or {}
    if apply_chat_template:
        if processor is not None and hasattr(processor, "apply_chat_template"):
            text = processor.apply_chat_template(messages, tokenize=False, **chat_kwargs)
        else:
            text = tokenizer.apply_chat_template(messages, tokenize=False, **chat_kwargs)
    else:
        text = messages if isinstance(messages, str) else tokenizer.apply_chat_template(messages, tokenize=False)

    if processor is None:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        return input_ids, {"multimodal_inputs": None}

    multimodal_inputs = process_vision_info(messages, processor)
    if metadata and "images" in metadata and not multimodal_inputs.get("images"):
        multimodal_inputs = {**multimodal_inputs, "images": metadata["images"]}

    processor_output = processor(text=text, **multimodal_inputs)
    input_ids = processor_output["input_ids"][0]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    extra = {k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]}
    return input_ids, {"multimodal_inputs": extra}


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
