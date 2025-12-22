import base64
import io
import logging

import numpy as np
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)


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


def prepare_model_inputs(
    prompt, tokenizer, processor=None, metadata=None, apply_chat_template=False, apply_chat_template_kwargs=None
):
    """Prepare all inputs for model inference.

    Returns:
        tuple: (input_ids, extra_info)
            - input_ids: Token IDs for the prompt
            - extra_info: Dict with 'images', 'videos', 'multimodal_inputs' (or empty dict)
    """
    tools = metadata.get("tools") if metadata else None
    if isinstance(prompt, (list, np.ndarray)):
        assert (
            apply_chat_template
        ), f"apply_chat_template must be True when prompt is a list or numpy array, current prompt is {prompt}"
        formatted_prompt = tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **(apply_chat_template_kwargs or {}),
        )
    elif isinstance(prompt, str):
        assert (
            not apply_chat_template
        ), f"apply_chat_template must be False when prompt is a string, current prompt is {prompt}"
        formatted_prompt = prompt
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}, current prompt is {prompt}")

    if not processor:
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        return input_ids, {"formatted_prompt": formatted_prompt}
    else:
        # temporary solution, will write image utils for slime later
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(prompt)

        # Get input IDs with full prompt (text + multimodal)
        processor_output = processor(text=formatted_prompt, images=images, videos=videos)
        input_ids = processor_output["input_ids"][0]

        # Extract multimodal tokens (exclude text-related tokens)
        multimodal_inputs = {k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]}

        extra_info = {
            "formatted_prompt": formatted_prompt,
            "images": images,
            "videos": videos,
            "multimodal_inputs": multimodal_inputs,
        }

        return input_ids, extra_info


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as JPEG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
