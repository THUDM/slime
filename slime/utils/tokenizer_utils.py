from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception:
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def prepare_model_inputs(prompt, tokenizer, processor=None, metadata=None, apply_chat_template_kwargs=None):
    """Prepare all inputs for model inference.

    Returns:
        tuple: (input_ids, encoding_info)
            - input_ids: Token IDs for the prompt
            - encoding_info: Dict with 'images', 'videos', 'multimodal_inputs' (or empty dict)
    """
    tools = metadata.get("tools") if metadata else None
    text_prompt = tokenizer.apply_chat_template(
        prompt,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **(apply_chat_template_kwargs or {}),
    )

    if not processor:
        input_ids = tokenizer.encode(text_prompt, add_special_tokens=False)
        return input_ids, {}
    else:
        # temporary solution, will write image utils for slime later
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(prompt)

        # Get input IDs with full prompt (text + multimodal)
        processor_output = processor(text=text_prompt, images=images, videos=videos)
        input_ids = processor_output["input_ids"][0]

        # Extract multimodal tokens (exclude text-related tokens)
        multimodal_inputs = {k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]}

        encoding_info = {
            "images": images,
            "videos": videos,
            "multimodal_inputs": multimodal_inputs,
        }

        return input_ids, encoding_info
