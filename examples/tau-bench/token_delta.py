from typing import Any


def get_token_delta(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    include_generation_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    """Return the tokens and loss mask contributed by the last chat message."""
    if not messages:
        raise ValueError("Cannot calculate a token delta for an empty conversation")

    is_assistant = messages[-1]["role"] == "assistant"
    curr = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

    if is_assistant:
        prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=False, tokenize=False)
        generation_prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
        if not generation_prompt.startswith(prev):
            raise ValueError("Adding the assistant generation prompt rewrote the rendered conversation")
        if not curr.startswith(generation_prompt):
            raise ValueError("The assistant response does not extend its generation prompt")

        generation_prompt_text = generation_prompt[len(prev) :]
        if not include_generation_prompt:
            new_text = curr[len(generation_prompt) :]
            new_tokens = tokenizer.encode(new_text, add_special_tokens=False)
            return new_tokens, [1] * len(new_tokens)

        new_text = curr[len(prev) :]
        new_tokens = tokenizer.encode(new_text, add_special_tokens=False)
        generation_prompt_length = len(tokenizer.encode(generation_prompt_text, add_special_tokens=False))
        masked_prefix_length = min(generation_prompt_length, len(new_tokens))
        loss_mask = [0] * masked_prefix_length
        loss_mask.extend([1] * (len(new_tokens) - masked_prefix_length))
        return new_tokens, loss_mask

    prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=False, tokenize=False)

    if curr.startswith(prev):
        new_text = curr[len(prev) :]
    elif messages[-1]["role"] == "user":
        # Reasoning templates such as Qwen3 can rewrite history when a new user
        # message arrives. Render that message independently instead of slicing
        # the rewritten conversation at the old conversation length.
        new_text = tokenizer.apply_chat_template(
            [messages[-1]],
            add_generation_prompt=False,
            tokenize=False,
        )
        if not curr.endswith(new_text):
            raise ValueError("The latest user message is not a standalone suffix of the rendered conversation")
    else:
        raise ValueError("The chat template rewrote history while calculating a non-user token delta")

    new_tokens = tokenizer.encode(new_text, add_special_tokens=False)
    return new_tokens, [0] * len(new_tokens)
