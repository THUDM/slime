from typing import Any


def get_token_delta(tokenizer: Any, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Return the tokens and loss mask contributed by the last chat message."""
    if not messages:
        raise ValueError("Cannot calculate a token delta for an empty conversation")

    is_assistant = messages[-1]["role"] == "assistant"
    curr = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    prev = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=is_assistant,
        tokenize=False,
    )

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
    loss_value = 1 if is_assistant else 0
    return new_tokens, [loss_value] * len(new_tokens)
