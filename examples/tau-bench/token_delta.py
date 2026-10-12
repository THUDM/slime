"""Token-exact multi-turn continuations for tau-bench.

SGLang samples token ids. Training must use those ids. Decoding the text and
running it back through the chat template is not token-exact: BPE does not
round-trip, and reasoning templates can rewrite earlier assistant turns.

Only the observation is tokenized here. It was never sampled. The returned ids
are the template tail after the assistant end token, so they concatenate onto
the sampled ids without retokenizing them.
"""

import math
from typing import Any

_MARKER = "\u241ftau-assistant-marker"
_PREFIX = (
    {"role": "user", "content": "user"},
    {"role": "assistant", "content": "assistant"},
)


def encode_token_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(token_id) for token_id in token_ids]


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def sampled_output_tokens(output: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Return the token ids SGLang sampled, paired with their logprobs.

    ``output["text"]`` is intentionally ignored. Re-encoding that string would
    build a different training target from the tokens the policy sampled.
    """
    meta = output.get("meta_info") or {}
    rows = meta.get("output_token_logprobs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SGLang response is missing output_token_logprobs; refusing to retokenize output text")

    token_ids: list[int] = []
    log_probs: list[float] = []
    for row in rows:
        try:
            log_prob = float(row[0])
            token_id = int(row[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("SGLang returned invalid output token ids or non-finite logprobs") from exc
        if token_id < 0 or not math.isfinite(log_prob):
            raise ValueError("SGLang returned invalid output token ids or non-finite logprobs")
        token_ids.append(token_id)
        log_probs.append(log_prob)

    output_ids = output.get("output_ids")
    if output_ids is not None and list(output_ids) != token_ids:
        raise ValueError("SGLang output_ids do not match output_token_logprobs")
    return token_ids, log_probs


def _render(tokenizer: Any, messages: list[dict[str, Any]], *, tools: Any, add_generation_prompt: bool) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if tools is not None:
        kwargs["tools"] = tools
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str):
        raise ValueError("apply_chat_template must return text when tokenize=False")
    return rendered


def assistant_end_token_id(tokenizer: Any, *, tools: Any = None) -> int:
    """Special token id that closes an assistant message in this chat template."""
    opened = [{"role": "user", "content": "user"}]
    closed_messages = [*opened, {"role": "assistant", "content": _MARKER}]
    closed = _render(tokenizer, closed_messages, tools=tools, add_generation_prompt=False)
    marker_at = closed.rfind(_MARKER)
    if marker_at < 0:
        raise ValueError("chat template dropped the assistant content; cannot locate the assistant end token")

    prefix_text = closed[: marker_at + len(_MARKER)]
    prefix_ids = encode_token_ids(tokenizer, prefix_text)
    closed_ids = encode_token_ids(tokenizer, closed)
    if closed_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "assistant end token merged with the assistant content; refusing to retokenize sampled tokens"
        )
    if len(closed_ids) == len(prefix_ids):
        raise ValueError("chat template does not append a token after assistant content")

    end_id = closed_ids[len(prefix_ids)]
    special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    if special_ids and end_id not in special_ids:
        raise ValueError("assistant message does not end with a special token; refusing to retokenize sampled tokens")
    return end_id


def _rindex(token_ids: list[int], token_id: int) -> int:
    for index in range(len(token_ids) - 1, -1, -1):
        if token_ids[index] == token_id:
            return index
    raise ValueError("assistant end token is missing from the closed assistant message")


def observation_suffix_token_ids(
    tokenizer: Any,
    observation_message: dict[str, Any],
    *,
    tools: Any = None,
    add_generation_prompt: bool,
    end_token_id: int | None = None,
) -> list[int]:
    """Template tokens that follow a closed assistant turn.

    The suffix belongs after the assistant end token. It includes the newline
    the template inserts after that token, the observation message, and, when
    requested, the next assistant generation prompt. Concatenate it onto the
    sampled assistant ids; do not re-encode those ids.
    """
    if observation_message.get("role") == "assistant":
        raise ValueError("observation suffix expects the environment message, not another assistant turn")

    end_id = assistant_end_token_id(tokenizer, tools=tools) if end_token_id is None else end_token_id
    prefix = [dict(message) for message in _PREFIX]
    anchor = _render(tokenizer, prefix, tools=tools, add_generation_prompt=False)
    full = _render(tokenizer, [*prefix, observation_message], tools=tools, add_generation_prompt=add_generation_prompt)
    anchor_ids = encode_token_ids(tokenizer, anchor)
    full_ids = encode_token_ids(tokenizer, full)
    end = _rindex(anchor_ids, end_id)
    if decode_token_ids(tokenizer, anchor_ids[end + 1 :]).strip():
        raise ValueError("assistant end token is not the closer of the assistant message; refusing to retokenize")
    if full_ids[: end + 1] != anchor_ids[: end + 1]:
        raise ValueError(
            "chat template rewrote the assistant turn while rendering the observation; refusing to retokenize"
        )
    suffix = full_ids[end + 1 :]
    if not suffix:
        raise ValueError("chat template appended no tokens for the observation")
    return suffix


def tokens_after_assistant(
    tokenizer: Any,
    output_ids: list[int],
    observation_message: dict[str, Any],
    *,
    tools: Any = None,
    add_generation_prompt: bool,
) -> list[int]:
    """Ids to append after sampled assistant tokens.

    When the sample already ends with the assistant end token, the suffix is
    only the observation. Otherwise that end token is inserted once, ahead of
    the observation, and the caller must leave it out of the loss.
    """
    end_id = assistant_end_token_id(tokenizer, tools=tools)
    suffix = observation_suffix_token_ids(
        tokenizer,
        observation_message,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        end_token_id=end_id,
    )
    if output_ids and output_ids[-1] == end_id:
        return suffix
    return [end_id, *suffix]


def assistant_message_text(tokenizer: Any, output_ids: list[int], *, tools: Any = None) -> str:
    """Decode sampled ids for tool parsing without changing the stored ids."""
    text = decode_token_ids(tokenizer, output_ids)
    if not output_ids:
        return text
    end_id = assistant_end_token_id(tokenizer, tools=tools)
    if output_ids[-1] != end_id:
        return text
    end_text = decode_token_ids(tokenizer, [end_id])
    if end_text and text.endswith(end_text):
        return text[: -len(end_text)]
    return text
