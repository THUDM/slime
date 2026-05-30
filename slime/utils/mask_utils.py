import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase, ProcessorMixin


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # Return the length from the first supervised token to the end of each mask.
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


@dataclass(frozen=True)
class LossMaskResult:
    token_ids: list[int]
    loss_mask: list[int]
    response_start: int

    @property
    def response_length(self) -> int:
        return len(self.token_ids) - self.response_start

    @property
    def response_loss_mask(self) -> list[int]:
        return self.loss_mask[self.response_start :]


@dataclass(frozen=True)
class RoleMarkerPattern:
    assistant_starts: tuple[str, ...]
    boundaries: tuple[str, ...]


class MultiTurnLossMaskGenerator:
    """Build SFT loss masks from chat-template assistant masks.

    The tokenizer chat template is the source of truth for assistant-generated
    spans. Non-assistant context such as tool responses, observations, and
    compacted history stays in the response tail with loss_mask=0.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        chat_template_kwargs: dict | None = None,
        chat_template_processor: ProcessorMixin | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_processor = chat_template_processor or tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}

    ROLE_MARKER_PATTERNS = (
        RoleMarkerPattern(
            assistant_starts=("<|assistant|>",),
            boundaries=("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|tool|>"),
        ),
        RoleMarkerPattern(
            assistant_starts=("<|im_start|>assistant\n",),
            boundaries=(
                "<|im_start|>system\n",
                "<|im_start|>user\n",
                "<|im_start|>assistant\n",
                "<|im_start|>tool\n",
                "<|im_start|>observation\n",
            ),
        ),
        RoleMarkerPattern(
            assistant_starts=("<｜Assistant｜>", "<｜tool▁outputs▁end｜>"),
            boundaries=(
                "<｜System｜>",
                "<｜User｜>",
                "<｜Assistant｜>",
                "<｜tool▁outputs▁begin｜>",
                "<｜tool▁output▁begin｜>",
            ),
        ),
    )

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return get_response_lengths(loss_masks)

    @staticmethod
    def _to_list(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    @staticmethod
    def _first_assistant_mask_index(assistant_mask: list[int]) -> int:
        for index, value in enumerate(assistant_mask):
            if value:
                return index
        raise ValueError(
            "Chat template produced no assistant mask tokens. "
            "Please wrap assistant-generated text/tool calls in `{% generation %}` blocks."
        )

    def _chat_template_supports_generation_blocks(self, tools: list[dict] = None) -> bool:
        chat_template = self.chat_template_kwargs.get("chat_template")
        if chat_template is None:
            get_chat_template = getattr(self.chat_template_processor, "get_chat_template", None)
            if get_chat_template is not None:
                try:
                    chat_template = get_chat_template(chat_template=None, tools=tools)
                except ValueError:
                    chat_template = None
            else:
                chat_template = getattr(self.chat_template_processor, "chat_template", None)
                if chat_template is None:
                    chat_template = getattr(self.tokenizer, "chat_template", None)
        if not isinstance(chat_template, str):
            return True
        return re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template) is not None

    def _apply_chat_template(self, messages: list[dict], *, tools: list[dict] = None, **kwargs):
        return self.chat_template_processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            **self.chat_template_kwargs,
            **kwargs,
        )

    def get_loss_mask_result(self, messages: list[dict], tools: list[dict] = None) -> LossMaskResult:
        if not self._chat_template_supports_generation_blocks(tools):
            return self._get_loss_mask_result_from_role_markers(messages, tools)

        tokenized = self._apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            tools=tools,
        )
        token_ids = self._to_list(tokenized["input_ids"])
        assistant_mask = tokenized.get("assistant_masks")
        if assistant_mask is None:
            raise ValueError(
                "`apply_chat_template(..., return_assistant_tokens_mask=True)` did not return `assistant_masks`. "
                "Please use a Transformers version/tokenizer template that supports assistant masks."
            )
        assistant_mask = self._to_list(assistant_mask)

        if len(token_ids) != len(assistant_mask):
            raise ValueError(
                "Chat template returned mismatched input_ids/assistant_masks lengths: "
                f"{len(token_ids)} != {len(assistant_mask)}"
            )
        if not any(assistant_mask):
            return self._get_loss_mask_result_from_role_markers(messages, tools, expected_token_ids=token_ids)

        response_start = self._first_assistant_mask_index(assistant_mask)
        loss_mask = [int(bool(value)) for value in assistant_mask]

        if any(message.get("step_loss_mask", 1) != 1 for message in messages):
            loss_mask = self._apply_step_loss_masks(messages, loss_mask)

        return LossMaskResult(token_ids=token_ids, loss_mask=loss_mask, response_start=response_start)

    def _get_loss_mask_result_from_role_markers(
        self, messages: list[dict], tools: list[dict] = None, expected_token_ids: list[int] | None = None
    ) -> LossMaskResult:
        rendered = self._apply_chat_template(
            messages,
            tokenize=False,
            tools=tools,
        )
        if not isinstance(rendered, str):
            raise ValueError("Chat template role-marker fallback expects a single rendered string.")

        try:
            tokenized = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        except NotImplementedError as exc:
            raise ValueError(
                "Chat template returned no assistant mask tokens, and role-marker fallback requires "
                "`return_offsets_mapping` support."
            ) from exc

        token_ids = self._to_list(tokenized["input_ids"])
        if expected_token_ids is not None and token_ids != expected_token_ids:
            raise ValueError(
                "Role-marker fallback tokenization does not match `apply_chat_template(..., tokenize=True)` output."
            )

        offset_mapping = tokenized.get("offset_mapping")
        if offset_mapping is None:
            raise ValueError(
                "Chat template returned no assistant mask tokens, and role-marker fallback requires offset mappings."
            )

        char_spans = self._assistant_char_spans_from_template_diffs(rendered, messages, tools)
        if char_spans is None:
            char_spans = self._assistant_char_spans_from_role_markers(rendered, messages)
        loss_mask = [0] * len(token_ids)
        for token_index, (token_start, token_end) in enumerate(offset_mapping):
            if token_end <= token_start:
                continue
            if any(token_start < span_end and token_end > span_start for span_start, span_end in char_spans):
                loss_mask[token_index] = 1

        if not any(loss_mask):
            raise ValueError(
                "Chat template produced no assistant mask tokens and role-marker fallback could not map "
                "assistant spans to tokens."
            )

        response_start = self._first_assistant_mask_index(loss_mask)
        if any(message.get("step_loss_mask", 1) != 1 for message in messages):
            loss_mask = self._apply_step_loss_masks(messages, loss_mask)

        return LossMaskResult(token_ids=token_ids, loss_mask=loss_mask, response_start=response_start)

    @staticmethod
    def _common_prefix_len(left: str, right: str) -> int:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    @staticmethod
    def _common_suffix_len(left: str, right: str, prefix_len: int) -> int:
        limit = min(len(left), len(right)) - prefix_len
        index = 0
        while index < limit and left[len(left) - index - 1] == right[len(right) - index - 1]:
            index += 1
        return index

    def _assistant_char_spans_from_template_diffs(
        self, rendered: str, messages: list[dict], tools: list[dict] = None
    ) -> list[tuple[int, int]] | None:
        spans = []
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            sentinel = f"SLIME_ASSISTANT_SPAN_{message_index}"
            replacement = {
                key: value
                for key, value in message.items()
                if key not in ("content", "reasoning_content", "tool_calls")
            }
            replacement["content"] = sentinel
            probe_messages = list(messages)
            probe_messages[message_index] = replacement

            try:
                probe_rendered = self._apply_chat_template(probe_messages, tokenize=False, tools=tools)
            except Exception:
                return None
            if not isinstance(probe_rendered, str) or sentinel not in probe_rendered:
                return None

            span_start = self._common_prefix_len(rendered, probe_rendered)
            suffix_len = self._common_suffix_len(rendered, probe_rendered, span_start)
            span_end = len(rendered) - suffix_len
            if span_start >= span_end:
                return None
            spans.append((span_start, span_end))

        return spans or None

    @staticmethod
    def _first_marker_after(rendered: str, markers: tuple[str, ...], start: int) -> tuple[int, str] | None:
        matches = [(pos, marker) for marker in markers if (pos := rendered.find(marker, start)) >= 0]
        return min(matches, key=lambda item: item[0]) if matches else None

    @classmethod
    def _assistant_char_spans_from_role_markers(cls, rendered: str, messages: list[dict]) -> list[tuple[int, int]]:
        spans = []
        cursor = 0

        for message in messages:
            if message.get("role") != "assistant":
                continue

            start_match = cls._first_marker_after(
                rendered,
                tuple(marker for pattern in cls.ROLE_MARKER_PATTERNS for marker in pattern.assistant_starts),
                cursor,
            )
            if start_match is None:
                raise ValueError(
                    "Chat template produced no assistant mask tokens and role-marker fallback could not locate "
                    "a supported assistant role marker."
                )

            header_pos, assistant_header = start_match
            span_start = header_pos + len(assistant_header)
            boundary_markers = tuple(marker for pattern in cls.ROLE_MARKER_PATTERNS for marker in pattern.boundaries)
            boundary_match = cls._first_marker_after(rendered, boundary_markers, span_start)
            span_end = boundary_match[0] if boundary_match is not None else len(rendered)
            spans.append((span_start, span_end))
            cursor = span_end

        if not spans:
            raise ValueError("Cannot build a loss mask because the message list contains no assistant messages.")
        return spans

    @staticmethod
    def _get_mask_spans(mask: list[int]) -> list[tuple[int, int]]:
        spans = []
        start = None
        for index, value in enumerate(mask):
            if value and start is None:
                start = index
            elif not value and start is not None:
                spans.append((start, index))
                start = None
        if start is not None:
            spans.append((start, len(mask)))
        return spans

    def _apply_step_loss_masks(self, messages: list[dict], loss_mask: list[int]) -> list[int]:
        """Apply per-assistant step masks when the template emits one span per assistant turn."""
        spans = self._get_mask_spans(loss_mask)

        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        if len(spans) != len(assistant_messages):
            raise ValueError(
                "`step_loss_mask` requires one contiguous assistant mask span per assistant message. "
                f"Got {len(spans)} span(s) for {len(assistant_messages)} assistant message(s)."
            )

        masked = list(loss_mask)
        for (start, end), message in zip(spans, assistant_messages, strict=True):
            if message.get("step_loss_mask", 1) == 1:
                continue
            masked[start:end] = [0] * (end - start)
        return masked

    def get_loss_mask(self, messages: list[dict], tools: list[dict] = None) -> tuple[list[int], list[int]]:
        result = self.get_loss_mask_result(messages, tools)
        return result.token_ids, result.loss_mask

    def get_loss_mask_with_multimodal_alignment(
        self, messages: list[dict], input_ids: list[int], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        text = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text.append({**msg, "content": " ".join(text_parts)})
            else:
                text.append(msg)

        _, loss_mask_text = self.get_loss_mask(text, tools=tools)
        diff = len(input_ids) - len(loss_mask_text)
        assert diff >= 0, (
            f"input_ids (length={len(input_ids)}) is shorter than text loss_mask (length={len(loss_mask_text)}) "
            f"Please check if processor and tokenizer tokenization are consistent."
        )
        return input_ids, [0] * diff + loss_mask_text

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts
