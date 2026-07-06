import json

from transformers import AutoTokenizer


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # return the lengths starting from the first occurrence of 1 to the end of each loss mask
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


def _normalize_chat_template_kwargs(chat_template_kwargs) -> dict:
    if chat_template_kwargs is None:
        return {}
    if isinstance(chat_template_kwargs, str):
        if not chat_template_kwargs.strip():
            return {}
        chat_template_kwargs = json.loads(chat_template_kwargs)
    if not isinstance(chat_template_kwargs, dict):
        raise TypeError(f"chat_template_kwargs must be a dict or JSON string, got {type(chat_template_kwargs)}")
    return dict(chat_template_kwargs)


class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen", chat_template_kwargs=None):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.chat_template_kwargs = _normalize_chat_template_kwargs(chat_template_kwargs)
        self.system_message_length = 0
        self.gen_token_length = 0
        if self.tokenizer_type in ("qwen", "qwen3"):
            self.system_message_length, self.gen_token_length = self.get_system_message_length()

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return get_response_lengths(loss_masks)

    def find_all_sublist_indices(self, main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages,
                add_special_tokens=False,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            if i == 0:
                message_ids = self.tokenizer.apply_chat_template(
                    [message], tokenize=True, tools=tools, return_dict=False
                )
            else:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
        prefix_token_ids = self.tokenizer.apply_chat_template([prefix_message], tokenize=True, return_dict=False)

        for i, message in enumerate(messages):
            if i == 0:
                tailed_message_ids = self.tokenizer.apply_chat_template(
                    [message, prefix_message],
                    tokenize=True,
                    tools=tools,
                    return_dict=False,
                )
                message_ids = tailed_message_ids[: -len(prefix_token_ids)]
            else:
                prefixed_message_ids = self.tokenizer.apply_chat_template(
                    [prefix_message, message],
                    tokenize=True,
                    return_dict=False,
                )
                message_ids = prefixed_message_ids[len(prefix_token_ids) :]

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3_5(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        rendered_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=tools,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        token_ids, loss_mask = self.get_loss_mask_from_rendered_qwen3_5(messages, rendered_text)

        expected_token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            tools=tools,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        if token_ids != expected_token_ids:
            raise ValueError(
                "Qwen3.5/Qwen3.6 rendered text tokenization does not match "
                "`apply_chat_template(..., tokenize=True)` output."
            )

        return token_ids, loss_mask

    def get_loss_mask_from_rendered_qwen3_5(
        self,
        messages: list[dict],
        rendered_text: str,
        input_ids: list[int] | None = None,
    ) -> tuple[list[int], list[int]]:
        tokenized = self.tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = tokenized["input_ids"]
        offset_mapping = tokenized.get("offset_mapping")

        if offset_mapping is None:
            raise ValueError(
                "Qwen3.5/Qwen3.6 loss mask generation requires a fast tokenizer with offset mapping support."
            )

        if input_ids is not None and token_ids != input_ids:
            mismatch_idx = next(
                (idx for idx, (actual, expected) in enumerate(zip(token_ids, input_ids)) if actual != expected),
                min(len(token_ids), len(input_ids)),
            )
            raise ValueError(
                "Rendered multimodal text tokenization does not match processor input_ids: "
                f"{len(token_ids)=}, {len(input_ids)=}, first_mismatch_index={mismatch_idx}. "
                "Please check processor visual-token expansion logic."
            )

        assistant_header = "<|im_start|>assistant\n"
        think_prefix = "<think>\n"
        empty_think_block = "<think>\n\n</think>\n\n"
        end_marker = "<|im_end|>"

        char_mask = [0] * len(rendered_text)
        cursor = 0

        def mark_span(start: int, end: int) -> None:
            for pos in range(max(start, 0), min(end, len(char_mask))):
                char_mask[pos] = 1

        for message in messages:
            if message["role"] != "assistant":
                continue

            header_pos = rendered_text.find(assistant_header, cursor)
            if header_pos < 0:
                raise ValueError(
                    "Failed to locate assistant message in rendered Qwen3.5/Qwen3.6 chat template."
                )

            content_start = header_pos + len(assistant_header)
            end_pos = rendered_text.find(end_marker, content_start)
            if end_pos < 0:
                raise ValueError("Failed to locate <|im_end|> for assistant message in rendered text.")

            span_end = end_pos + len(end_marker)
            if span_end < len(rendered_text) and rendered_text[span_end] == "\n":
                span_end += 1
            cursor = span_end

            if message.get("step_loss_mask", 1) != 1:
                continue

            # In non-thinking mode, Qwen3.6's generation prompt pre-fills the
            # empty think block. The model should learn only the answer/tool
            # call after it, not generate a duplicate closing </think>.
            if rendered_text[content_start : content_start + len(empty_think_block)] == empty_think_block:
                mask_start = content_start + len(empty_think_block)
            # In thinking mode, only the opening <think>\n is prompt scaffold;
            # reasoning, </think>, answer, and <|im_end|> are model targets.
            elif rendered_text[content_start : content_start + len(think_prefix)] == think_prefix:
                mask_start = content_start + len(think_prefix)
            else:
                mask_start = content_start

            mark_span(mask_start, span_end)

        char_mask_prefix_sum = [0]
        for value in char_mask:
            char_mask_prefix_sum.append(char_mask_prefix_sum[-1] + value)

        loss_mask = []
        for start, end in offset_mapping:
            if end <= start:
                loss_mask.append(0)
            else:
                loss_mask.append(1 if char_mask_prefix_sum[end] - char_mask_prefix_sum[start] > 0 else 0)

        return token_ids, loss_mask

    @staticmethod
    def _to_plain_list(value):
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return value

    @staticmethod
    def _grid_token_count(grid, merge_size: int) -> int:
        prod = 1
        for dim in grid:
            prod *= int(dim)
        return prod // (int(merge_size) ** 2)

    def _expand_image_tokens_in_rendered_text(
        self,
        rendered_text: str,
        image_grid_thw,
        *,
        image_token: str,
        merge_size: int,
    ) -> str:
        grids = self._to_plain_list(image_grid_thw)
        if not grids:
            return rendered_text

        pieces = []
        cursor = 0
        for grid in grids:
            image_pos = rendered_text.find(image_token, cursor)
            if image_pos < 0:
                raise ValueError(
                    "Failed to locate image token in rendered multimodal text while expanding visual tokens: "
                    f"{image_token=}, expanded_images={len(pieces)}."
                )

            num_image_tokens = self._grid_token_count(grid, merge_size)
            pieces.append(rendered_text[cursor:image_pos])
            pieces.append(image_token * num_image_tokens)
            cursor = image_pos + len(image_token)

        pieces.append(rendered_text[cursor:])
        return "".join(pieces)

    @staticmethod
    def _message_content_to_text(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        text_parts.append(item["text"])
                    elif item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
            return "".join(text_parts)
        return str(content)

    def _get_assistant_content_parts(self, message: dict) -> list[str]:
        content = self._message_content_to_text(message.get("content")).strip()
        reasoning_content = message.get("reasoning_content")
        parts = []

        if isinstance(reasoning_content, str):
            reasoning_content = reasoning_content.strip()
            if reasoning_content:
                parts.append(reasoning_content)
        elif "</think>" in content:
            reasoning_content = content.split("</think>", 1)[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
            if reasoning_content:
                parts.append(reasoning_content)
            content = content.split("</think>", 1)[-1].lstrip("\n")

        content = content.strip()
        if content:
            parts.append(content)

        return parts

    def gen_multi_turn_loss_mask_distill_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, tools=tools
        )
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask

    def get_loss_mask(self, messages: list[dict], tools: list[dict] = None) -> tuple[list[int], list[int]]:
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)

            return self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "qwen3_5":
            return self.gen_multi_turn_loss_mask_qwen3_5(messages, tools)
        elif self.tokenizer_type == "gemma4":
            return self.gen_multi_turn_loss_mask_gemma4(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_loss_mask_with_multimodal_alignment(
        self,
        messages: list[dict],
        input_ids: list[int],
        tools: list[dict] = None,
        rendered_text: str | None = None,
        image_grid_thw=None,
        image_token: str = "<|image_pad|>",
        image_merge_size: int = 2,
    ) -> tuple[list[int], list[int]]:
        if self.tokenizer_type == "qwen3_5" and rendered_text is not None:
            expanded_rendered_text = self._expand_image_tokens_in_rendered_text(
                rendered_text,
                image_grid_thw,
                image_token=image_token,
                merge_size=image_merge_size,
            )
            return self.get_loss_mask_from_rendered_qwen3_5(
                messages,
                expanded_rendered_text,
                input_ids=input_ids,
            )

        raise ValueError(
            "Multimodal loss-mask alignment requires the actual rendered multimodal text for Qwen3.5/Qwen3.6. "
            "Pass rendered_text plus processor image_grid_thw so visual token positions can be masked directly."
        )

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
