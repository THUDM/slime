from slime.utils.mask_utils import MultiTurnLossMaskGenerator


class FakeQwen35Tokenizer:
    """A tiny char-level tokenizer that models the Qwen3.5 assistant formatting rule.

    The critical behavior we need for these tests is:
    1. `add_generation_prompt=True` appends `<|im_start|>assistant\n<think>\n`
    2. Only assistant turns after the last non-tool user query are wrapped in
       `<think>...</think>`
    """

    assistant_generation_prompt = "<|im_start|>assistant\n<think>\n"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": [ord(ch) for ch in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        tools=None,
        add_generation_prompt=False,
        return_dict=False,
        add_special_tokens=False,
        **kwargs,
    ):
        rendered = self.render(messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return [ord(ch) for ch in rendered]
        return rendered

    def render(self, messages, add_generation_prompt=False):
        rendered, _ = self.render_with_expected_mask(messages, add_generation_prompt=add_generation_prompt)
        return rendered

    def render_with_expected_mask(self, messages, add_generation_prompt=False):
        if not messages:
            raise ValueError("No messages provided.")

        pieces = []
        mask = []
        last_query_index = self._find_last_query_index(messages)
        has_assistant = any(message["role"] == "assistant" for message in messages)
        if has_assistant and last_query_index is None:
            raise ValueError("No user query found in messages.")

        for index, message in enumerate(messages):
            role = message["role"]
            if role == "system":
                if index != 0:
                    raise ValueError("System message must be at the beginning.")
                piece = f"<|im_start|>system\n{message['content']}<|im_end|>\n"
                pieces.append(piece)
                mask.extend([0] * len(piece))
                continue

            if role == "user":
                piece = f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                pieces.append(piece)
                mask.extend([0] * len(piece))
                continue

            if role == "tool":
                piece = ""
                if index > 0 and messages[index - 1]["role"] != "tool":
                    piece += "<|im_start|>user"
                piece += f"\n<tool_response>\n{message['content']}\n</tool_response>"
                if index == len(messages) - 1 or messages[index + 1]["role"] != "tool":
                    piece += "<|im_end|>\n"
                pieces.append(piece)
                mask.extend([0] * len(piece))
                continue

            if role != "assistant":
                raise NotImplementedError(f"Unsupported role in test tokenizer: {role}")

            reasoning, answer = self._split_assistant_content(message["content"])
            if index > last_query_index:
                prefix = "<|im_start|>assistant\n<think>\n"
                target = f"{reasoning}\n</think>\n\n{answer}"
            else:
                prefix = "<|im_start|>assistant\n"
                target = answer

            target += self._render_tool_calls(answer, message.get("tool_calls"))
            target += "<|im_end|>\n"

            pieces.append(prefix + target)
            mask.extend([0] * len(prefix))
            mask.extend([1] * len(target))

        if add_generation_prompt:
            pieces.append(self.assistant_generation_prompt)
            mask.extend([0] * len(self.assistant_generation_prompt))

        return "".join(pieces), mask

    @staticmethod
    def _split_assistant_content(content):
        if "</think>" not in content:
            return "", content
        reasoning = content.split("</think>")[0].split("<think>")[-1].strip("\n")
        answer = content.split("</think>")[-1].lstrip("\n")
        return reasoning, answer

    @staticmethod
    def _render_tool_calls(answer, tool_calls):
        if not tool_calls:
            return ""

        pieces = []
        for index, tool_call in enumerate(tool_calls):
            function_call = tool_call.get("function", tool_call)
            function_name = function_call["name"]
            if index == 0:
                if answer.strip():
                    pieces.append(f"\n\n<tool_call>\n<function={function_name}>\n")
                else:
                    pieces.append(f"<tool_call>\n<function={function_name}>\n")
            else:
                pieces.append(f"\n<tool_call>\n<function={function_name}>\n")

            for argument_name, argument_value in function_call.get("arguments", {}).items():
                pieces.append(f"<parameter={argument_name}>\n")
                pieces.append(str(argument_value))
                pieces.append("\n</parameter>\n")

            pieces.append("</function>\n</tool_call>")

        return "".join(pieces)

    @staticmethod
    def _find_last_query_index(messages):
        last_query_index = None
        for index, message in enumerate(messages):
            if message["role"] == "user":
                last_query_index = index
        return last_query_index


def test_qwen3_and_qwen3_5_match_on_single_turn_qwen35_data():
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "<think>REASONING</think>\nANSWER"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    for loss_mask_type in ("qwen3", "qwen3_5"):
        generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=loss_mask_type)
        token_ids, loss_mask = generator.get_loss_mask(messages)
        assert token_ids == expected_token_ids
        assert loss_mask == expected_mask


def test_qwen3_and_qwen3_5_diverge_on_multi_turn_qwen35_data():
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER_1"},
        {"role": "assistant", "content": "ANSWER_1"},
        {"role": "user", "content": "USER_2"},
        {"role": "assistant", "content": "<think>REASONING_2</think>\nANSWER_2"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    qwen3_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    qwen35_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")

    qwen3_token_ids, qwen3_loss_mask = qwen3_generator.get_loss_mask(messages)
    qwen35_token_ids, qwen35_loss_mask = qwen35_generator.get_loss_mask(messages)

    assert qwen3_token_ids != qwen35_token_ids
    assert qwen3_loss_mask != qwen35_loss_mask

    # `qwen3` is incompatible with the real Qwen3.5 full-text rendering:
    # it rebuilds each assistant turn in isolation and fabricates a `<think>` block
    # for the earlier assistant answer.
    assert qwen3_token_ids != expected_token_ids
    assert qwen3_loss_mask != expected_mask

    # `qwen3_5` matches the expected full-text rendering and supervises every
    # assistant turn in the full conversation.
    assert qwen35_token_ids == expected_token_ids
    assert qwen35_loss_mask == expected_mask

    assert qwen3_generator.get_text_from_loss_mask(qwen3_token_ids, qwen3_loss_mask) == [
        "\n</think>\n\nANSWER_1<|im_end|>\n",
        "REASONING_2\n</think>\n\nANSWER_2<|im_end|>\n",
    ]
    expected_selected_texts = [
        "ANSWER_1<|im_end|>\n",
        "REASONING_2\n</think>\n\nANSWER_2<|im_end|>\n",
    ]
    assert qwen35_generator.get_text_from_loss_mask(qwen35_token_ids, qwen35_loss_mask) == expected_selected_texts
    assert qwen35_generator.get_text_from_loss_mask(expected_token_ids, expected_mask) == expected_selected_texts


def test_qwen3_5_matches_expected_mask_for_tool_call_flow():
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
        {
            "role": "assistant",
            "content": "TOOL_CALL",
            "tool_calls": [{"function": {"name": "terminal", "arguments": {"command": "ls"}}}],
        },
        {"role": "tool", "content": "README.md"},
        {"role": "assistant", "content": "<think>REASONING</think>\nFINAL"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    assert generator.get_text_from_loss_mask(token_ids, loss_mask) == [
        "\n</think>\n\nTOOL_CALL\n\n<tool_call>\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call><|im_end|>\n",
        "REASONING\n</think>\n\nFINAL<|im_end|>\n",
    ]
