from slime.utils.mask_utils import MultiTurnLossMaskGenerator


class FakeGLM5Tokenizer:
    """A small char-level tokenizer that models GLM5 chat template boundaries."""

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
        rendered = self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return [ord(ch) for ch in rendered]
        return rendered

    def render(self, messages, tools=None, add_generation_prompt=False):
        rendered, _ = self.render_with_expected_mask(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )
        return rendered

    def render_with_expected_mask(self, messages, tools=None, add_generation_prompt=False):
        pieces = ["[gMASK]<sop>"]
        mask = [0] * len(pieces[0])
        last_user_index = self._find_last_user_index(messages)

        if tools:
            tools_text = "<|system|># Tools" + "".join(str(tool) for tool in tools)
            pieces.append(tools_text)
            mask.extend([0] * len(tools_text))

        for index, message in enumerate(messages):
            role = message["role"]

            if role == "system":
                marker = "<|system|>"
                piece = f"{marker}{self._visible_text(message['content'])}"
                pieces.append(piece)
                mask.extend(self._role_mask(marker, piece, messages, index))
                continue

            if role == "user":
                marker = "<|user|>"
                piece = f"{marker}{self._visible_text(message['content'])}"
                pieces.append(piece)
                mask.extend(self._role_mask(marker, piece, messages, index))
                continue

            if role == "tool":
                marker = "<|observation|>"
                piece = f"{marker}<tool_response>{self._visible_text(message['content'])}</tool_response>"
                pieces.append(piece)
                mask.extend(self._role_mask(marker, piece, messages, index))
                continue

            if role != "assistant":
                raise NotImplementedError(f"Unsupported role in test tokenizer: {role}")

            prefix = "<|assistant|>"
            pieces.append(prefix)
            mask.extend([0] * len(prefix))

            reasoning, content = self._split_assistant_content(self._visible_text(message.get("content", "")))
            if reasoning and index > last_user_index:
                think_prefix = "<think>"
                piece = f"{think_prefix}{reasoning}</think>"
                pieces.append(piece)
                if message.get("step_loss_mask", 1) == 1:
                    mask.extend([0] * len(think_prefix))
                    mask.extend([1] * (len(piece) - len(think_prefix)))
                else:
                    mask.extend([0] * len(piece))
            else:
                no_think_prefix = "</think>"
                pieces.append(no_think_prefix)
                mask.extend([0] * len(no_think_prefix))

            if content.strip():
                piece = content.strip()
                pieces.append(piece)
                mask.extend([message.get("step_loss_mask", 1)] * len(piece))

            tool_call_text = self._render_tool_calls(message.get("tool_calls"))
            if tool_call_text:
                pieces.append(tool_call_text)
                mask.extend([message.get("step_loss_mask", 1)] * len(tool_call_text))

        if add_generation_prompt:
            piece = "<|assistant|><think>"
            pieces.append(piece)
            mask.extend([0] * len(piece))

        return "".join(pieces), mask

    @staticmethod
    def _visible_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text.append(item.get("text", ""))
                elif isinstance(item, str):
                    text.append(item)
            return "".join(text)
        return "" if content is None else str(content)

    @staticmethod
    def _split_assistant_content(content):
        if "</think>" not in content:
            return "", content
        reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        answer = content.split("</think>")[-1].lstrip("\n")
        return reasoning, answer

    @staticmethod
    def _render_tool_calls(tool_calls):
        if not tool_calls:
            return ""

        pieces = []
        for tool_call in tool_calls:
            function_call = tool_call.get("function", tool_call)
            pieces.append(f"<tool_call>{function_call['name']}")
            for key, value in function_call.get("arguments", {}).items():
                pieces.append(f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>")
            pieces.append("</tool_call>")
        return "".join(pieces)

    @staticmethod
    def _find_last_user_index(messages):
        last_user_index = -1
        for index, message in enumerate(messages):
            if message["role"] == "user":
                last_user_index = index
        return last_user_index

    @staticmethod
    def _role_mask(marker, piece, messages, index):
        if index > 0 and messages[index - 1]["role"] == "assistant" and messages[index - 1].get("step_loss_mask", 1) == 1:
            return [1] * len(marker) + [0] * (len(piece) - len(marker))
        return [0] * len(piece)


def test_glm5_loss_mask_matches_multi_turn_rendering():
    tokenizer = FakeGLM5Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER_1"},
        {"role": "assistant", "content": "<think>OLD_REASONING</think>\nANSWER_1"},
        {"role": "user", "content": "USER_2"},
        {"role": "assistant", "content": "<think>REASONING_2</think>\nANSWER_2"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_text += "<|user|>"
    expected_mask += [1] * len("<|user|>")
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    assert generator.get_text_from_loss_mask(token_ids, loss_mask) == [
        "ANSWER_1<|user|>",
        "REASONING_2</think>ANSWER_2<|user|>",
    ]


def test_glm5_loss_mask_handles_tool_calls_and_step_loss_mask():
    tokenizer = FakeGLM5Tokenizer()
    messages = [
        {"role": "user", "content": "USER"},
        {
            "role": "assistant",
            "content": "CALL",
            "tool_calls": [{"function": {"name": "terminal", "arguments": {"command": "ls"}}}],
        },
        {"role": "tool", "content": "README.md"},
        {"role": "assistant", "content": "FINAL", "step_loss_mask": 0},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    assert generator.get_text_from_loss_mask(token_ids, loss_mask) == [
        "CALL<tool_call>terminal<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call><|observation|>",
    ]
