from slime.utils.mask_utils import MultiTurnLossMaskGenerator


class FakeGlm5Tokenizer:
    """A tiny char-level tokenizer that models the GLM-5 chat template formatting.

    GLM-5 uses role-token delimiters with no closing tags:
        [gMASK]<sop><|system|>...<|user|>...<|assistant|></think>content...

    Key behaviors:
    1. Sequence starts with ``[gMASK]<sop>``
    2. Role tokens: ``<|system|>``, ``<|user|>``, ``<|assistant|>``, ``<|observation|>``
    3. No closing tags — messages end at the next role token or end of string
    4. Assistant turns always start with ``</think>`` (thinking disabled)
    5. Tool calls use ``<tool_call>...<arg_key>...<arg_value>...</tool_call>``
    6. Tool responses use ``<|observation|><tool_response>...</tool_response>``
    """

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
        pieces = []
        mask = []

        # GLM-5 prefix
        prefix = "[gMASK]<sop>"
        pieces.append(prefix)
        mask.extend([0] * len(prefix))

        # Tool instructions go into the system message
        tool_instruction = self._build_tool_instructions(tools) if tools else ""

        for index, message in enumerate(messages):
            role = message["role"]

            if role == "system":
                piece = f"<|system|>{message['content']}{tool_instruction}"
                pieces.append(piece)
                mask.extend([0] * len(piece))

            elif role == "user":
                piece = f"<|user|>{message['content']}"
                pieces.append(piece)
                mask.extend([0] * len(piece))

            elif role == "assistant":
                header = "<|assistant|></think>"
                content = message.get("content", "") or ""
                tool_calls_text = self._render_tool_calls(message.get("tool_calls"))
                target = f"{content}{tool_calls_text}"

                pieces.append(header)
                mask.extend([0] * len(header))

                pieces.append(target)
                if message.get("step_loss_mask", 1) != 1:
                    mask.extend([0] * len(target))
                else:
                    mask.extend([1] * len(target))

            elif role == "tool":
                # First tool response in a group gets <|observation|>
                if index == 0 or messages[index - 1]["role"] != "tool":
                    piece = f"<|observation|><tool_response>{message['content']}</tool_response>"
                else:
                    piece = f"<tool_response>{message['content']}</tool_response>"
                pieces.append(piece)
                mask.extend([0] * len(piece))

        if add_generation_prompt:
            gen = "<|assistant|><think>"
            pieces.append(gen)
            mask.extend([0] * len(gen))

        return "".join(pieces), mask

    @staticmethod
    def _build_tool_instructions(tools):
        if not tools:
            return ""
        import json

        tool_specs = "\n".join(json.dumps(tool) for tool in tools)
        return (
            "\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "<tools>\n"
            f"{tool_specs}\n"
            "</tools>"
        )

    @staticmethod
    def _render_tool_calls(tool_calls):
        if not tool_calls:
            return ""
        pieces = []
        for tc in tool_calls:
            func = tc.get("function", tc)
            pieces.append(f"<tool_call>{func['name']}")
            for k, v in func.get("arguments", {}).items():
                val = v if isinstance(v, str) else str(v)
                pieces.append(f"<arg_key>{k}</arg_key><arg_value>{val}</arg_value>")
            pieces.append("</tool_call>")
        return "".join(pieces)


def test_glm5_single_turn():
    """Basic single-turn: only assistant content is masked."""
    tokenizer = FakeGlm5Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "ANSWER"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    selected = generator.get_text_from_loss_mask(token_ids, loss_mask)
    assert selected == ["ANSWER"]


def test_glm5_multi_turn():
    """Multi-turn: each assistant turn is independently masked."""
    tokenizer = FakeGlm5Tokenizer()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER_1"},
        {"role": "assistant", "content": "ANSWER_1"},
        {"role": "user", "content": "USER_2"},
        {"role": "assistant", "content": "ANSWER_2"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    selected = generator.get_text_from_loss_mask(token_ids, loss_mask)
    assert selected == ["ANSWER_1", "ANSWER_2"]


def test_glm5_step_loss_mask():
    """step_loss_mask=0 suppresses loss on specific assistant turns."""
    tokenizer = FakeGlm5Tokenizer()
    messages = [
        {"role": "user", "content": "USER_1"},
        {"role": "assistant", "content": "SKIP_THIS", "step_loss_mask": 0},
        {"role": "user", "content": "USER_2"},
        {"role": "assistant", "content": "KEEP_THIS"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    selected = generator.get_text_from_loss_mask(token_ids, loss_mask)
    assert selected == ["KEEP_THIS"]


def test_glm5_tool_call_flow():
    """Tool calling: assistant tool calls are masked, tool responses are not."""
    tokenizer = FakeGlm5Tokenizer()
    messages = [
        {"role": "user", "content": "USER"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": '{"temp": 22}'},
        {"role": "assistant", "content": "It is 22C in Paris."},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    selected = generator.get_text_from_loss_mask(token_ids, loss_mask)
    assert selected == [
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        "It is 22C in Paris.",
    ]


def test_glm5_tool_call_with_tools_schema():
    """Tool calling with tools schema injected into system message."""
    tokenizer = FakeGlm5Tokenizer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}],
        },
        {"role": "tool", "content": '{"temp": 15}'},
        {"role": "assistant", "content": "15C in NYC."},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages, tools=tools)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    token_ids, loss_mask = generator.get_loss_mask(messages, tools=tools)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    selected = generator.get_text_from_loss_mask(token_ids, loss_mask)
    assert selected == [
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>NYC</arg_value></tool_call>",
        "15C in NYC.",
    ]


def test_glm5_no_system_message():
    """Conversation without system message."""
    tokenizer = FakeGlm5Tokenizer()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    expected_text, expected_mask = tokenizer.render_with_expected_mask(messages)
    expected_token_ids = tokenizer(expected_text, add_special_tokens=False)["input_ids"]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert token_ids == expected_token_ids
    assert loss_mask == expected_mask
    selected = generator.get_text_from_loss_mask(token_ids, loss_mask)
    assert selected == ["Hi!"]


def test_glm5_lengths_match():
    """token_ids and loss_mask always have the same length."""
    tokenizer = FakeGlm5Tokenizer()
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2", "step_loss_mask": 0},
        {"role": "user", "content": "U3"},
        {"role": "assistant", "content": "A3"},
    ]

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="glm5")
    token_ids, loss_mask = generator.get_loss_mask(messages)

    assert len(token_ids) == len(loss_mask)
