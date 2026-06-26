import ast
import pathlib

from slime.utils.mask_utils import MultiTurnLossMaskGenerator


class FakeGemma4Tokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": [ord(ch) for ch in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return encoded

    def decode(self, token_ids):
        return "".join(chr(t) for t in token_ids)

    def get_added_vocab(self):
        return {}

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
        pieces = ["<bos>"]
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            content = message.get("content", "")
            reasoning = message.get("reasoning")
            body = ""
            if role == "model" and reasoning:
                body += f"<|channel>thought\n{reasoning}\n<channel|>"
            body += content
            pieces.append(f"<|turn>{role}\n{body}<turn|>\n")
        if add_generation_prompt:
            pieces.append("<|turn>model\n<|channel>thought\n<channel|>")
        return "".join(pieces)


def _masked_text(gen, messages):
    token_ids, mask = gen.get_loss_mask(messages)
    assert len(token_ids) == len(mask)
    return gen.tokenizer.decode([token_ids[i] for i in range(len(token_ids)) if mask[i] == 1])


def _unmasked_text(gen, messages):
    token_ids, mask = gen.get_loss_mask(messages)
    return gen.tokenizer.decode([token_ids[i] for i in range(len(token_ids)) if mask[i] == 0])


def _make_gen():
    return MultiTurnLossMaskGenerator(FakeGemma4Tokenizer(), tokenizer_type="gemma4")


def test_single_turn_masks_only_assistant():
    gen = _make_gen()
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]
    assert _masked_text(gen, msgs) == "Hello.<turn|>\n"


def test_multi_turn_masks_each_assistant_turn():
    gen = _make_gen()
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "It is 6."},
    ]
    assert _masked_text(gen, msgs) == "It is 4.<turn|>\nIt is 6.<turn|>\n"


def test_system_and_user_never_masked():
    gen = _make_gen()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "ASST"},
    ]
    unmasked = _unmasked_text(gen, msgs)
    assert "SYS" in unmasked
    assert "USR" in unmasked
    assert "ASST" not in unmasked


def test_turn_terminator_included_in_loss():
    gen = _make_gen()
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]
    assert "<turn|>" in _masked_text(gen, msgs)


def test_model_header_not_masked():
    gen = _make_gen()
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}]
    assert "<|turn>model" not in _masked_text(gen, msgs)


def test_step_loss_mask_excludes_turn():
    gen = _make_gen()
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "step_loss_mask": 0},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    masked = _masked_text(gen, msgs)
    assert "A1" not in masked
    assert masked == "A2<turn|>\n"


def test_thinking_channel_excluded_from_loss():
    gen = _make_gen()
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "ANSWER", "reasoning": "secret chain of thought"},
    ]
    masked = _masked_text(gen, msgs)
    assert "secret chain of thought" not in masked
    assert "ANSWER<turn|>\n" == masked


def test_consecutive_assistant_turns():
    gen = _make_gen()
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    masked = _masked_text(gen, msgs)
    assert "first" in masked
    assert "second" in masked


def test_response_lengths_helper():
    gen = _make_gen()
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]
    _, mask = gen.get_loss_mask(msgs)
    (length,) = gen.get_response_lengths([mask])
    assert length == sum(mask)
    assert length > 0


def test_gemma4_is_an_accepted_argparse_choice():
    arguments_py = pathlib.Path(__file__).resolve().parents[2] / "slime/utils/arguments.py"
    tree = ast.parse(arguments_py.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(isinstance(arg, ast.Constant) and arg.value == "--loss-mask-type" for arg in node.args):
            continue

        choices = next((kw.value for kw in node.keywords if kw.arg == "choices"), None)
        assert choices is not None, "no choices=[...] found for --loss-mask-type"
        assert "gemma4" in ast.literal_eval(choices)
        break
    else:
        raise AssertionError("could not locate --loss-mask-type in arguments.py")
