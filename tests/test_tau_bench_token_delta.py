import importlib.util
import re
from pathlib import Path

import pytest


TOKEN_DELTA_PATH = Path(__file__).parents[1] / "examples" / "tau-bench" / "token_delta.py"


def _load_get_token_delta():
    if not TOKEN_DELTA_PATH.exists():
        pytest.fail("tau-bench token_delta helper does not exist")

    spec = importlib.util.spec_from_file_location("tau_bench_token_delta", TOKEN_DELTA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_token_delta


class HistoryRewritingTokenizer:
    """Minimal chat template that hides old reasoning after a new user turn."""

    @staticmethod
    def _strip_reasoning(content: str) -> str:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert tokenize is False
        last_user = max((i for i, message in enumerate(messages) if message["role"] == "user"), default=-1)
        rendered = []
        for i, message in enumerate(messages):
            content = message["content"]
            if message["role"] == "assistant" and i < last_user:
                content = self._strip_reasoning(content)
            rendered.append(f'<{message["role"]}>{content}</{message["role"]}>')
        if add_generation_prompt:
            rendered.append("<assistant>")
        return "".join(rendered)

    @staticmethod
    def encode(text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())

    @staticmethod
    def decode(token_ids):
        return bytes(token_ids).decode()


class BoundaryMergingTokenizer(HistoryRewritingTokenizer):
    """Tokenizer where the generation-prefix tail merges with a leading newline."""

    @staticmethod
    def encode(text, *, add_special_tokens):
        assert add_special_tokens is False
        raw = text.encode()
        token_ids = []
        index = 0
        while index < len(raw):
            if raw[index : index + 2] == b">\n":
                token_ids.append(1000)
                index += 2
            else:
                token_ids.append(raw[index])
                index += 1
        return token_ids


@pytest.mark.unit
def test_new_user_delta_survives_history_rewrite():
    get_token_delta = _load_get_token_delta()
    tokenizer = HistoryRewritingTokenizer()
    messages = [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "<think>first reasoning</think>first answer"},
        {"role": "user", "content": "second user must remain complete"},
    ]

    token_ids, loss_mask = get_token_delta(tokenizer, messages)

    assert tokenizer.decode(token_ids) == "<user>second user must remain complete</user>"
    assert loss_mask == [0] * len(token_ids)


@pytest.mark.unit
def test_assistant_delta_keeps_existing_append_only_behavior():
    get_token_delta = _load_get_token_delta()
    tokenizer = HistoryRewritingTokenizer()
    messages = [
        {"role": "user", "content": "user question"},
        {"role": "assistant", "content": "<think>reasoning</think>answer"},
    ]

    token_ids, loss_mask = get_token_delta(tokenizer, messages)

    assert tokenizer.decode(token_ids) == "<think>reasoning</think>answer</assistant>"
    assert loss_mask == [1] * len(token_ids)


@pytest.mark.unit
def test_accumulated_multiturn_tokens_keep_every_assistant_generation_prefix():
    get_token_delta = _load_get_token_delta()
    tokenizer = HistoryRewritingTokenizer()
    messages = [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "<think>first reasoning</think>first answer"},
        {"role": "user", "content": "second user"},
        {"role": "assistant", "content": "<think>second reasoning</think>second answer"},
    ]

    initial_prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)
    token_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)
    loss_mask = [0] * len(token_ids)

    for end in range(2, len(messages) + 1):
        include_generation_prompt = messages[end - 1]["role"] == "assistant" and end > 2
        delta_ids, delta_mask = get_token_delta(
            tokenizer,
            messages[:end],
            include_generation_prompt=include_generation_prompt,
        )
        token_ids.extend(delta_ids)
        loss_mask.extend(delta_mask)

    decoded = tokenizer.decode(token_ids)
    assert decoded.count("<assistant>") == 2
    assert decoded.count("</assistant>") == 2
    assert "first reasoning" in decoded
    assert "second reasoning" in decoded
    assert "second user" in decoded

    second_prefix = decoded.rindex("<assistant>")
    assert loss_mask[second_prefix : second_prefix + len("<assistant>")] == [0] * len("<assistant>")


@pytest.mark.unit
def test_accumulated_six_real_user_turns_keep_every_user_and_assistant_boundary():
    get_token_delta = _load_get_token_delta()
    tokenizer = HistoryRewritingTokenizer()
    messages = [{"role": "user", "content": "user 1"}]

    initial_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    token_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)
    loss_mask = [0] * len(token_ids)

    for turn in range(1, 7):
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>reasoning {turn}</think>answer {turn}",
            }
        )
        delta_ids, delta_mask = get_token_delta(
            tokenizer,
            messages,
            include_generation_prompt=turn > 1,
        )
        token_ids.extend(delta_ids)
        loss_mask.extend(delta_mask)

        if turn < 6:
            messages.append({"role": "user", "content": f"user {turn + 1}"})
            delta_ids, delta_mask = get_token_delta(tokenizer, messages)
            token_ids.extend(delta_ids)
            loss_mask.extend(delta_mask)

    decoded = tokenizer.decode(token_ids)
    assert decoded.count("<assistant>") == 6
    assert decoded.count("</assistant>") == 6

    for turn in range(1, 7):
        user_span = f"<user>user {turn}</user>"
        user_start = decoded.index(user_span)
        assert loss_mask[user_start : user_start + len(user_span)] == [0] * len(user_span)

        assistant_span = f"<think>reasoning {turn}</think>answer {turn}</assistant>"
        assistant_start = decoded.index(assistant_span)
        assert loss_mask[assistant_start : assistant_start + len(assistant_span)] == [1] * len(assistant_span)

        prefix_start = decoded.rfind("<assistant>", 0, assistant_start)
        assert loss_mask[prefix_start:assistant_start] == [0] * (assistant_start - prefix_start)


@pytest.mark.unit
def test_later_assistant_allows_bpe_merge_across_generation_prefix_boundary():
    get_token_delta = _load_get_token_delta()
    tokenizer = BoundaryMergingTokenizer()
    messages = [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "<think>first reasoning</think>first answer"},
        {"role": "user", "content": "second user"},
        {"role": "assistant", "content": "\nsecond answer"},
    ]

    token_ids, loss_mask = get_token_delta(
        tokenizer,
        messages,
        include_generation_prompt=True,
    )

    expected_text = "<assistant>\nsecond answer</assistant>"
    expected_ids = tokenizer.encode(expected_text, add_special_tokens=False)
    generation_prefix_length = len(tokenizer.encode("<assistant>", add_special_tokens=False))
    assert token_ids == expected_ids
    assert loss_mask == [0] * generation_prefix_length + [1] * (len(expected_ids) - generation_prefix_length)
