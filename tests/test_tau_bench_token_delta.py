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
