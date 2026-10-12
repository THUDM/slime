import importlib.util
from pathlib import Path

import pytest

NUM_GPUS = 0


TOKEN_DELTA_PATH = Path(__file__).parents[1] / "examples" / "tau-bench" / "token_delta.py"


def _load_token_delta():
    if not TOKEN_DELTA_PATH.exists():
        pytest.fail("tau-bench token_delta helper does not exist")

    spec = importlib.util.spec_from_file_location("tau_bench_token_delta", TOKEN_DELTA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QwenLikeTokenizer:
    """Chat template with atomic special tokens, close enough to Qwen to splice on."""

    special_tokens = (
        ("<|im_start|>", 1000),
        ("<|im_end|>", 1001),
        ("<tool_response>", 1002),
        ("</tool_response>", 1003),
    )

    def __init__(self, *, inject_think: bool = False, newline_before_end: bool = False):
        self.inject_think = inject_think
        self.newline_before_end = newline_before_end

    @property
    def all_special_ids(self) -> list[int]:
        return [token_id for _, token_id in self.special_tokens]

    def format_message(self, role: str, content: str) -> str:
        if self.newline_before_end:
            return f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, tools=None):
        assert tokenize is False
        parts = []
        for index, message in enumerate(messages):
            content = message.get("content") or ""
            later_env = any(later["role"] in {"user", "tool"} for later in messages[index + 1 :])
            if message["role"] == "assistant" and later_env and "</think>" in content:
                content = content.split("</think>", 1)[1]
            role = message["role"]
            if role == "tool":
                if tools is not None:
                    content = f"<tool_response>\n{content}\n</tool_response>"
                role = "user"
            parts.append(self.format_message(role, content))
        text = "".join(parts)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if self.inject_think:
                text += "<think>\n"
        return text

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return _encode(text, merge_newline_start=False)

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return _decode(token_ids)


class BoundaryMergingTokenizer(QwenLikeTokenizer):
    """Merges the newline after <|im_end|> with the next <|im_start|>."""

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return _encode(text, merge_newline_start=True)


def _encode(text: str, *, merge_newline_start: bool) -> list[int]:
    specials = QwenLikeTokenizer.special_tokens
    token_ids = []
    index = 0
    while index < len(text):
        if merge_newline_start and text.startswith("\n<|im_start|>", index):
            token_ids.append(2000)
            index += len("\n<|im_start|>")
            continue
        matched = False
        for token, token_id in specials:
            if text.startswith(token, index):
                token_ids.append(token_id)
                index += len(token)
                matched = True
                break
        if matched:
            continue
        token_ids.append(ord(text[index]))
        index += 1
    return token_ids


def _decode(token_ids) -> str:
    specials = {token_id: token for token, token_id in QwenLikeTokenizer.special_tokens}
    parts = []
    for token_id in token_ids:
        if token_id == 2000:
            parts.append("\n<|im_start|>")
        elif token_id == 42:
            # Not produced by encode(). A retokenized trajectory cannot preserve it.
            parts.append("ab")
        elif token_id in specials:
            parts.append(specials[token_id])
        else:
            parts.append(chr(token_id))
    return "".join(parts)


def _tail_after_assistant(tokenizer, messages, observation, *, tools=None, add_generation_prompt):
    module = _load_token_delta()
    anchor = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False, tools=tools)
    full = tokenizer.apply_chat_template(
        [*messages, observation],
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        tools=tools,
    )
    anchor_ids = tokenizer.encode(anchor, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    end_id = module.assistant_end_token_id(tokenizer, tools=tools)
    end = max(index for index, token_id in enumerate(anchor_ids) if token_id == end_id)
    assert full_ids[: end + 1] == anchor_ids[: end + 1]
    return full_ids[end + 1 :]


@pytest.mark.unit
def test_observation_suffix_keeps_the_newline_after_the_end_token():
    module = _load_token_delta()
    tokenizer = QwenLikeTokenizer()
    observation = {"role": "user", "content": "next user"}
    history = [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "first answer"},
    ]

    suffix = module.observation_suffix_token_ids(tokenizer, observation, add_generation_prompt=True)

    assert suffix == _tail_after_assistant(tokenizer, history, observation, add_generation_prompt=True)
    anchor = tokenizer.apply_chat_template(history, add_generation_prompt=False, tokenize=False)
    full = tokenizer.apply_chat_template([*history, observation], add_generation_prompt=True, tokenize=False)
    naive = tokenizer.encode(full[len(anchor) :], add_special_tokens=False)
    assert tokenizer.decode(suffix).startswith("\n")
    assert suffix != naive


@pytest.mark.unit
def test_observation_suffix_ignores_rewritten_history():
    module = _load_token_delta()
    tokenizer = QwenLikeTokenizer()
    observation = {"role": "user", "content": "third"}
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "<think>secret</think>answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "plain"},
    ]
    rewritten = tokenizer.apply_chat_template([*history, observation], add_generation_prompt=True, tokenize=False)

    suffix = module.observation_suffix_token_ids(tokenizer, observation, add_generation_prompt=True)

    assert "secret" not in rewritten
    assert "secret" not in tokenizer.decode(suffix)
    assert suffix == _tail_after_assistant(tokenizer, history, observation, add_generation_prompt=True)


@pytest.mark.unit
def test_sampled_assistant_tokens_are_not_retokenized():
    module = _load_token_delta()
    tokenizer = QwenLikeTokenizer()
    end_id = module.assistant_end_token_id(tokenizer)
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    sampled = [42, end_id]
    observation = {"role": "tool", "name": "lookup", "content": "result"}

    suffix = module.tokens_after_assistant(
        tokenizer,
        sampled,
        observation,
        tools=tools,
        add_generation_prompt=True,
    )
    response = sampled + suffix

    assert response[: len(sampled)] == sampled
    assert 42 not in tokenizer.encode(tokenizer.decode([42]), add_special_tokens=False)
    assert "<tool_response>" in tokenizer.decode(suffix)
    assert tokenizer.decode(suffix).endswith("<|im_start|>assistant\n")
    assert suffix[0] != end_id


@pytest.mark.unit
def test_multiturn_context_is_the_exact_token_prefix():
    module = _load_token_delta()
    tokenizer = QwenLikeTokenizer(inject_think=True)
    end_id = module.assistant_end_token_id(tokenizer)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "wiki"},
            {"role": "user", "content": "question"},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    turn1 = [42, end_id]
    suffix1 = module.tokens_after_assistant(
        tokenizer,
        turn1,
        {"role": "tool", "name": "lookup", "content": "result"},
        tools=[{"type": "function"}],
        add_generation_prompt=True,
    )
    turn2 = [7, 8, end_id]
    suffix2 = module.tokens_after_assistant(
        tokenizer,
        turn2,
        {"role": "user", "content": "thanks"},
        add_generation_prompt=False,
    )

    context = prompt_ids + turn1 + suffix1
    response = turn1 + suffix1 + turn2 + suffix2
    loss_mask = [1] * len(turn1) + [0] * len(suffix1) + [1] * len(turn2) + [0] * len(suffix2)
    generation_prompt = tokenizer.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)

    assert context[-len(generation_prompt) :] == generation_prompt
    assert response[len(turn1) + len(suffix1) : len(turn1) + len(suffix1) + len(turn2)] == turn2
    assert loss_mask[: len(turn1)] == [1, 1]
    assert set(loss_mask[len(turn1) : len(turn1) + len(suffix1)]) == {0}
    assert not tokenizer.decode(suffix2).endswith("<|im_start|>assistant\n<think>\n")
    assert "result" in tokenizer.decode(suffix1)


@pytest.mark.unit
def test_missing_end_token_is_inserted_once_before_the_observation():
    module = _load_token_delta()
    tokenizer = QwenLikeTokenizer()
    end_id = module.assistant_end_token_id(tokenizer)

    suffix = module.tokens_after_assistant(
        tokenizer,
        [42],
        {"role": "user", "content": "next"},
        add_generation_prompt=True,
    )

    assert suffix[0] == end_id
    assert suffix[1] != end_id
    assert tokenizer.decode(suffix).startswith(tokenizer.decode([end_id]) + "\n")


@pytest.mark.unit
def test_bpe_merge_across_the_end_token_stays_in_the_suffix():
    module = _load_token_delta()
    tokenizer = BoundaryMergingTokenizer()
    observation = {"role": "user", "content": "next"}
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]

    suffix = module.observation_suffix_token_ids(tokenizer, observation, add_generation_prompt=True)

    assert suffix[0] == 2000
    assert suffix == _tail_after_assistant(tokenizer, history, observation, add_generation_prompt=True)
    anchor = tokenizer.apply_chat_template(history, add_generation_prompt=False, tokenize=False)
    full = tokenizer.apply_chat_template([*history, observation], add_generation_prompt=True, tokenize=False)
    assert tokenizer.encode(full[len(anchor) :], add_special_tokens=False)[0] != 2000


@pytest.mark.unit
def test_non_special_assistant_closer_is_rejected():
    module = _load_token_delta()
    tokenizer = QwenLikeTokenizer(newline_before_end=True)

    with pytest.raises(ValueError, match="special token"):
        module.assistant_end_token_id(tokenizer)


@pytest.mark.unit
def test_text_only_generation_is_rejected():
    module = _load_token_delta()

    with pytest.raises(ValueError, match="output_token_logprobs"):
        module.sampled_output_tokens({"text": "hello<|im_end|>", "meta_info": {"finish_reason": {"type": "stop"}}})


@pytest.mark.unit
def test_sampled_tokens_must_match_output_ids():
    module = _load_token_delta()

    token_ids, log_probs = module.sampled_output_tokens(
        {
            "output_ids": [5, 6],
            "meta_info": {"output_token_logprobs": [[-0.2, 5, "a"], [-0.4, 6, "b"]]},
        }
    )
    assert token_ids == [5, 6]
    assert log_probs == [-0.2, -0.4]

    with pytest.raises(ValueError, match="output_ids"):
        module.sampled_output_tokens(
            {"output_ids": [1], "meta_info": {"output_token_logprobs": [[-0.1, 2]]}}
        )
    with pytest.raises(ValueError, match="non-finite"):
        module.sampled_output_tokens({"meta_info": {"output_token_logprobs": [[None, 1]]}})


@pytest.mark.unit
def test_assistant_text_strips_only_a_sampled_end_token():
    module = _load_token_delta()
    tokenizer = QwenLikeTokenizer()
    end_id = module.assistant_end_token_id(tokenizer)
    call = tokenizer.encode("<tool_call>lookup</tool_call>", add_special_tokens=False)

    assert module.assistant_message_text(tokenizer, [*call, end_id]) == "<tool_call>lookup</tool_call>"
    assert module.assistant_message_text(tokenizer, call) == "<tool_call>lookup</tool_call>"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
