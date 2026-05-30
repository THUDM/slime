import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from transformers import AutoProcessor, AutoTokenizer

from slime.utils.mask_utils import MultiTurnLossMaskGenerator

NUM_GPUS = 0


@dataclass(frozen=True)
class ChatTemplateSpec:
    name: str
    repo_id: str
    local_names: tuple[str, ...]
    tool_arguments: object
    tool_context_markers: tuple[str, ...]
    assistant_tool_content: str | None
    assistant_tool_call_snippets: tuple[str, ...]
    tool_context_snippets: tuple[str, ...]


REAL_CHAT_TEMPLATE_SPECS = (
    ChatTemplateSpec(
        name="qwen3",
        repo_id="Qwen/Qwen3-4B",
        local_names=("Qwen3-4B", "Qwen3-0.6B", "Qwen3-30B-A3B"),
        tool_arguments='{"query":"Lisbon evening weather"}',
        tool_context_markers=("<tool_response>",),
        assistant_tool_content="",
        assistant_tool_call_snippets=('<tool_call>\n{"name": "slime_lookup"',),
        tool_context_snippets=("<tool_response>\nraw_weather_payload",),
    ),
    ChatTemplateSpec(
        name="qwen3_5",
        repo_id="Qwen/Qwen3.5-0.8B",
        local_names=("Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.5-35B-A3B"),
        tool_arguments={"query": "Lisbon evening weather"},
        tool_context_markers=("<tool_response>",),
        assistant_tool_content="",
        assistant_tool_call_snippets=("<tool_call>\n<function=slime_lookup>",),
        tool_context_snippets=("<tool_response>\nraw_weather_payload",),
    ),
    ChatTemplateSpec(
        name="glm4_7",
        repo_id="zai-org/GLM-4.7",
        local_names=("GLM-4.7", "GLM-4.7-355B-A32B"),
        tool_arguments={"query": "Lisbon evening weather"},
        tool_context_markers=("<|observation|>", "<tool_response>"),
        assistant_tool_content="",
        assistant_tool_call_snippets=("<tool_call>slime_lookup<arg_key>query",),
        tool_context_snippets=("<|observation|><tool_response>raw_weather_payload",),
    ),
    ChatTemplateSpec(
        name="glm4_7_flash",
        repo_id="zai-org/GLM-4.7-Flash",
        local_names=("GLM-4.7-Flash",),
        tool_arguments={"query": "Lisbon evening weather"},
        tool_context_markers=("<|observation|>", "<tool_response>"),
        assistant_tool_content="",
        assistant_tool_call_snippets=("<tool_call>slime_lookup<arg_key>query",),
        tool_context_snippets=("<|observation|><tool_response>raw_weather_payload",),
    ),
    ChatTemplateSpec(
        name="glm5_1",
        repo_id="zai-org/GLM-5.1",
        local_names=("GLM-5.1",),
        tool_arguments={"query": "Lisbon evening weather"},
        tool_context_markers=("<|observation|>", "<tool_response>"),
        assistant_tool_content="",
        assistant_tool_call_snippets=("<tool_call>slime_lookup<arg_key>query",),
        tool_context_snippets=("<|observation|><tool_response>raw_weather_payload",),
    ),
    ChatTemplateSpec(
        name="deepseek_v3",
        repo_id="deepseek-ai/DeepSeek-V3",
        local_names=("DeepSeek-V3", "DeepSeek-V3-0324", "DeepSeek-R1"),
        tool_arguments='{"query":"Lisbon evening weather"}',
        tool_context_markers=("<｜tool▁outputs▁begin｜>", "<｜tool▁output▁begin｜>", "<｜tool▁outputs▁end｜>"),
        assistant_tool_content=None,
        assistant_tool_call_snippets=(
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>slime_lookup",
        ),
        tool_context_snippets=("<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>raw_weather_payload",),
    ),
)


def _candidate_model_refs(spec: ChatTemplateSpec) -> list[str]:
    refs = [spec.repo_id]
    roots = [os.environ.get("SLIME_TEST_MODEL_ROOT")]
    if not os.environ.get("CI"):
        roots.extend(["/root/models", "/mnt/nvme0n1/slime_ci/models"])
    for root in roots:
        if not root:
            continue
        for local_name in spec.local_names:
            path = Path(root) / local_name
            if _path_exists(path):
                refs.append(str(path))
    return refs


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _has_chat_template(obj) -> bool:
    if obj is None:
        return False
    if getattr(obj, "chat_template", None):
        return True
    get_chat_template = getattr(obj, "get_chat_template", None)
    if get_chat_template is None:
        return False
    try:
        return bool(get_chat_template(chat_template=None, tools=None))
    except Exception:
        return False


def _load_real_chat_template(spec: ChatTemplateSpec):
    errors = []
    loaded_without_template = False
    for model_ref in _candidate_model_refs(spec):
        is_local = _path_exists(Path(model_ref))
        tokenizer = None
        processor = None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                trust_remote_code=True,
                local_files_only=is_local,
            )
        except Exception as exc:
            errors.append(f"{model_ref} tokenizer: {type(exc).__name__}: {exc}")

        if tokenizer is not None and _has_chat_template(tokenizer):
            return tokenizer, None, model_ref

        try:
            processor = AutoProcessor.from_pretrained(
                model_ref,
                trust_remote_code=True,
                local_files_only=is_local,
            )
        except Exception as exc:
            errors.append(f"{model_ref} processor: {type(exc).__name__}: {exc}")

        if tokenizer is None and getattr(processor, "tokenizer", None) is not None:
            tokenizer = processor.tokenizer

        if tokenizer is None:
            continue

        chat_template_processor = processor if processor is not None and _has_chat_template(processor) else None
        if _has_chat_template(tokenizer) or chat_template_processor is not None:
            return tokenizer, chat_template_processor, model_ref

        loaded_without_template = True
        errors.append(f"{model_ref}: no tokenizer/processor chat template")

    message = f"Could not load a real chat template for {spec.name}.\n" + "\n".join(errors[-8:])
    if os.environ.get("CI") and not loaded_without_template:
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(params=REAL_CHAT_TEMPLATE_SPECS, ids=lambda spec: spec.name, scope="session")
def real_chat_template(request):
    spec = request.param
    tokenizer, processor, model_ref = _load_real_chat_template(spec)
    generator = MultiTurnLossMaskGenerator(tokenizer, chat_template_processor=processor)
    return spec, tokenizer, generator, model_ref


def _render(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _selected(generator: MultiTurnLossMaskGenerator, token_ids: list[int], loss_mask: list[int]) -> str:
    return "".join(generator.get_text_from_loss_mask(token_ids, loss_mask))


BOUNDARY_MARKERS_NOT_SUPERVISED = (
    "[gMASK]",
    "<sop>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|observation|>",
    "<｜begin▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜end▁of▁sentence｜>",
    "<｜tool▁outputs▁begin｜>",
    "<｜tool▁outputs▁end｜>",
    "<｜tool▁output▁begin｜>",
    "<｜tool▁output▁end｜>",
    "<tool_response>",
    "</tool_response>",
)


def _to_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _mask_spans(mask: list[int]) -> list[tuple[int, int]]:
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


def _common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _common_suffix_len(left: str, right: str, prefix_len: int) -> int:
    limit = min(len(left), len(right)) - prefix_len
    index = 0
    while index < limit and left[len(left) - index - 1] == right[len(right) - index - 1]:
        index += 1
    return index


def _render_messages(generator: MultiTurnLossMaskGenerator, messages: list[dict], tools: list[dict] = None) -> str:
    rendered = generator._apply_chat_template(messages, tokenize=False, tools=tools)
    assert isinstance(rendered, str)
    return rendered


def _assistant_char_spans_from_template_diffs(
    generator: MultiTurnLossMaskGenerator,
    rendered: str,
    messages: list[dict],
    tools: list[dict] = None,
) -> list[tuple[int, int]]:
    spans = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        sentinel = f"SLIME_ASSISTANT_SPAN_{message_index}_BOUNDARY_SENTINEL"
        assert sentinel not in rendered
        replacement = {
            key: value for key, value in message.items() if key not in ("content", "reasoning_content", "tool_calls")
        }
        replacement["content"] = sentinel
        probe_messages = list(messages)
        probe_messages[message_index] = replacement
        probe_rendered = _render_messages(generator, probe_messages, tools=tools)
        assert sentinel in probe_rendered

        span_start = _common_prefix_len(rendered, probe_rendered)
        suffix_len = _common_suffix_len(rendered, probe_rendered, span_start)
        span_end = len(rendered) - suffix_len
        assert span_start < span_end
        spans.append((span_start, span_end))

    assert spans
    return spans


def _expected_loss_mask_from_template_boundaries(
    tokenizer,
    generator: MultiTurnLossMaskGenerator,
    messages: list[dict],
    tools: list[dict] = None,
) -> tuple[str, list[int], list[int], list[tuple[int, int]]]:
    rendered = _render_messages(generator, messages, tools=tools)
    tokenized = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = _to_list(tokenized["input_ids"])
    offsets = [(int(start), int(end)) for start, end in _to_list(tokenized["offset_mapping"])]
    char_spans = _assistant_char_spans_from_template_diffs(generator, rendered, messages, tools=tools)

    loss_mask = [0] * len(token_ids)
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if any(token_start < span_end and token_end > span_start for span_start, span_end in char_spans):
            loss_mask[token_index] = 1

    if any(message.get("step_loss_mask", 1) != 1 for message in messages):
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        spans = _mask_spans(loss_mask)
        assert len(spans) == len(assistant_messages)
        for (span_start, span_end), message in zip(spans, assistant_messages, strict=True):
            if message.get("step_loss_mask", 1) != 1:
                loss_mask[span_start:span_end] = [0] * (span_end - span_start)

    return rendered, token_ids, loss_mask, offsets


def _assert_boundary_markers_unmasked(
    tokenizer,
    rendered: str,
    token_ids: list[int],
    loss_mask: list[int],
    offsets: list[tuple[int, int]],
) -> None:
    for marker in BOUNDARY_MARKERS_NOT_SUPERVISED:
        marker_start = rendered.find(marker)
        while marker_start >= 0:
            marker_end = marker_start + len(marker)
            token_indexes = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_start < marker_end and token_end > marker_start
            ]
            assert token_indexes, f"marker {marker!r} was rendered but mapped to no tokens"
            supervised = [index for index in token_indexes if loss_mask[index]]
            assert not supervised, (
                f"boundary marker {marker!r} at chars {marker_start}:{marker_end} is supervised: "
                f"{tokenizer.decode([token_ids[index] for index in supervised], skip_special_tokens=False)!r}"
            )
            marker_start = rendered.find(marker, marker_end)


def _assert_mask_matches_template_boundaries(
    tokenizer,
    generator: MultiTurnLossMaskGenerator,
    messages: list[dict],
    result,
    tools: list[dict] = None,
) -> None:
    rendered, expected_token_ids, expected_loss_mask, offsets = _expected_loss_mask_from_template_boundaries(
        tokenizer,
        generator,
        messages,
        tools=tools,
    )

    assert result.token_ids == expected_token_ids
    assert _mask_spans(result.loss_mask) == _mask_spans(expected_loss_mask)
    assert result.loss_mask == expected_loss_mask
    _assert_boundary_markers_unmasked(tokenizer, rendered, result.token_ids, result.loss_mask, offsets)


def _assert_rendered_substring_mask_value(
    tokenizer,
    token_ids: list[int],
    loss_mask: list[int],
    rendered: str,
    text: str,
    expected: int,
) -> None:
    start = rendered.find(text)
    assert start >= 0, f"{text!r} was not rendered"
    end = start + len(text)

    tokenized = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    assert _to_list(tokenized["input_ids"]) == token_ids
    offsets = [(int(token_start), int(token_end)) for token_start, token_end in _to_list(tokenized["offset_mapping"])]
    token_indexes = [
        index for index, (token_start, token_end) in enumerate(offsets) if token_start < end and token_end > start
    ]
    assert token_indexes, f"{text!r} was rendered but mapped to no tokens"

    unexpected = [index for index in token_indexes if loss_mask[index] != expected]
    assert not unexpected, (
        f"{text!r} expected mask={expected}, but got "
        f"{[(index, loss_mask[index], tokenizer.decode([token_ids[index]], skip_special_tokens=False)) for index in unexpected]}"
    )


def test_sft_loss_mask_uses_real_model_assistant_content_spans(real_chat_template):
    _spec, tokenizer, generator, _model_ref = real_chat_template
    messages = [
        {
            "role": "system",
            "content": (
                "Use tools only when needed. Available schema: "
                '<function name="calculator">integer arithmetic</function>.'
            ),
        },
        {
            "role": "user",
            "content": ("Solve 19 + 23. Context quote: user supplied observation should stay context only."),
        },
        {
            "role": "assistant",
            "content": "<think>Compute the sum before answering.</think>\nThe answer is 42.",
        },
        {
            "role": "user",
            "content": "Now give the final response. <tool>quoted tool text from the user</tool>",
        },
        {
            "role": "assistant",
            "content": "<think>Keep the final answer concise.</think>\nFinal answer: 42.",
        },
    ]

    result = generator.get_loss_mask_result(messages)
    _assert_mask_matches_template_boundaries(tokenizer, generator, messages, result)
    rendered = _render(tokenizer, result.token_ids)
    selected = _selected(generator, result.token_ids, result.loss_mask)

    assert len(result.token_ids) == len(result.loss_mask)
    assert len(result.response_loss_mask) == result.response_length
    assert "The answer is 42" in selected
    assert "Keep the final answer concise" in selected
    assert "Final answer: 42" in selected
    assert "integer arithmetic" in rendered
    assert "user supplied observation should stay context only" in rendered
    assert "quoted tool text from the user" in rendered
    assert "integer arithmetic" not in selected
    assert "user supplied observation should stay context only" not in selected
    assert "quoted tool text from the user" not in selected


def test_sft_loss_mask_handles_real_model_tool_and_observation_spans(real_chat_template):
    spec, tokenizer, generator, _model_ref = real_chat_template
    messages = [
        {
            "role": "user",
            "content": (
                "Check whether I need an umbrella tonight. Tool schema reminder: "
                '<function name="slime_lookup">weather lookup</function>.'
            ),
        },
        {
            "role": "assistant",
            "content": spec.assistant_tool_content,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "slime_lookup",
                        "arguments": spec.tool_arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "raw_weather_payload: precip_code=RAIN_1900; wind_code=NE",
        },
        {
            "role": "assistant",
            "content": (
                "<think>Use the environment payload, but do not copy it verbatim.</think>\n"
                "Bring an umbrella after 7pm."
            ),
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "slime_lookup",
                "description": "lookup",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]

    result = generator.get_loss_mask_result(messages, tools=tools)
    _assert_mask_matches_template_boundaries(tokenizer, generator, messages, result, tools=tools)
    rendered = _render(tokenizer, result.token_ids)
    selected = _selected(generator, result.token_ids, result.loss_mask)

    for marker in spec.tool_context_markers:
        assert marker in rendered
    for snippet in spec.assistant_tool_call_snippets:
        _assert_rendered_substring_mask_value(
            tokenizer,
            result.token_ids,
            result.loss_mask,
            rendered,
            snippet,
            1,
        )
    for snippet in spec.tool_context_snippets:
        _assert_rendered_substring_mask_value(
            tokenizer,
            result.token_ids,
            result.loss_mask,
            rendered,
            snippet,
            0,
        )
    _assert_rendered_substring_mask_value(
        tokenizer,
        result.token_ids,
        result.loss_mask,
        rendered,
        "raw_weather_payload: precip_code=RAIN_1900; wind_code=NE",
        0,
    )
    assert "slime_lookup" in selected
    assert "Lisbon evening weather" in selected
    assert "Bring an umbrella after 7pm" in selected
    assert "raw_weather_payload" in rendered
    assert "precip_code=RAIN_1900" in rendered
    assert "wind_code=NE" in rendered
    assert "raw_weather_payload" not in selected
    assert "precip_code=RAIN_1900" not in selected
    assert "wind_code=NE" not in selected


def test_sft_loss_mask_preserves_response_start_when_first_turn_is_step_masked(real_chat_template):
    _spec, tokenizer, generator, _model_ref = real_chat_template
    messages = [
        {
            "role": "user",
            "content": "Draft an answer from the old cached result.",
        },
        {
            "role": "assistant",
            "content": (
                "<think>This draft should be kept in context only.</think>\n"
                "Outdated draft answer.\n"
                "<func_call>cached_lookup()</func_call>"
            ),
            "step_loss_mask": 0,
        },
        {
            "role": "user",
            "content": "Revise after fresh context: <tool>confirmed current result</tool>.",
        },
        {
            "role": "assistant",
            "content": "<think>Use the fresh context.</think>\nUpdated final answer.",
        },
    ]

    result = generator.get_loss_mask_result(messages)
    _assert_mask_matches_template_boundaries(tokenizer, generator, messages, result)
    rendered_tail = _render(tokenizer, result.token_ids[result.response_start :])
    selected = _selected(generator, result.token_ids, result.loss_mask)

    assert "Outdated draft answer" in rendered_tail
    assert "Outdated draft answer" not in selected
    assert "cached_lookup" in rendered_tail
    assert "cached_lookup" not in selected
    assert "This draft should be kept in context only" not in selected
    assert "Updated final answer" in selected
    assert "Use the fresh context" in selected
    assert len(result.response_loss_mask) == result.response_length


def test_sft_loss_mask_trains_thinking_tags_embedded_in_assistant_content(real_chat_template):
    _spec, tokenizer, generator, _model_ref = real_chat_template
    messages = [
        {
            "role": "user",
            "content": (
                "Explain why the parser should ignore user tags: "
                "<func_call>not actually a call</func_call>. "
                "external observation remains context"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "<think>The tags are plain assistant text in this sample.</think>\n"
                "<func_call>none</func_call>\n"
                "They should be supervised because they are in assistant content."
            ),
        },
    ]

    result = generator.get_loss_mask_result(messages)
    _assert_mask_matches_template_boundaries(tokenizer, generator, messages, result)
    rendered = _render(tokenizer, result.token_ids)
    selected = _selected(generator, result.token_ids, result.loss_mask)

    assert "The tags are plain assistant text in this sample" in selected
    assert "none" in selected
    assert "They should be supervised because they are in assistant content" in selected
    assert "not actually a call" in rendered
    assert "external observation remains context" in rendered
    assert "not actually a call" not in selected
    assert "external observation remains context" not in selected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
