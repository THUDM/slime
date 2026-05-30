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


REAL_CHAT_TEMPLATE_SPECS = (
    ChatTemplateSpec(
        name="qwen3",
        repo_id="Qwen/Qwen3-4B",
        local_names=("Qwen3-4B", "Qwen3-0.6B", "Qwen3-30B-A3B"),
        tool_arguments='{"query":"Lisbon evening weather"}',
        tool_context_markers=("<tool_response>",),
    ),
    ChatTemplateSpec(
        name="qwen3_5",
        repo_id="Qwen/Qwen3.5-0.8B",
        local_names=("Qwen3.5-0.8B", "Qwen3.5-4B", "Qwen3.5-35B-A3B"),
        tool_arguments={"query": "Lisbon evening weather"},
        tool_context_markers=("<tool_response>",),
    ),
    ChatTemplateSpec(
        name="glm4_7",
        repo_id="zai-org/GLM-4.7",
        local_names=("GLM-4.7", "GLM-4.7-355B-A32B"),
        tool_arguments={"query": "Lisbon evening weather"},
        tool_context_markers=("<|observation|>", "<tool_response>"),
    ),
    ChatTemplateSpec(
        name="glm4_7_flash",
        repo_id="zai-org/GLM-4.7-Flash",
        local_names=("GLM-4.7-Flash",),
        tool_arguments={"query": "Lisbon evening weather"},
        tool_context_markers=("<|observation|>", "<tool_response>"),
    ),
    ChatTemplateSpec(
        name="glm5_1",
        repo_id="zai-org/GLM-5.1",
        local_names=("GLM-5.1",),
        tool_arguments={"query": "Lisbon evening weather"},
        tool_context_markers=("<|observation|>", "<tool_response>"),
    ),
    ChatTemplateSpec(
        name="deepseek_v3",
        repo_id="deepseek-ai/DeepSeek-V3",
        local_names=("DeepSeek-V3", "DeepSeek-V3-0324", "DeepSeek-R1"),
        tool_arguments='{"query":"Lisbon evening weather"}',
        tool_context_markers=("<｜tool▁outputs▁begin｜>", "<｜tool▁output▁begin｜>", "<｜tool▁outputs▁end｜>"),
    ),
)


def _candidate_model_refs(spec: ChatTemplateSpec) -> list[str]:
    refs = [spec.repo_id]
    roots = [
        os.environ.get("SLIME_TEST_MODEL_ROOT"),
        "/root/models",
        "/mnt/nvme0n1/slime_ci/models",
    ]
    for root in roots:
        if not root:
            continue
        for local_name in spec.local_names:
            path = Path(root) / local_name
            if path.exists():
                refs.append(str(path))
    return refs


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
        is_local = Path(model_ref).exists()
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
            "content": (
                "Solve 19 + 23. Context quote: "
                "<observation>user supplied observation should stay context only</observation>."
            ),
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
            "content": (
                "<think>Need current weather before answering.</think>\n"
                "<observation>assistant plans to call the weather lookup</observation>\n"
                "<func_call>slime_lookup(query='Lisbon evening weather')</func_call>"
            ),
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
            "content": "<observation>weather station says rain after 7pm</observation>\n<tool>wind northeast</tool>",
        },
        {
            "role": "assistant",
            "content": (
                "<think>Use the observation, but do not copy the raw tool payload.</think>\n"
                "<observation>assistant concludes rain is likely tonight</observation>\n"
                "Bring an umbrella."
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
    rendered = _render(tokenizer, result.token_ids)
    selected = _selected(generator, result.token_ids, result.loss_mask)

    for marker in spec.tool_context_markers:
        assert marker in rendered
    assert "slime_lookup" in selected
    assert "Lisbon evening weather" in selected
    assert "assistant plans to call the weather lookup" in selected
    assert "assistant concludes rain is likely tonight" in selected
    assert "Bring an umbrella" in selected
    assert "weather station says rain after 7pm" in rendered
    assert "wind northeast" in rendered
    assert "weather station says rain after 7pm" not in selected
    assert "wind northeast" not in selected


def test_sft_loss_mask_preserves_response_start_when_first_turn_is_step_masked(real_chat_template):
    _spec, tokenizer, generator, _model_ref = real_chat_template
    messages = [
        {
            "role": "user",
            "content": "Draft an answer from <observation>old cached result</observation>.",
        },
        {
            "role": "assistant",
            "content": "<think>This draft should be kept in context only.</think>\nOutdated draft answer.",
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
    rendered_tail = _render(tokenizer, result.token_ids[result.response_start :])
    selected = _selected(generator, result.token_ids, result.loss_mask)

    assert "Outdated draft answer" in rendered_tail
    assert "Outdated draft answer" not in selected
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
                "<observation>external observation remains context</observation>"
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
