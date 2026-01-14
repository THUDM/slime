import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from transformers import AutoTokenizer

from slime.utils.mask_utils import MultiTurnLossMaskGenerator


def _assert_loss_mask_alignment(
    mask_generator: MultiTurnLossMaskGenerator,
    all_token_ids: list[int],
    all_loss_masks: list[int],
    expected_text_count: int,
) -> list[str]:
    assert len(all_token_ids) == len(all_loss_masks), f"{len(all_token_ids)} != {len(all_loss_masks)}"
    selected_texts = mask_generator.get_text_from_loss_mask(all_token_ids, all_loss_masks)
    assert len(selected_texts) == expected_text_count, f"Expected {expected_text_count} texts, got {len(selected_texts)}"
    return selected_texts


def _simple_messages() -> list[dict]:
    return [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        {"role": "assistant", "content": "ASSISTANT RESPONSE FOR TESTING ONLY"},
    ]


def _tool_messages(tokenizer_type: str) -> tuple[list[dict], list[dict]]:
    tool_args = {"command": "ls"}
    if tokenizer_type == "distill_qwen":
        tool_args = json.dumps(tool_args)

    assistant_first = {
        "role": "assistant",
        "content": "I WILL CALL terminal",
        "tool_calls": [
            {"function": {"name": "terminal", "arguments": tool_args}, "id": "call_0", "type": "function"},
            {"function": {"name": "terminal", "arguments": tool_args}, "id": "call_0", "type": "function"},
        ],
    }
    if tokenizer_type == "distill_qwen":
        assistant_first["content"] = None

    messages = [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        assistant_first,
        {"role": "tool", "name": "terminal", "content": "LICENSE  README.md  README_zh.md"},
        {"role": "tool", "name": "terminal", "content": "LICENSE  README.md  README_zh.md"},
        {"role": "assistant", "content": "ASSISTANT RESPONSE FOR TESTING ONLY"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Perform operations from the terminal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute as `bash -c <command>`",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of the command for the user.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the content of a file given its path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to be read.",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
    ]
    return messages, tools


def _run_simple_case(model_name: str, tokenizer_type: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=tokenizer_type)
    messages = _simple_messages()
    all_token_ids, all_loss_masks = mask_generator.get_loss_mask(messages)
    selected_texts = _assert_loss_mask_alignment(mask_generator, all_token_ids, all_loss_masks, 1)

    print(f"==== Single Turn Test {model_name} ({tokenizer_type}) ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


def _run_tool_case(model_name: str, tokenizer_type: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=tokenizer_type)
    messages, tools = _tool_messages(tokenizer_type)
    all_token_ids, all_loss_masks = mask_generator.get_loss_mask(messages, tools=tools)
    selected_texts = _assert_loss_mask_alignment(mask_generator, all_token_ids, all_loss_masks, 2)

    print(f"==== Multi-turn with Tools Test {model_name} ({tokenizer_type}) ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


def test_loss_mask_simple_all() -> None:
    cases = [
        ("Qwen/Qwen3-0.6B", "qwen3"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "qwen25"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "distill_qwen"),
    ]
    for model_name, tokenizer_type in cases:
        _run_simple_case(model_name, tokenizer_type)


def test_loss_mask_tools_all() -> None:
    cases = [
        ("Qwen/Qwen3-0.6B", "qwen3"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "qwen25"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "distill_qwen"),
    ]
    for model_name, tokenizer_type in cases:
        _run_tool_case(model_name, tokenizer_type)


if __name__ == "__main__":
    test_loss_mask_simple_all()
    test_loss_mask_tools_all()
