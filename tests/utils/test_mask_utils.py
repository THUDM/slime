from transformers import AutoTokenizer

from slime.utils.mask_utils import (
    MultiTurnLossMaskGenerator,
    compress_loss_mask,
    decompress_loss_mask,
)


def test_compress_decompress_all_ones():
    """Test compression/decompression with all 1s (common default case)."""
    mask = [1] * 1000
    compressed = compress_loss_mask(mask)
    assert compressed == ([1000], 1), f"Expected ([1000], 1), got {compressed}"
    decompressed = decompress_loss_mask(compressed)
    assert decompressed == mask, "Decompressed mask doesn't match original"


def test_compress_decompress_all_zeros():
    """Test compression/decompression with all 0s (remove_sample case)."""
    mask = [0] * 500
    compressed = compress_loss_mask(mask)
    assert compressed == ([500], 0), f"Expected ([500], 0), got {compressed}"
    decompressed = decompress_loss_mask(compressed)
    assert decompressed == mask, "Decompressed mask doesn't match original"


def test_compress_decompress_prefix_zeros():
    """Test compression/decompression with prefix zeros (common multi-turn pattern)."""
    mask = [0] * 100 + [1] * 200
    compressed = compress_loss_mask(mask)
    assert compressed == ([100, 200], 0), f"Expected ([100, 200], 0), got {compressed}"
    decompressed = decompress_loss_mask(compressed)
    assert decompressed == mask, "Decompressed mask doesn't match original"


def test_compress_decompress_alternating():
    """Test compression/decompression with alternating pattern."""
    mask = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    compressed = compress_loss_mask(mask)
    assert compressed == ([3, 5, 2, 3], 0), f"Expected ([3, 5, 2, 3], 0), got {compressed}"
    decompressed = decompress_loss_mask(compressed)
    assert decompressed == mask, "Decompressed mask doesn't match original"


def test_compress_decompress_empty():
    """Test compression/decompression with empty mask."""
    mask = []
    compressed = compress_loss_mask(mask)
    assert compressed == ([], 0), f"Expected ([], 0), got {compressed}"
    decompressed = decompress_loss_mask(compressed)
    assert decompressed == mask, "Decompressed mask doesn't match original"


def test_compress_decompress_single_element():
    """Test compression/decompression with single element masks."""
    mask_one = [1]
    compressed = compress_loss_mask(mask_one)
    assert compressed == ([1], 1), f"Expected ([1], 1), got {compressed}"
    assert decompress_loss_mask(compressed) == mask_one

    mask_zero = [0]
    compressed = compress_loss_mask(mask_zero)
    assert compressed == ([1], 0), f"Expected ([1], 0), got {compressed}"
    assert decompress_loss_mask(compressed) == mask_zero


def test_compression_efficiency():
    """Test that compression actually reduces size for typical patterns."""
    import sys

    # All 1s case (8192 tokens)
    mask = [1] * 8192
    compressed = compress_loss_mask(mask)
    # Compressed: ([8192], 1) - just 2 elements
    assert len(compressed[0]) == 1, f"Expected 1 run, got {len(compressed[0])}"

    # Prefix zeros case (common in multi-turn)
    mask = [0] * 2000 + [1] * 6000
    compressed = compress_loss_mask(mask)
    # Compressed: ([2000, 6000], 0) - just 2 elements
    assert len(compressed[0]) == 2, f"Expected 2 runs, got {len(compressed[0])}"


def test_loss_mask_qwen3_simple(model_name: str = "Qwen/Qwen3-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        {"role": "assistant", "content": "ASSISTANT RESPONSE FOR TESTING ONLY"},
    ]
    all_token_ids, all_loss_masks = mask_generator.gen_multi_turn_loss_mask_qwen3(messages)
    assert len(all_token_ids) == len(all_loss_masks), f"{len(all_token_ids)} != {len(all_loss_masks)}"
    selected_texts = mask_generator.get_text_from_loss_mask(all_token_ids, all_loss_masks)
    assert len(selected_texts) == 1, f"Expected 1 text, got {len(selected_texts)}"

    print(f"==== Single Turn Test {model_name} ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


def test_loss_mask_qwen3_tools(model_name: str = "Qwen/Qwen3-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        {
            "role": "assistant",
            "content": "I WILL CALL terminal",
            "tool_calls": [
                {"function": {"name": "terminal", "arguments": {"command": "ls"}}, "id": "call_0", "type": "function"},
                {"function": {"name": "terminal", "arguments": {"command": "ls"}}, "id": "call_0", "type": "function"},
            ],
        },
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

    all_token_ids, all_loss_masks = mask_generator.gen_multi_turn_loss_mask_qwen3(messages, tools)
    assert len(all_token_ids) == len(all_loss_masks), f"{len(all_token_ids)} != {len(all_loss_masks)}"
    selected_texts = mask_generator.get_text_from_loss_mask(all_token_ids, all_loss_masks)
    assert len(selected_texts) == 2, f"Expected 2 texts, got {len(selected_texts)}"

    print(f"==== Multi-turn with Tools Test {model_name} ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


if __name__ == "__main__":
    test_loss_mask_qwen3_simple("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    test_loss_mask_qwen3_tools("Qwen/Qwen3-Coder-30B-A3B-Instruct")
