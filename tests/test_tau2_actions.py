import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAU2_DIR = ROOT / "examples" / "tau-bench" / "tau2"
sys.path.insert(0, str(TAU2_DIR))

from actions import parse_action


def test_parse_action_rejects_multiple_tool_calls():
    text = (
        '<tool_call>{"name": "respond", "arguments": {"content": "hi"}}</tool_call>'
        "\n"
        '<tool_call>{"name": "done", "arguments": {}}</tool_call>'
    )
    try:
        parse_action(text)
    except ValueError as exc:
        assert "Multiple <tool_call>" in str(exc)
    else:
        raise AssertionError("Expected ValueError for multiple tool calls")
