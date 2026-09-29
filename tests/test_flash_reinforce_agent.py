"""CPU tests for examples/flash_reinforce_agent: the calculator and the multi-turn generate loop.

The loop runs against a scripted SGLang server: tool results must stay out of the loss, and a
turn cut by a weight update must continue in place instead of restarting the trajectory.
"""

import asyncio
import copy
import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

try:
    import sglang_router  # noqa: F401
except ImportError:
    _router_stub = types.ModuleType("sglang_router")
    _router_stub.__version__ = "0.2.3"
    sys.modules["sglang_router"] = _router_stub
try:
    import transformers  # noqa: F401
except ImportError:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

import pytest

import slime.rollout.sglang_rollout as sr
from slime.utils.types import Sample

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "flash_reinforce_agent"))
import calculator_agent  # noqa: E402

NUM_GPUS = 0


class _Tokenizer:
    """One token per character, and a minimal chat template."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


class _State:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.aborted = False


def _reply(text, tokens, version, finish):
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": [[-0.5, token] for token in tokens],
            "weight_version": version,
            "finish_reason": {"type": finish},
        },
    }


@pytest.fixture
def server(monkeypatch):
    state = _State()
    server = SimpleNamespace(replies=deque(), payloads=[], state=state)

    async def post(url, payload, **kwargs):
        server.payloads.append(copy.deepcopy(payload))
        return server.replies.popleft()

    monkeypatch.setattr(sr, "post", post)
    monkeypatch.setattr(sr, "GenerateState", lambda args: state)
    monkeypatch.setattr(sr, "_TURN_RESUME_DELAY", 0)
    monkeypatch.setattr(calculator_agent, "GenerateState", lambda args: state)
    return server


def _args():
    return SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=0,
        use_rollout_routing_replay=False,
        router_policy=None,
        rollout_top_p=1.0,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "expression,expected",
    [
        ("(12 + 7) * 3", "57"),
        ("7 / 2", "3.5"),
        ("1,234 + 1", "1235"),
        ("10 / 4 * 2", "5"),
        ("-3 ** 2", "-9"),
    ],
)
def test_calculator_evaluates_arithmetic(expression, expected):
    assert calculator_agent.calculate(expression) == expected


@pytest.mark.unit
@pytest.mark.parametrize("expression", ["__import__('os')", "2 ** 100", "1 / 0", "x + 1", "1 +"])
def test_calculator_reports_what_it_cannot_evaluate(expression):
    assert calculator_agent.calculate(expression).startswith("error:")


@pytest.mark.unit
def test_tool_results_stay_out_of_the_loss(server):
    # The result closes the calling turn, arrives as a user message and opens the next assistant turn.
    result = "</assistant><user><result>5</result></user><assistant>"
    server.replies.extend(
        [
            _reply("so <calc>2+3</calc>", [1, 2], "1", "stop"),
            # A weight update cuts the second turn; it continues where it stopped.
            _reply("\\boxed", [3], "1", "abort"),
            _reply("{5}", [4], "2", "stop"),
        ]
    )
    sample = Sample(index=0, prompt="Q")

    asyncio.run(calculator_agent.generate(_args(), sample, {"max_new_tokens": 64}))

    tool_tokens = [ord(char) for char in result]
    prompt = [ord("Q")]
    assert [payload["input_ids"] for payload in server.payloads] == [
        prompt,
        prompt + [1, 2] + tool_tokens,
        prompt + [1, 2] + tool_tokens + [3],
    ]
    assert server.payloads[0]["sampling_params"]["stop"] == ["</calc>"]
    assert sample.loss_mask == [1, 1] + [0] * len(tool_tokens) + [1, 1]
    assert sample.response == "so <calc>2+3</calc>" + result + "\\boxed{5}"
    assert sample.weight_versions == ["1", "1", "2"]
    assert sample.status == Sample.Status.COMPLETED


@pytest.mark.unit
def test_the_trajectory_budget_spans_turns_and_tool_results(server):
    server.replies.append(_reply("<calc>1+1</calc>", [1, 2], "1", "stop"))
    sample = Sample(index=0, prompt="Q")

    asyncio.run(calculator_agent.generate(_args(), sample, {"max_new_tokens": 10}))

    # The result message does not fit in what is left of the 10-token budget, so the trajectory
    # ends on the model's turn rather than on tool tokens that no later turn would route.
    assert len(server.payloads) == 1 and sample.status == Sample.Status.TRUNCATED
    assert sample.tokens == [ord("Q"), 1, 2] and sample.loss_mask == [1, 1]


@pytest.mark.unit
def test_a_trajectory_out_of_calls_ends_on_the_model_turn(monkeypatch, server):
    monkeypatch.setattr(calculator_agent, "MAX_TOOL_CALLS", 1)
    server.replies.extend(
        [_reply("<calc>1+1</calc>", [1, 2], "1", "stop"), _reply("<calc>2+2</calc>", [3], "1", "stop")]
    )
    sample = Sample(index=0, prompt="Q")

    asyncio.run(calculator_agent.generate(_args(), sample, {"max_new_tokens": 1000}))

    assert len(server.payloads) == 2 and sample.status == Sample.Status.TRUNCATED
    assert sample.tokens[-1] == 3 and sample.loss_mask[-1] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
