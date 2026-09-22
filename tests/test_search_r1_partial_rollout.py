from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_R1_DIR = REPO_ROOT / "examples" / "search-r1"
for path in (REPO_ROOT, SEARCH_R1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "sglang_router" not in sys.modules:
    sglang_router_stub = types.ModuleType("sglang_router")
    sglang_router_stub.__version__ = "0.2.3"
    sys.modules["sglang_router"] = sglang_router_stub

if "ray" not in sys.modules:
    ray_stub = types.ModuleType("ray")
    ray_stub._private = types.SimpleNamespace(services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"))
    sys.modules["ray"] = ray_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    for name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(transformers_stub, name, type(name, (), {}))
    sys.modules["transformers"] = transformers_stub

sglang_rollout_stub = types.ModuleType("slime.rollout.sglang_rollout")


class _StubGenerateState:
    pass


sglang_rollout_stub.GenerateState = _StubGenerateState
previous_sglang_rollout = sys.modules.get("slime.rollout.sglang_rollout")
sys.modules["slime.rollout.sglang_rollout"] = sglang_rollout_stub

import generate_with_search as search_gen  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

if previous_sglang_rollout is None:
    sys.modules.pop("slime.rollout.sglang_rollout", None)
else:
    sys.modules["slime.rollout.sglang_rollout"] = previous_sglang_rollout

NUM_GPUS = 0


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) for ch in text]


class FakeGenerateState:
    def __init__(self, args) -> None:
        self.args = args
        self.tokenizer = FakeTokenizer()
        self.aborted = False


def _args(*, partial_rollout: bool = True, mask_offpolicy_in_partial_rollout: bool = False):
    return SimpleNamespace(
        partial_rollout=partial_rollout,
        mask_offpolicy_in_partial_rollout=mask_offpolicy_in_partial_rollout,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1234,
        sglang_speculative_algorithm=False,
    )


def _output(text: str, token_base: int, finish_reason: str = "stop"):
    return {
        "text": text,
        "meta_info": {
            "finish_reason": {"type": finish_reason},
            "output_token_logprobs": [[-0.1 * (i + 1), token_base + i] for i in range(len(text))],
        },
    }


@pytest.fixture(autouse=True)
def patch_generate_state(monkeypatch):
    monkeypatch.setattr(search_gen, "GenerateState", FakeGenerateState)
    monkeypatch.setitem(search_gen.SEARCH_R1_CONFIGS, "return_logprob", True)
    monkeypatch.setitem(search_gen.SEARCH_R1_CONFIGS, "max_turns", 2)
    monkeypatch.setattr(search_gen, "execute_predictions", _fake_execute_predictions)


async def _fake_execute_predictions(prediction: str):
    action, content = search_gen.postprocess_predictions(prediction)
    if action == "search":
        return f"\n\n<information>result for {content}</information>\n\n", False
    if action == "answer":
        return "", True
    return "\ninvalid action\n", False


def test_search_r1_fresh_rollout_records_turn_state_and_keeps_logprob_alignment(monkeypatch):
    calls = []
    outputs = [
        _output("<search>cats</search>", 100),
        _output("<answer>cats</answer>", 200),
    ]

    async def fake_post(_url, payload):
        calls.append(payload)
        return outputs.pop(0)

    monkeypatch.setattr(search_gen, "post", fake_post)
    sample = Sample(prompt="Question: cats\n", status=Sample.Status.PENDING)

    result = asyncio.run(search_gen.generate(_args(), sample, {"stop": "</old>"}))

    assert result.status is Sample.Status.COMPLETED
    assert result.metadata[search_gen._SEARCH_R1_TURN_COUNT_KEY] == 2
    assert result.response == (
        "<search>cats</search>\n\n" "<information>result for cats</information>\n\n" "<answer>cats</answer>"
    )
    assert len(result.loss_mask) == result.response_length
    assert len(result.rollout_log_probs) == result.response_length
    assert sum(result.loss_mask) == len("<search>cats</search>") + len("<answer>cats</answer>")
    assert calls[0]["text"] == "Question: cats\n"
    assert (
        calls[1]["text"] == "Question: cats\n<search>cats</search>\n\n<information>result for cats</information>\n\n"
    )
    assert calls[0]["sampling_params"]["stop"] == ["</old>", "</search>", "</answer>"]


def test_search_r1_partial_rollout_resumes_without_clearing_existing_trajectory(monkeypatch):
    old_response = "<search>cats</search>\n\n<information>result for cats</information>\n\n"
    old_prompt = "Question: cats\n"
    old_prompt_tokens = FakeTokenizer().encode(old_prompt)
    old_response_tokens = [10] * len(old_response)
    old_log_probs = [-0.7] * len(old_response)
    calls = []

    async def fake_post(_url, payload):
        calls.append(payload)
        return _output("<answer>cats</answer>", 300)

    monkeypatch.setattr(search_gen, "post", fake_post)
    sample = Sample(
        prompt=old_prompt,
        tokens=old_prompt_tokens + old_response_tokens,
        response=old_response,
        response_length=len(old_response),
        loss_mask=[0] * len(old_response),
        rollout_log_probs=list(old_log_probs),
        status=Sample.Status.ABORTED,
        metadata={search_gen._SEARCH_R1_TURN_COUNT_KEY: 1},
    )

    result = asyncio.run(search_gen.generate(_args(mask_offpolicy_in_partial_rollout=True), sample, {}))

    assert result.status is Sample.Status.COMPLETED
    assert result.response == old_response + "<answer>cats</answer>"
    assert (
        result.tokens[: len(old_prompt_tokens) + len(old_response_tokens)] == old_prompt_tokens + old_response_tokens
    )
    assert result.rollout_log_probs[: len(old_log_probs)] == old_log_probs
    assert result.loss_mask[: len(old_response)] == [0] * len(old_response)
    assert result.loss_mask[len(old_response) :] == [1] * len("<answer>cats</answer>")
    assert len(result.rollout_log_probs) == result.response_length
    assert result.metadata[search_gen._SEARCH_R1_TURN_COUNT_KEY] == 2
    assert calls == [
        {
            "text": old_prompt + old_response,
            "sampling_params": {"stop": ["</search>", "</answer>"]},
            "return_logprob": True,
        }
    ]


def test_search_r1_partial_rollout_falls_back_to_response_tags_for_older_samples(monkeypatch):
    old_response = "<search>cats</search>\n\n<information>result for cats</information>\n\n"
    calls = []

    async def fake_post(_url, payload):
        calls.append(payload)
        return _output("<answer>cats</answer>", 400)

    monkeypatch.setattr(search_gen, "post", fake_post)
    sample = Sample(
        prompt="Question: cats\n",
        response=old_response,
        response_length=len(old_response),
        tokens=FakeTokenizer().encode("Question: cats\n") + [1] * len(old_response),
        loss_mask=[0] * len(old_response),
        rollout_log_probs=[0.0] * len(old_response),
        status=Sample.Status.ABORTED,
    )

    result = asyncio.run(search_gen.generate(_args(), sample, {}))

    assert result.status is Sample.Status.COMPLETED
    assert len(calls) == 1
    assert result.metadata[search_gen._SEARCH_R1_TURN_COUNT_KEY] == 2


def test_search_r1_partial_rollout_with_completed_answer_does_not_generate(monkeypatch):
    async def fake_post(_url, _payload):
        raise AssertionError("completed partial answer should not call SGLang")

    monkeypatch.setattr(search_gen, "post", fake_post)
    sample = Sample(
        prompt="Question: cats\n",
        response="<answer>cats</answer>",
        response_length=len("<answer>cats</answer>"),
        tokens=FakeTokenizer().encode("Question: cats\n") + [1] * len("<answer>cats</answer>"),
        loss_mask=[1] * len("<answer>cats</answer>"),
        rollout_log_probs=[-0.1] * len("<answer>cats</answer>"),
        status=Sample.Status.ABORTED,
        metadata={search_gen._SEARCH_R1_TURN_COUNT_KEY: 1},
    )

    result = asyncio.run(search_gen.generate(_args(), sample, {}))

    assert result.status is Sample.Status.COMPLETED
    assert result.response == "<answer>cats</answer>"


def test_search_r1_partial_rollout_supports_non_logprob_mode(monkeypatch):
    monkeypatch.setitem(search_gen.SEARCH_R1_CONFIGS, "return_logprob", False)

    async def fake_post(_url, _payload):
        return {
            "text": "<answer>cats</answer> trailing junk",
            "meta_info": {"finish_reason": {"type": "stop"}},
        }

    monkeypatch.setattr(search_gen, "post", fake_post)
    sample = Sample(prompt="Question: cats\n", status=Sample.Status.PENDING)

    result = asyncio.run(search_gen.generate(_args(), sample, {}))

    assert result.status is Sample.Status.COMPLETED
    assert result.response == "<answer>cats</answer>"
    assert result.rollout_log_probs is None
    assert len(result.loss_mask) == result.response_length
    assert result.loss_mask == [1] * len("<answer>cats</answer>")
