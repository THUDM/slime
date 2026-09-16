import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample

NUM_GPUS = 0
IMAGE_TOKEN = 99


@pytest.fixture
def rollout(monkeypatch):
    # The interaction loop does not need a live router or a rollout worker.
    sglang_rollout = ModuleType("slime.rollout.sglang_rollout")
    sglang_rollout.GenerateState = None
    monkeypatch.setitem(sys.modules, sglang_rollout.__name__, sglang_rollout)
    vision_utils = ModuleType("qwen_vl_utils")
    vision_utils.process_vision_info = lambda messages: (messages[0].get("images"), None)
    monkeypatch.setitem(sys.modules, vision_utils.__name__, vision_utils)
    path = Path(__file__).resolve().parents[1] / "examples/geo3k_vlm_multi_turn/rollout.py"
    spec = importlib.util.spec_from_file_location("geo3k_rollout_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "encode_image_for_rollout_engine", lambda image: image)
    return module


class Tokenizer:
    def __init__(self, initial_image, observation_image, bos_token_id):
        self.bos_token_id = bos_token_id
        self.initial_ids = [10] + ([IMAGE_TOKEN] if initial_image else []) + [11]
        self.observation_ids = ([bos_token_id] if bos_token_id is not None else []) + [20]
        self.observation_ids += ([IMAGE_TOKEN] if observation_image else []) + [21]

    def apply_chat_template(self, messages, **kwargs):
        return "dummy" if len(messages) == 2 else "dummy+observation"

    def encode(self, text, **kwargs):
        # Generated IDs must never be reconstructed from decoded response text.
        return {
            "initial": self.initial_ids,
            "dummy": [30, 31],
            "dummy+observation": [30, 31] + self.observation_ids,
        }[text].copy()

    def decode(self, tokens, **kwargs):
        return "decoded response"


class Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, images=None, **kwargs):
        ids = self.tokenizer.encode(text)
        expanded = [token for value in ids for token in ([value] * 3 if value == IMAGE_TOKEN else [value])]
        output = {"input_ids": [expanded], "attention_mask": [[1] * len(expanded)]}
        if images:
            output["pixel_values"] = torch.ones(len(images), 2)
            output["image_grid_thw"] = torch.tensor([[1, 2, 6]] * len(images))
        return output


@pytest.mark.unit
@pytest.mark.parametrize(
    "initial_image,observation_image,use_processor",
    [(True, False, True), (True, True, True), (False, True, True), (False, False, True), (False, False, False)],
)
@pytest.mark.parametrize("bos_token_id", [None, 101])
def test_multiturn_keeps_rollout_ids_separate_from_expanded_training_ids(
    rollout, monkeypatch, initial_image, observation_image, use_processor, bos_token_id
):
    tokenizer = Tokenizer(initial_image, observation_image, bos_token_id)
    processor = Processor(tokenizer)
    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_context_len=100,
        apply_chat_template=True,
        apply_chat_template_kwargs={},
    )
    sample = Sample(prompt="initial", multimodal_inputs={"images": ["first"]} if initial_image else None)
    responses = [[479, 3770, 3], [42, 3]]
    requests = []

    class Env:
        def __init__(self):
            self.turn = 0
            self.closed = False

        def reset(self):
            pass

        def step(self, text):
            self.turn += 1
            return {}, self.turn == 2, {}

        def format_observation(self, observation):
            return {"role": "user", "content": "feedback", "images": ["second"] if observation_image else None}

        def close(self):
            self.closed = True

    env = Env()
    state = SimpleNamespace(tokenizer=tokenizer, processor=processor if use_processor else None)
    monkeypatch.setattr(
        rollout, "_initialize_resources", lambda *unused: (env, None, {"max_turns": 2}, state, "unused")
    )

    async def post(url, payload):
        requests.append(
            {
                "input_ids": list(payload["input_ids"]),
                "image_data": list(payload.get("image_data", [])),
                "max_new_tokens": payload["sampling_params"]["max_new_tokens"],
            }
        )
        ids = responses[len(requests) - 1]
        return {
            "text": "decoded response",
            "meta_info": {
                "output_token_logprobs": [(-0.5, token, None) for token in ids],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(rollout, "post", post)
    result = asyncio.run(rollout.generate(args, sample, {"max_new_tokens": 100}))
    observation_ids = tokenizer.observation_ids[int(bos_token_id is not None) :]
    initial_train_ids = processor("initial")["input_ids"][0]
    observation_train_ids = [
        token for value in observation_ids for token in ([value] * 3 if value == IMAGE_TOKEN else [value])
    ]
    expected_response = responses[0] + observation_train_ids + responses[1]

    assert requests[0]["input_ids"] == tokenizer.initial_ids
    assert requests[1]["input_ids"] == tokenizer.initial_ids + responses[0] + observation_ids
    assert requests[0]["image_data"] == (["first"] if initial_image else [])
    assert requests[1]["image_data"] == (["first"] if initial_image else []) + (
        ["second"] if observation_image else []
    )
    assert result.tokens == initial_train_ids + expected_response
    assert result.response_length == len(expected_response)
    assert result.loss_mask == [1] * 3 + [0] * len(observation_train_ids) + [1] * 2
    assert result.rollout_log_probs == [-0.5] * 3 + [0.0] * len(observation_train_ids) + [-0.5] * 2
    assert requests[0]["max_new_tokens"] == 100 - len(initial_train_ids)
    assert requests[1]["max_new_tokens"] == 100 - len(initial_train_ids) - 3 - len(observation_train_ids)
    if initial_image or observation_image:
        assert result.multimodal_train_inputs["pixel_values"].shape[0] == int(initial_image) + int(observation_image)
    assert result.status == Sample.Status.COMPLETED
    assert env.closed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
