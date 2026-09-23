"""Parity of the payload that ``generate()`` sends to SGLang.

The async change touches the two lines in ``generate()`` that build the request
body (``input_ids`` for text, ``image_data`` + ``text`` for multimodal). This
test drives the *real* ``generate()`` with a stubbed ``post`` that captures the
outgoing payload, then asserts it is identical to a reference payload
reconstructed with the original *synchronous* helpers (which still live in the
tree). If the async path ever changed what SGLang receives -- different
``input_ids``, reordered/altered ``image_data``, missing ``text`` -- this fails.

No GPU / model needed: a deterministic fake tokenizer + processor stand in for
the HF objects, and real PIL images drive the base64 encoding.
"""

from __future__ import annotations

import asyncio
import copy
from argparse import Namespace
from types import SimpleNamespace

import pytest
from PIL import Image

from slime.rollout import sglang_rollout
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

NUM_GPUS = 0


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


class _FakeProcessor:
    def __call__(self, *, text, **kwargs):
        num_images = len(kwargs.get("images", []) or [])
        return {
            "input_ids": [[ord(c) for c in text]],
            "attention_mask": [[1] * len(text)],
            "pixel_values": f"px::{text}",
            "image_grid_thw": f"thw::{num_images}",
        }


class _PayloadCaptured(Exception):
    """Raised by the stubbed post() so we stop right after the payload is built."""


def _make_args():
    return Namespace(
        ci_test=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=12345,
        use_rollout_routing_replay=False,
        router_policy=None,
    )


def _build_reference_payload(sample, tokenizer, processor, sampling_params):
    """Reconstruct the SGLang payload using the ORIGINAL synchronous helpers."""
    ref_sample = copy.deepcopy(sample)
    sp = sampling_params.copy()
    sp["max_new_tokens"] -= ref_sample.response_length
    payload = {"sampling_params": sp, "return_logprob": True}

    images = ref_sample.multimodal_inputs.get("images") if ref_sample.multimodal_inputs else None
    if images:
        payload["image_data"] = [encode_image_for_rollout_engine(im) for im in images]
        payload["text"] = ref_sample.prompt
    else:
        payload["input_ids"] = sglang_rollout._prepare_prompt_ids(ref_sample, tokenizer, processor)
    return payload


@pytest.fixture
def captured_generate(monkeypatch):
    """Install a fake GenerateState + capturing post(), return a driver helper."""
    tokenizer = _FakeTokenizer()
    processor = _FakeProcessor()
    fake_state = SimpleNamespace(tokenizer=tokenizer, processor=processor)

    # SingletonMeta caches GenerateState by class; inject our fake instance.
    instances = sglang_rollout.GenerateState._instances
    had_instance = sglang_rollout.GenerateState in instances
    saved = instances.get(sglang_rollout.GenerateState)
    instances[sglang_rollout.GenerateState] = fake_state

    captured = {}

    async def fake_post(url, payload, headers=None, **_kwargs):
        captured["url"] = url
        captured["payload"] = copy.deepcopy(payload)
        captured["headers"] = headers
        raise _PayloadCaptured

    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    def drive(sample, sampling_params):
        args = _make_args()
        with pytest.raises(_PayloadCaptured):
            asyncio.run(sglang_rollout.generate(args, sample, sampling_params.copy()))
        return captured, tokenizer, processor

    try:
        yield drive
    finally:
        if had_instance:
            instances[sglang_rollout.GenerateState] = saved
        else:
            instances.pop(sglang_rollout.GenerateState, None)


@pytest.mark.unit
def test_text_only_payload_matches_sync_reference(captured_generate):
    sample = Sample(prompt="what is 2 + 2?")
    sampling_params = {"max_new_tokens": 128, "temperature": 0.0}

    captured, tokenizer, processor = captured_generate(sample, sampling_params)
    reference = _build_reference_payload(Sample(prompt="what is 2 + 2?"), tokenizer, processor, sampling_params)

    assert captured["payload"] == reference
    assert "image_data" not in captured["payload"]
    assert captured["payload"]["input_ids"] == [ord(c) for c in "what is 2 + 2?"]


@pytest.mark.unit
def test_multi_image_payload_matches_sync_reference(captured_generate):
    images = [
        Image.new("RGB", (4, 4), color=(200, 0, 0)),
        Image.new("RGB", (6, 6), color=(0, 200, 0)),
        Image.new("RGB", (8, 8), color=(0, 0, 200)),
    ]
    prompt = "<image><image><image> compare the three tiles."
    sample = Sample(prompt=prompt, multimodal_inputs={"images": images, "videos": None})
    sampling_params = {"max_new_tokens": 256, "temperature": 0.0}

    captured, tokenizer, processor = captured_generate(sample, sampling_params)

    ref_sample = Sample(prompt=prompt, multimodal_inputs={"images": images, "videos": None})
    reference = _build_reference_payload(ref_sample, tokenizer, processor, sampling_params)

    assert captured["payload"] == reference
    # multimodal path sends text + image_data, never input_ids
    assert captured["payload"]["text"] == prompt
    assert "input_ids" not in captured["payload"]
    # image_data preserves image order and content
    assert captured["payload"]["image_data"] == [encode_image_for_rollout_engine(im) for im in images]
    assert len(captured["payload"]["image_data"]) == 3


@pytest.mark.unit
def test_sampling_params_max_new_tokens_decrement_is_preserved(captured_generate):
    sample = Sample(prompt="abc")
    sample.response_length = 10
    sampling_params = {"max_new_tokens": 100, "temperature": 0.0}

    captured, _tok, _proc = captured_generate(sample, sampling_params)

    # generate() subtracts the already-produced response length, unchanged by async
    assert captured["payload"]["sampling_params"]["max_new_tokens"] == 90


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
