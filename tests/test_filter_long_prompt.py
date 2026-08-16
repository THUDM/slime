"""CPU unit tests for ``slime.utils.data.filter_long_prompt``.

With a processor configured, the function scores text-only samples with a
batched tokenizer call and multimodal samples one at a time through the
processor. Splitting the work that way is a throughput optimization; it must not
change which samples survive, or the order they survive in.

Order matters concretely: ``--rollout-shuffle`` defaults to False, so
``Dataset.samples`` is consumed in exactly the order this function returns.
Grouping the survivors by modality would make a mixed dataset train every
text-only prompt before any prompt carrying an image.
"""

from __future__ import annotations

import sys
import types

import pytest

from slime.utils.data import _build_messages, filter_long_prompt
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.fixture
def stub_process_vision_info(monkeypatch):
    """`slime.utils.processing_utils` pulls in transformers + PIL.

    The multimodal branch imports it lazily, and only `process_vision_info` is
    needed here, so stub the module rather than requiring transformers on the
    CPU image.
    """
    module = types.ModuleType("slime.utils.processing_utils")
    module.process_vision_info = lambda prompt, processor: {"images": None}
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", module)


class _Tokenizer:
    """Batched tokenizer stand-in: prompt "pN:len" tokenizes to `len` ids."""

    def __call__(self, prompts, add_special_tokens=False):
        return {"input_ids": [[0] * _encoded_length(p) for p in prompts]}


class _Processor:
    """Per-sample processor stand-in, same length convention."""

    def __call__(self, text=None, **kwargs):
        return {"input_ids": [[0] * _encoded_length(text)]}


def _encoded_length(prompt: str) -> int:
    return int(prompt.split(":")[1])


def _make_samples(specs):
    """specs: list of (is_multimodal, encoded_length)."""
    samples = []
    for i, (is_multimodal, length) in enumerate(specs):
        sample = Sample(prompt=f"p{i}:{length}")
        sample.multimodal_inputs = {"images": ["img"]} if is_multimodal else None
        samples.append(sample)
    return samples


@pytest.mark.unit
def test_preserves_order_when_nothing_is_filtered(stub_process_vision_info):
    # Alternating modality, every prompt well under the limit.
    samples = _make_samples([(i % 2 == 0, 5) for i in range(6)])

    kept = filter_long_prompt(samples, _Tokenizer(), _Processor(), max_length=100)

    assert [s.prompt for s in kept] == [s.prompt for s in samples]


@pytest.mark.unit
def test_preserves_order_when_some_are_filtered(stub_process_vision_info):
    specs = [
        (True, 5),  # p0 multimodal, keep
        (False, 500),  # p1 text-only, drop
        (False, 5),  # p2 text-only, keep
        (True, 500),  # p3 multimodal, drop
        (False, 5),  # p4 text-only, keep
        (True, 5),  # p5 multimodal, keep
    ]
    samples = _make_samples(specs)

    kept = filter_long_prompt(samples, _Tokenizer(), _Processor(), max_length=100)

    assert [s.prompt for s in kept] == ["p0:5", "p2:5", "p4:5", "p5:5"]


@pytest.mark.unit
@pytest.mark.parametrize("all_multimodal", [True, False])
def test_single_modality_batches_are_unchanged(stub_process_vision_info, all_multimodal):
    samples = _make_samples([(all_multimodal, 5)] * 4)

    kept = filter_long_prompt(samples, _Tokenizer(), _Processor(), max_length=100)

    assert [s.prompt for s in kept] == [s.prompt for s in samples]


@pytest.mark.unit
def test_no_processor_path_still_preserves_order():
    samples = _make_samples([(False, 5), (False, 500), (False, 5)])

    kept = filter_long_prompt(samples, _Tokenizer(), None, max_length=100)

    assert [s.prompt for s in kept] == ["p0:5", "p2:5"]


@pytest.mark.unit
def test_text_only_row_keeps_string_content_with_multimodal_config():
    data = {"prompt": "hello"}

    prompt = _build_messages(data, "prompt", as_conversation=True, multimodal_keys={"image": "images"})

    assert prompt == [{"role": "user", "content": "hello"}]


@pytest.mark.unit
def test_multimodal_placeholder_expansion_is_unchanged():
    data = {"prompt": "look <image> here", "images": ["image.png"]}

    prompt = _build_messages(data, "prompt", as_conversation=True, multimodal_keys={"image": "images"})

    assert prompt == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look "},
                {"type": "image", "image": "image.png"},
                {"type": "text", "text": " here"},
            ],
        }
    ]
