"""CPU unit test for ``slime.utils.data.filter_long_prompt`` on multimodal samples.

Regression test for the bug where length-filtering a multimodal dataset crashed.

When ``apply_chat_template=True``, ``Sample.prompt`` holds the *rendered string*
(images collapsed into ``<|image_pad|>`` placeholders), while the structured
images live in ``Sample.multimodal_inputs`` (already computed during dataset
construction). The multimodal branch of ``filter_long_prompt`` re-derived vision
info from the *string* prompt via ``process_vision_info`` -- which expects a
*conversation list* -- and crashed with ``TypeError: string indices must be
integers, not 'str'``.

The fix reuses the already-computed ``Sample.multimodal_inputs`` instead of
re-extracting from the string prompt.
"""

from __future__ import annotations

import types

import pytest

from slime.utils.data import filter_long_prompt
from slime.utils.types import Sample


class _FakeProcessor:
    """Stand-in for a HF VL processor.

    Returns ``input_ids`` whose length equals the number of images it is given,
    so the test can assert that filtering used the images from
    ``Sample.multimodal_inputs`` (1 image -> length 1, 3 images -> length 3).
    """

    def __init__(self):
        self.image_processor = types.SimpleNamespace(patch_size=14)

    def __call__(self, text=None, images=None, videos=None, **kwargs):
        n_images = len(images) if images else 0
        return {"input_ids": [list(range(n_images))]}


def _mm_sample(n_images: int, prompt: str) -> Sample:
    # prompt is the rendered string produced by apply_chat_template; the
    # structured images are kept in multimodal_inputs.
    return Sample(prompt=prompt, multimodal_inputs={"images": ["img"] * n_images, "videos": None})


@pytest.mark.unit
def test_filter_long_prompt_multimodal_reuses_stored_inputs(monkeypatch):
    # Faithfully reproduce the crash: feeding a *string* prompt to
    # process_vision_info (qwen_vl_utils) raises. The fixed code must not call it.
    def _boom(prompt, processor):
        raise TypeError("string indices must be integers, not 'str'")

    monkeypatch.setattr("slime.utils.processing_utils.process_vision_info", _boom)

    processor = _FakeProcessor()
    short = _mm_sample(1, prompt="<rendered short prompt>")
    long = _mm_sample(3, prompt="<rendered long prompt>")

    # max_length=2: short (1 image -> 1 token) kept, long (3 images -> 3 tokens) dropped.
    result = filter_long_prompt([short, long], tokenizer=None, processor=processor, max_length=2)

    assert result == [short]


class _FakeTokenizer:
    """Batched tokenizer for text-only samples: token count = word count."""

    def __call__(self, prompts, add_special_tokens=False):
        return {"input_ids": [list(range(len(p.split()))) for p in prompts]}


@pytest.mark.unit
def test_filter_long_prompt_mixed_text_and_multimodal(monkeypatch):
    # The multimodal branch must not re-derive vision info from the string prompt.
    def _boom(prompt, processor):
        raise TypeError("string indices must be integers, not 'str'")

    monkeypatch.setattr("slime.utils.processing_utils.process_vision_info", _boom)

    processor = _FakeProcessor()
    tokenizer = _FakeTokenizer()

    text_short = Sample(prompt="a b", multimodal_inputs=None)  # 2 tokens -> kept
    text_long = Sample(prompt="a b c d", multimodal_inputs=None)  # 4 tokens -> dropped
    mm_short = _mm_sample(1, prompt="<rendered>")  # 1 image -> kept
    mm_long = _mm_sample(5, prompt="<rendered>")  # 5 images -> dropped

    result = filter_long_prompt(
        [text_short, text_long, mm_short, mm_long],
        tokenizer=tokenizer,
        processor=processor,
        max_length=2,
    )

    # Both branches filter independently; order within each branch is preserved.
    assert text_short in result and mm_short in result
    assert text_long not in result and mm_long not in result
