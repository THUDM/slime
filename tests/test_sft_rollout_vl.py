"""Unit tests for the VL SFT rollout.

Two layers, because the thing being tested has two halves:

* the wiring -- that `encode_multimodal_sample` calls the processor the way the
  alignment needs and packages its output for the trainer. Stubbed, so it runs
  anywhere, with no checkpoint and no GPU;
* the alignment itself -- that the loss mask lands on the assistant turn once
  the processor has expanded the image placeholder. That needs a real chat
  template and a real fast tokenizer, so it is skipped unless
  ``SLIME_TEST_QWEN3_5_VL_CHECKPOINT`` points at a staged checkpoint.

The second layer is the one that would catch a silent regression: the alignment
is a left-pad of zeros onto a mask computed from the text-only projection of the
messages (`MultiTurnLossMaskGenerator.get_loss_mask_with_multimodal_alignment`),
which is correct only while every token the processor adds sits before the first
trainable one.
"""

import os

import pytest

from slime.rollout.sft_rollout_vl import encode_multimodal_sample, has_multimodal_content

NUM_GPUS = 0

CHECKPOINT = os.environ.get("SLIME_TEST_QWEN3_5_VL_CHECKPOINT", "")

PROBLEM = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n<image>Find x."
)
ANSWER = "Answer: \\boxed{3}"


def _messages():
    """What `_build_messages` produces from a geo3k row plus --multimodal-keys."""
    head, tail = PROBLEM.split("<image>")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": head},
                {"type": "image", "image": "data:image/png;base64,AAAA"},
                {"type": "text", "text": tail},
            ],
        },
        {"role": "assistant", "content": ANSWER},
    ]


class _StubProcessor:
    """Minimal stand-in: records how it was called, returns a fixed encoding."""

    def __init__(self, input_ids, extra=None):
        self._input_ids = input_ids
        self._extra = extra if extra is not None else {"pixel_values": "PIXELS", "image_grid_thw": "GRID"}
        self.template_kwargs = None
        self.call_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "RENDERED"

    def __call__(self, text=None, **kwargs):
        self.call_kwargs = {"text": text, **kwargs}
        return {"input_ids": [self._input_ids], "attention_mask": [[1] * len(self._input_ids)], **self._extra}


class _StubMaskGenerator:
    def __init__(self):
        self.seen = None

    def get_loss_mask_with_multimodal_alignment(self, messages, input_ids, tools=None):
        self.seen = (messages, list(input_ids), tools)
        return list(input_ids), [0] * (len(input_ids) - 2) + [1, 1]


@pytest.mark.unit
def test_has_multimodal_content_ignores_the_all_none_dict():
    # Dataset sets every declared modality, with None for the ones a row lacks.
    assert not has_multimodal_content(None)
    assert not has_multimodal_content({})
    assert not has_multimodal_content({"images": None, "videos": None})
    assert has_multimodal_content({"images": ["an image"], "videos": None})


@pytest.mark.unit
def test_encode_renders_the_assistant_turn_into_the_text():
    processor = _StubProcessor(input_ids=[10, 11, 12, 13])
    generator = _StubMaskGenerator()

    encode_multimodal_sample(_messages(), {"images": ["img"], "videos": None}, processor, generator)

    # add_generation_prompt=False is what keeps the assistant turn -- the thing
    # SFT trains on -- inside the rendered text.
    assert processor.template_kwargs["add_generation_prompt"] is False
    assert processor.template_kwargs["tokenize"] is False


@pytest.mark.unit
def test_encode_passes_the_images_and_the_forced_return_tensors():
    processor = _StubProcessor(input_ids=[10, 11, 12, 13])
    generator = _StubMaskGenerator()

    encode_multimodal_sample(_messages(), {"images": ["img"], "videos": None}, processor, generator)

    call = processor.call_kwargs
    assert call["text"] == "RENDERED"
    assert call["images"] == ["img"]
    # build_processor_kwargs: list token ids for text, tensors for the modalities.
    assert call["text_kwargs"]["return_tensors"] is None
    assert call["images_kwargs"]["return_tensors"] == "pt"


@pytest.mark.unit
def test_encode_hands_the_mask_generator_the_processor_token_ids():
    processor = _StubProcessor(input_ids=[10, 11, 12, 13])
    generator = _StubMaskGenerator()

    token_ids, loss_mask, _ = encode_multimodal_sample(
        _messages(), {"images": ["img"], "videos": None}, processor, generator, tools=["a tool"]
    )

    _, seen_input_ids, seen_tools = generator.seen
    # [0], not the batch: build_processor_kwargs asks for list token ids.
    assert seen_input_ids == [10, 11, 12, 13]
    assert seen_tools == ["a tool"]
    assert len(token_ids) == len(loss_mask) == 4


@pytest.mark.unit
def test_multimodal_train_inputs_drops_the_token_stream():
    processor = _StubProcessor(input_ids=[10, 11, 12, 13])

    _, _, multimodal_train_inputs = encode_multimodal_sample(
        _messages(), {"images": ["img"], "videos": None}, processor, _StubMaskGenerator()
    )

    # These are forward kwargs for the model provider; input_ids travel as
    # sample.tokens and attention_mask is rebuilt from cu_seqlens.
    assert multimodal_train_inputs == {"pixel_values": "PIXELS", "image_grid_thw": "GRID"}


@pytest.mark.unit
def test_multimodal_train_inputs_is_none_when_the_processor_returned_only_tokens():
    processor = _StubProcessor(input_ids=[10, 11], extra={})

    _, _, multimodal_train_inputs = encode_multimodal_sample(
        _messages(), {"images": ["img"], "videos": None}, processor, _StubMaskGenerator()
    )

    # None rather than {}, so slime/ray/rollout.py's `is not None` test is false
    # and the batch carries no multimodal key at all.
    assert multimodal_train_inputs is None


@pytest.mark.unit
@pytest.mark.skipif(not CHECKPOINT, reason="set SLIME_TEST_QWEN3_5_VL_CHECKPOINT to a staged checkpoint")
def test_alignment_puts_the_mask_exactly_on_the_assistant_turn():
    """The real thing: a real template, a real fast tokenizer, a real mask.

    The processor is stood in for by the tokenizer plus the placeholder
    expansion it would do, which is all the alignment can see. An odd image
    token count is deliberate -- packed sequences of odd length are what broke
    the RL path once (`get_packed_cp_local_indices`).
    """
    from transformers import AutoTokenizer

    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    assert tokenizer.is_fast, "qwen3_5 masking needs offset_mapping"
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")

    messages = _messages()
    image_tokens = 137

    class _TokenizerBackedProcessor:
        def apply_chat_template(self, conversation, **kwargs):
            kwargs.pop("tools", None)
            return tokenizer.apply_chat_template(conversation, **kwargs)

        def __call__(self, text=None, **kwargs):
            expanded = text.replace("<|image_pad|>", "<|image_pad|>" * image_tokens)
            return {
                "input_ids": [tokenizer(expanded, add_special_tokens=False)["input_ids"]],
                "pixel_values": "PIXELS",
                "image_grid_thw": "GRID",
            }

    token_ids, loss_mask, multimodal_train_inputs = encode_multimodal_sample(
        messages, {"images": ["img"], "videos": None}, _TokenizerBackedProcessor(), generator
    )

    assert len(token_ids) == len(loss_mask)
    assert token_ids.count(tokenizer.convert_tokens_to_ids("<|image_pad|>")) == image_tokens
    assert multimodal_train_inputs == {"pixel_values": "PIXELS", "image_grid_thw": "GRID"}

    trained = "".join(generator.get_text_from_loss_mask(token_ids, loss_mask))
    assert ANSWER in trained
    # Nothing from the prompt leaked into the trained span.
    assert "Find x." not in trained
    assert "<|image_pad|>" not in trained

    # And it is the same span the text-only path would train on.
    text_messages = [{"role": "user", "content": PROBLEM}, {"role": "assistant", "content": ANSWER}]
    text_ids, text_mask = generator.get_loss_mask(text_messages)
    assert "".join(generator.get_text_from_loss_mask(text_ids, text_mask)) == trained
    assert sum(loss_mask) == sum(text_mask)
