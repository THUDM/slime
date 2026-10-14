import pytest
from test_loss_mask_type_qwen35 import FakeQwen35Tokenizer

from slime.utils.mask_utils import MultiTurnLossMaskGenerator

NUM_GPUS = 0

# Stands in for <|image_pad|>. The one thing a processor does that a tokenizer
# cannot is expand a single placeholder into the number of tokens the vision
# encoder will emit for that particular image, so the fake does exactly that and
# nothing else.
IMAGE_PAD = "░"
PADS_PER_IMAGE = 5


def _processor_input_ids(tokenizer, messages):
    """The input_ids a processor would return for `messages`."""
    flattened = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            rendered = "".join(
                item["text"] if item["type"] == "text" else IMAGE_PAD * PADS_PER_IMAGE for item in content
            )
            flattened.append({**message, "content": rendered})
        else:
            flattened.append(message)
    return tokenizer(tokenizer.render(flattened), add_special_tokens=False)["input_ids"]


def _supervised_text(tokenizer, token_ids, loss_mask):
    return tokenizer.decode([token_id for token_id, mask in zip(token_ids, loss_mask, strict=True) if mask == 1])


# The fake tokenizer models Qwen3.5 formatting; "qwen" takes a different branch
# in get_loss_mask() that needs a real tokenizer's added vocab.
@pytest.mark.parametrize("loss_mask_type", ["qwen3", "qwen3_5"])
def test_image_before_the_assistant_turn_supervises_only_the_reply(loss_mask_type):
    """The shape SFT on a VLM actually has: one image, one user turn, one reply."""
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this."},
                {"type": "image", "image": "data:image/png;base64,AAAA"},
                {"type": "text", "text": "Find x."},
            ],
        },
        {"role": "assistant", "content": "x is 3."},
    ]
    input_ids = _processor_input_ids(tokenizer, messages)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=loss_mask_type)

    token_ids, loss_mask = generator.get_loss_mask_with_multimodal_alignment(messages, input_ids)

    assert token_ids == input_ids
    assert len(loss_mask) == len(input_ids)
    # The supervised span is the reply, and nothing of the expanded image.
    supervised = _supervised_text(tokenizer, token_ids, loss_mask)
    assert "x is 3." in supervised
    assert IMAGE_PAD not in supervised
    # Contiguous suffix, which is what response_length assumes.
    assert loss_mask[loss_mask.index(1) :] == [1] * (len(loss_mask) - loss_mask.index(1))


def test_images_only_in_the_first_user_turn_still_align():
    """Multi-turn is fine as long as no image follows an assistant turn."""
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image/png;base64,AAAA"},
                {"type": "text", "text": "Find x."},
            ],
        },
        {"role": "assistant", "content": "x is 3."},
        {"role": "user", "content": "And y?"},
        {"role": "assistant", "content": "y is 4."},
    ]
    input_ids = _processor_input_ids(tokenizer, messages)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")

    token_ids, loss_mask = generator.get_loss_mask_with_multimodal_alignment(messages, input_ids)

    supervised = _supervised_text(tokenizer, token_ids, loss_mask)
    assert "x is 3." in supervised
    assert "y is 4." in supervised
    assert IMAGE_PAD not in supervised


def test_image_after_an_assistant_turn_is_rejected():
    """The case the leading-zero alignment cannot express.

    Without the check this does not raise -- it silently pads the mask past the
    first reply, so training supervises image pad tokens and never sees
    "x is 3." at all. Verified against the real Qwen3.5-VL processor before this
    test was written.
    """
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image/png;base64,AAAA"},
                {"type": "text", "text": "Find x."},
            ],
        },
        {"role": "assistant", "content": "x is 3."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image/png;base64,BBBB"},
                {"type": "text", "text": "And y?"},
            ],
        },
        {"role": "assistant", "content": "y is 4."},
    ]
    input_ids = _processor_input_ids(tokenizer, messages)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")

    with pytest.raises(ValueError, match="precede the first assistant turn"):
        generator.get_loss_mask_with_multimodal_alignment(messages, input_ids)


def test_text_only_conversation_is_unaffected():
    """A VLM checkpoint with a text-only sample must behave like the plain path."""
    tokenizer = FakeQwen35Tokenizer()
    messages = [
        {"role": "user", "content": "Find x."},
        {"role": "assistant", "content": "x is 3."},
    ]
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_5")
    plain_ids, plain_mask = generator.get_loss_mask(messages)

    token_ids, loss_mask = generator.get_loss_mask_with_multimodal_alignment(messages, plain_ids)

    assert token_ids == plain_ids
    assert loss_mask == plain_mask


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
