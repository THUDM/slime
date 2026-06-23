"""Integration test for the Gemma4 SFT rollout wiring.

Layer above ``test_loss_mask_type_gemma4.py``: that test pins the mask
*function* in isolation; this one exercises ``sft_rollout.generate_rollout``
end-to-end (minus the GPU trainer) with the **real** Gemma4 tokenizer, and
checks the contract the trainer actually consumes:

  - ``sample.tokens``        = the FULL rendered token sequence
  - ``sample.loss_mask``     = only the tail, ``loss_mask[-response_length:]``
  - ``sample.response_length`` = first assistant token .. end of sequence

The subtle, previously-untested assumption is that ``loss_mask`` aligns to
the END of ``tokens`` (the unmasked prefix is implicitly loss-0). If that
slice were off, SFT would silently train on the wrong tokens with no crash.

Requires the real checkpoint tokenizer, so it is skipped when the gemma-4
checkpoint is not present (e.g. laptop CI). Runs on the CPU dev pod.
"""

import os

import pytest

GEMMA4_CKPT = os.environ.get("GEMMA4_CKPT", "/fsx-shopper-intel/dev/jianhfan/gemma-4-31b-it")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(GEMMA4_CKPT, "tokenizer_config.json")),
    reason=f"Gemma4 checkpoint tokenizer not found at {GEMMA4_CKPT}",
)


class _FakeArgs:
    def __init__(self, ckpt, batch_size):
        self.hf_checkpoint = ckpt
        self.loss_mask_type = "gemma4"
        self.rollout_batch_size = batch_size
        self.rollout_global_dataset = True


class _FakeDataBuffer:
    """Mimics data_buffer.get_samples: returns a list of 1-tuples, each
    wrapping a Sample, exactly as sft_rollout unpacks via ``(sample,) = sample``."""

    def __init__(self, samples):
        self._samples = samples

    def get_samples(self, n):
        return [(s,) for s in self._samples[:n]]


def _reset_sft_module_globals():
    # sft_rollout caches TOKENIZER/PROCESSOR/MASK_GENERATOR as module globals;
    # reset so the test controls construction.
    import slime.rollout.sft_rollout as sft

    sft.TOKENIZER = None
    sft.PROCESSOR = None
    sft.MASK_GENERATOR = None
    sft.SAMPLE_PRINTED = False


def _run_rollout(messages_list):
    """Run sft_rollout and return (list[Sample], tokenizer).

    generate_rollout returns the same structure get_samples produced — a
    list of 1-tuples wrapping each (mutated-in-place) Sample — so we unwrap
    the tuples here.
    """
    import slime.rollout.sft_rollout as sft
    from slime.utils.types import Sample

    _reset_sft_module_globals()
    samples = [Sample(prompt=msgs) for msgs in messages_list]
    args = _FakeArgs(GEMMA4_CKPT, batch_size=len(samples))
    buf = _FakeDataBuffer(samples)
    out = sft.generate_rollout(args, rollout_id=0, data_buffer=buf, evaluation=False)
    unwrapped = [item[0] if isinstance(item, tuple) else item for item in out]
    return unwrapped, sft.TOKENIZER


def test_tokens_full_mask_is_tail():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
    ]
    samples, tok = _run_rollout([messages])
    sample = samples[0]

    # tokens is the full rendered sequence; loss_mask is only the tail.
    assert len(sample.tokens) > 0
    assert sample.response_length > 0
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.loss_mask) <= len(sample.tokens)

    # The tail of tokens, masked, must decode to exactly the assistant content
    # (+ <turn|> terminator) — proving the end-alignment assumption holds.
    tail_tokens = sample.tokens[-sample.response_length :]
    masked = [tail_tokens[i] for i in range(len(tail_tokens)) if sample.loss_mask[i] == 1]
    decoded = tok.decode(masked)
    assert "It is 4." in decoded
    assert "<turn|>" in decoded
    # The user/system text must NOT be in the trained span.
    assert "What is 2+2?" not in decoded
    assert "You are helpful." not in decoded


def test_multi_turn_response_length_spans_from_first_assistant():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    samples, tok = _run_rollout([messages])
    sample = samples[0]

    # response_length runs from the FIRST assistant token to the end, so it
    # includes the intervening "Q2" user turn (which stays loss-0 inside it).
    tail_tokens = sample.tokens[-sample.response_length :]
    masked = tok.decode([tail_tokens[i] for i in range(len(tail_tokens)) if sample.loss_mask[i] == 1])
    assert "A1" in masked
    assert "A2" in masked
    # Q2 is inside the tail span but must be masked out.
    assert "Q2" not in masked

    # effective_response_length (a @property) = sum(loss_mask) <=
    # response_length, because the intervening user turn occupies positions
    # that are loss-0.
    assert sample.effective_response_length == sum(sample.loss_mask)
    assert sample.effective_response_length < sample.response_length


def test_batch_of_samples_all_populated():
    convos = [
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}],
        [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye."}],
    ]
    out, _ = _run_rollout(convos)
    assert len(out) == 2
    for sample in out:
        assert len(sample.tokens) > 0
        assert len(sample.loss_mask) == sample.response_length
        assert sample.reward == 0
        assert sum(sample.loss_mask) > 0  # at least some tokens carry loss


def test_loss_mask_never_all_zero():
    """An all-zero mask would make sft_loss zero/nan and silently train on
    nothing — the single most dangerous failure mode for SFT. Guard it."""
    messages = [
        {"role": "user", "content": "Solve x+1=2."},
        {"role": "assistant", "content": "x = 1."},
    ]
    samples, _ = _run_rollout([messages])
    sample = samples[0]
    assert sum(sample.loss_mask) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
