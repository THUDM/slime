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
    def __init__(self, samples):
        self._samples = samples

    def get_samples(self, n):
        return [(s,) for s in self._samples[:n]]


def _reset_sft_module_globals():
    import slime.rollout.sft_rollout as sft

    sft.TOKENIZER = None
    sft.PROCESSOR = None
    sft.MASK_GENERATOR = None
    sft.SAMPLE_PRINTED = False


def _run_rollout(messages_list):
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

    assert len(sample.tokens) > 0
    assert sample.response_length > 0
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.loss_mask) <= len(sample.tokens)

    tail_tokens = sample.tokens[-sample.response_length :]
    masked = [tail_tokens[i] for i in range(len(tail_tokens)) if sample.loss_mask[i] == 1]
    decoded = tok.decode(masked)
    assert "It is 4." in decoded
    assert "<turn|>" in decoded
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

    tail_tokens = sample.tokens[-sample.response_length :]
    masked = tok.decode([tail_tokens[i] for i in range(len(tail_tokens)) if sample.loss_mask[i] == 1])
    assert "A1" in masked
    assert "A2" in masked
    assert "Q2" not in masked

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
        assert sum(sample.loss_mask) > 0


def test_loss_mask_never_all_zero():
    messages = [
        {"role": "user", "content": "Solve x+1=2."},
        {"role": "assistant", "content": "x = 1."},
    ]
    samples, _ = _run_rollout([messages])
    sample = samples[0]
    assert sum(sample.loss_mask) > 0
