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

import asyncio
import json
import sys
import types

import pytest

import slime.rollout.sglang_rollout as sglang_rollout
from slime.rollout.data_source import _should_cache_prompt_token_ids
from slime.rollout.sglang_rollout import _prepare_prompt_ids
from slime.utils.data import Dataset, filter_long_prompt
from slime.utils.eval_config import EvalDatasetConfig
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

    def __init__(self):
        self.encode_calls = 0

    def __call__(self, prompts, add_special_tokens=False):
        return {"input_ids": [list(range(_encoded_length(p))) for p in prompts]}

    def encode(self, prompt, add_special_tokens=False):
        self.encode_calls += 1
        return list(range(_encoded_length(prompt)))


class _Processor:
    """Per-sample processor stand-in, same length convention."""

    def __init__(self):
        self.calls = 0

    def __call__(self, text=None, **kwargs):
        self.calls += 1
        return {"input_ids": [list(range(_encoded_length(text)))]}


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
    assert all(sample.pop_cached_prompt_token_ids() is None for sample in kept)


@pytest.mark.unit
def test_list_prompts_do_not_use_the_text_token_cache():
    class ConversationTokenizer:
        def __call__(self, prompts, add_special_tokens=False):
            assert prompts == [[{"role": "user", "content": "hello"}]]
            return {"input_ids": [[1, 2]]}

    samples = [Sample(prompt=[{"role": "user", "content": "hello"}])]

    kept = filter_long_prompt(
        samples,
        ConversationTokenizer(),
        None,
        max_length=100,
        cache_prompt_token_ids=True,
    )

    assert kept == samples
    assert kept[0].pop_cached_prompt_token_ids() is None


@pytest.mark.unit
def test_multimodal_inputs_do_not_use_cache_without_a_processor():
    samples = _make_samples([(True, 3)])

    kept = filter_long_prompt(
        samples,
        _Tokenizer(),
        None,
        max_length=100,
        cache_prompt_token_ids=True,
    )

    assert kept == samples
    assert kept[0].pop_cached_prompt_token_ids() is None


@pytest.mark.unit
def test_no_length_filter_does_not_create_a_cache():
    samples = _make_samples([(False, 3)])

    kept = filter_long_prompt(
        samples,
        _Tokenizer(),
        None,
        max_length=None,
        cache_prompt_token_ids=True,
    )

    assert kept == samples
    assert kept[0].pop_cached_prompt_token_ids() is None


@pytest.mark.unit
def test_caches_only_retained_text_prompt_ids(stub_process_vision_info):
    samples = _make_samples([(False, 3), (True, 4), (False, 500)])

    kept = filter_long_prompt(
        samples,
        _Tokenizer(),
        _Processor(),
        max_length=100,
        cache_prompt_token_ids=True,
    )

    assert [sample.prompt for sample in kept] == ["p0:3", "p1:4"]
    assert kept[0].pop_cached_prompt_token_ids() == [0, 1, 2]
    assert kept[1].pop_cached_prompt_token_ids() is None
    assert samples[2].pop_cached_prompt_token_ids() is None


@pytest.mark.unit
def test_invalid_token_ids_skip_cache_without_dropping_sample():
    class InvalidTokenizer:
        def __call__(self, prompts, add_special_tokens=False):
            return {"input_ids": [[-1] for _ in prompts]}

    samples = [Sample(prompt="prompt")]
    kept = filter_long_prompt(
        samples,
        InvalidTokenizer(),
        None,
        max_length=10,
        cache_prompt_token_ids=True,
    )

    assert kept == samples
    assert kept[0].pop_cached_prompt_token_ids() is None


@pytest.mark.unit
def test_prepare_prompt_ids_consumes_cache_without_encoding():
    sample = Sample(prompt="p0:3")
    assert sample.cache_prompt_token_ids([7, 8, 9])
    tokenizer = _Tokenizer()

    assert _prepare_prompt_ids(sample, tokenizer, None) == [7, 8, 9]
    assert tokenizer.encode_calls == 0
    assert _prepare_prompt_ids(sample, tokenizer, None) == [0, 1, 2]
    assert tokenizer.encode_calls == 1


@pytest.mark.unit
def test_prepare_prompt_ids_uses_an_empty_cached_prompt_without_encoding():
    sample = Sample(prompt="p0:0")
    assert sample.cache_prompt_token_ids([])
    tokenizer = _Tokenizer()

    assert _prepare_prompt_ids(sample, tokenizer, None) == []
    assert tokenizer.encode_calls == 0


@pytest.mark.unit
def test_dataset_caches_ids_for_the_rendered_chat_template(tmp_path):
    class TemplateTokenizer(_Tokenizer):
        def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt, **kwargs):
            assert messages == [{"role": "user", "content": "hello"}]
            assert tools == [{"type": "function", "function": {"name": "lookup"}}]
            assert not tokenize
            assert add_generation_prompt
            return "p0:3"

    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "hello",
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = TemplateTokenizer()

    dataset = Dataset(
        str(path),
        tokenizer,
        processor=None,
        max_length=10,
        tool_key="tools",
        apply_chat_template=True,
        cache_prompt_token_ids=True,
    )

    assert dataset[0].prompt == "p0:3"
    assert _prepare_prompt_ids(dataset[0], tokenizer, None) == [0, 1, 2]
    assert tokenizer.encode_calls == 0


@pytest.mark.unit
def test_prepare_prompt_ids_preserves_existing_token_precedence():
    sample = Sample(prompt="p0:3", tokens=[20, 21])
    assert sample.cache_prompt_token_ids([7, 8, 9])
    tokenizer = _Tokenizer()

    assert _prepare_prompt_ids(sample, tokenizer, None) is sample.tokens
    assert tokenizer.encode_calls == 0
    assert sample.pop_cached_prompt_token_ids() is None


@pytest.mark.unit
def test_prepare_prompt_ids_preserves_multimodal_processor_precedence():
    sample = Sample(prompt="p0:3", tokens=[20, 21], multimodal_inputs={"images": ["img"]})
    assert sample.cache_prompt_token_ids([7, 8, 9])
    tokenizer = _Tokenizer()
    processor = _Processor()

    assert _prepare_prompt_ids(sample, tokenizer, processor) == [0, 1, 2]
    assert processor.calls == 1
    assert tokenizer.encode_calls == 0
    assert sample.pop_cached_prompt_token_ids() is None


@pytest.mark.unit
def test_processed_multimodal_inputs_allow_existing_tokens_to_win():
    sample = Sample(
        prompt="p0:3",
        tokens=[20, 21],
        multimodal_inputs={"images": ["img"]},
        multimodal_train_inputs={"pixel_values": "processed"},
    )
    assert sample.cache_prompt_token_ids([7, 8, 9])
    tokenizer = _Tokenizer()
    processor = _Processor()

    assert _prepare_prompt_ids(sample, tokenizer, processor) is sample.tokens
    assert processor.calls == 0
    assert tokenizer.encode_calls == 0
    assert sample.pop_cached_prompt_token_ids() is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "override",
    [
        {},
        {"data_source_path": "slime.rollout.data_source.RolloutDataSource"},
        {"rollout_function_path": "custom.rollout"},
        {"custom_generate_function_path": "custom.generate"},
        {"data_source_path": "custom.DataSource"},
    ],
)
def test_train_cache_is_enabled_only_for_the_default_rollout_contract(override):
    values = {
        "rollout_function_path": "slime.rollout.sglang_rollout.generate_rollout",
        "custom_generate_function_path": None,
        "data_source_path": "slime.rollout.data_source.RolloutDataSourceWithBuffer",
    }
    values.update(override)
    args = types.SimpleNamespace(**values)

    expected = not override or override == {"data_source_path": "slime.rollout.data_source.RolloutDataSource"}
    assert _should_cache_prompt_token_ids(args) is expected


@pytest.mark.unit
def test_eval_dataset_cache_key_separates_default_and_custom_generate(monkeypatch):
    dataset_calls = []

    class EmptyDataset:
        def __init__(self, **kwargs):
            dataset_calls.append(kwargs)
            self.samples = []

    monkeypatch.setattr(sglang_rollout, "Dataset", EmptyDataset)
    monkeypatch.setattr(sglang_rollout, "EVAL_PROMPT_DATASET", {})
    monkeypatch.setattr(sglang_rollout, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(sglang_rollout, "load_processor", lambda *args, **kwargs: None)

    args = types.SimpleNamespace(
        group_rm=False,
        multimodal_keys=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        hf_checkpoint="checkpoint",
        eval_max_prompt_len=128,
        custom_generate_function_path=None,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        eval_min_new_tokens=None,
        sglang_enable_deterministic_inference=False,
        eval_reward_key=None,
        reward_key=None,
    )
    dataset_cfg = EvalDatasetConfig(
        name="eval",
        path="unused.jsonl",
        input_key="text",
        metadata_key="metadata",
        n_samples_per_eval_prompt=1,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_response_len=16,
    )

    asyncio.run(sglang_rollout.eval_rollout_single_dataset(args, 0, dataset_cfg))
    args.custom_generate_function_path = "custom.generate"
    asyncio.run(sglang_rollout.eval_rollout_single_dataset(args, 0, dataset_cfg))
    sglang_rollout.EVAL_PROMPT_DATASET.clear()
    args.custom_generate_function_path = None
    dataset_cfg.custom_generate_function_path = "dataset.generate"
    asyncio.run(sglang_rollout.eval_rollout_single_dataset(args, 0, dataset_cfg))

    assert [call["cache_prompt_token_ids"] for call in dataset_calls] == [True, False, False]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
