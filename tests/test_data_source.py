import json
import random
from itertools import chain
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from slime.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer

NUM_GPUS = 0


@pytest.fixture
def source_args(tmp_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")), unk_token="[UNK]"
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("".join(json.dumps({"text": str(i)}) + "\n" for i in range(3)))
    return SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=str(data_path),
        hf_checkpoint=str(tmp_path / "tokenizer"),
        dump_details=None,
        rollout_max_prompt_len=None,
        input_key="text",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=42,
        rollout_shuffle=False,
        n_samples_per_prompt=2,
        buffer_filter_path=None,
        save=str(tmp_path / "checkpoints"),
        load=str(tmp_path / "checkpoints"),
    )


@pytest.mark.parametrize("source_class", [RolloutDataSource, RolloutDataSourceWithBuffer])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("requests", [[8], [2, 8, 1], [3, 1, 2], [0, 2], [6, 3]])
def test_requests_follow_the_epoch_stream(source_args, source_class, shuffle, requests):
    source_args.rollout_shuffle = shuffle
    source = source_class(source_args)
    expected = []
    for epoch in range(sum(requests) // 3 + 1):
        prompts = [str(i) for i in range(3)]
        if shuffle:
            random.Random(source_args.rollout_seed + epoch).shuffle(prompts)
        expected.extend(prompts)

    groups = []
    for count in requests:
        batch = source.get_samples(count)
        assert len(batch) == count
        groups.extend(batch)
        assert [g[0].prompt for g in groups] == expected[: len(groups)]
    assert all(len(g) == source_args.n_samples_per_prompt for g in groups)
    assert [g[0].group_index for g in groups] == list(range(sum(requests)))
    assert [s.index for s in chain.from_iterable(groups)] == list(range(2 * sum(requests)))
    assert all(g[0] is not g[1] for g in groups)


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_after_request_spanning_multiple_epochs(source_args, shuffle):
    source_args.rollout_shuffle = shuffle
    source = RolloutDataSource(source_args)
    assert len(source.get_samples(8)) == 8
    source.save(7)
    expected = source.get_samples(4)
    restored = RolloutDataSource(source_args)
    restored.load(7)
    actual = restored.get_samples(4)
    assert [[(s.prompt, s.group_index, s.index) for s in g] for g in actual] == [
        [(s.prompt, s.group_index, s.index) for s in g] for g in expected
    ]


def test_buffer_is_consumed_before_crossing_epochs(source_args):
    source = RolloutDataSourceWithBuffer(source_args)
    buffered = source.get_samples(1)
    source.add_samples(buffered)
    groups = source.get_samples(9)
    assert len(groups) == 9
    assert groups[0] is buffered[0]
    assert [g[0].prompt for g in groups] == ["0", "1", "2", "0", "1", "2", "0", "1", "2"]


def test_empty_dataset_rejects_a_positive_request(source_args):
    with open(source_args.prompt_data, "w"):
        pass
    source = RolloutDataSource(source_args)
    assert source.get_samples(0) == []
    with pytest.raises(ValueError, match="empty dataset"):
        source.get_samples(1)


def test_source_without_global_dataset(source_args):
    source_args.rollout_global_dataset = False
    groups = RolloutDataSource(source_args).get_samples(8)
    assert len(groups) == 8
    assert all(len(g) == 2 for g in groups)
