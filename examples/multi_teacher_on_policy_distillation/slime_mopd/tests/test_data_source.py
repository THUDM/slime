from types import SimpleNamespace

import pytest

from slime_mopd.config import MopdConfig, clear_config_cache

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _config(**overrides):
    raw = {
        "teachers": [
            {"id": "math", "rm_url": "http://127.0.0.1:30001"},
            {"id": "code", "rm_url": "http://127.0.0.1:30002"},
            {"id": "if", "rm_url": "http://127.0.0.1:30003"},
        ]
    }
    raw.update(overrides)
    return MopdConfig.from_raw(raw)


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def _write_parquet(path, rows):
    table = pa.table(
        {
            "prompt": [[{"role": m[0], "content": m[1]} for m in row["prompt"]] for row in rows],
            "domain": [row["domain"] for row in rows],
            "data_source": [row.get("data_source") for row in rows],
            "answer": [row.get("answer") for row in rows],
            "reward_model": [{"style": "rule", "ground_truth": row.get("answer")} for row in rows],
            "extra_info": [{"dataset": row.get("dataset"), "index": str(i)} for i, row in enumerate(rows)],
        }
    )
    pq.write_table(table, path)


def _args(**overrides):
    args = SimpleNamespace(
        prompt_data=None,
        buffer_filter_path=None,
        dump_details=None,
        hf_checkpoint="x",
        rollout_max_prompt_len=None,
        input_key="prompt",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key="tools",
        apply_chat_template=True,
        apply_chat_template_kwargs={},
        rollout_seed=42,
        rollout_shuffle=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _rows():
    return [
        {
            "prompt": [("system", "s"), ("user", "1+1?")],
            "domain": "math",
            "data_source": "math_dapo_boxed",
            "answer": "2",
        },
        {"prompt": [("system", "s"), ("user", "rev")], "domain": "code", "data_source": "taco"},
        {"prompt": [("system", "s"), ("user", "two")], "domain": "if", "data_source": "nemotron_if_rl"},
    ]


def test_build_samples_stamps_domain_and_metadata(tmp_path):
    import slime_mopd.data_source as ds

    path = tmp_path / "train.parquet"
    _write_parquet(path, _rows())

    samples, counts = ds._build_samples(_args(), str(path), _config(), _FakeTokenizer(), None)

    assert counts == {"math": 1, "code": 1, "if": 1}
    assert [sample.source for sample in samples] == ["math", "code", "if"]
    assert [sample.metadata["mopd_domain"] for sample in samples] == ["math", "code", "if"]
    assert [sample.metadata["data_source"] for sample in samples] == [
        "math_dapo_boxed",
        "taco",
        "nemotron_if_rl",
    ]
    assert samples[0].prompt == "<system>s</system><user>1+1?</user>"
    assert samples[0].label is None


def test_build_samples_reads_label_key(tmp_path):
    import slime_mopd.data_source as ds

    path = tmp_path / "eval.parquet"
    _write_parquet(path, _rows())

    samples, _ = ds._build_samples(_args(label_key="answer"), str(path), _config(), _FakeTokenizer(), None)

    assert samples[0].label == "2"
    assert samples[1].label is None


def test_build_samples_reports_reward_style(tmp_path):
    import slime_mopd.data_source as ds

    path = tmp_path / "train.parquet"
    _write_parquet(path, _rows())

    samples, _ = ds._build_samples(_args(), str(path), _config(), _FakeTokenizer(), None)

    assert samples[0].metadata["reward_style"] == "rule"
    # ``extra_info.dataset`` is only kept when the row actually carries one.
    assert "dataset" not in samples[0].metadata


def test_build_samples_rejects_domain_without_teacher(tmp_path):
    import slime_mopd.data_source as ds

    path = tmp_path / "train.parquet"
    _write_parquet(path, _rows())
    config = MopdConfig.from_raw({"teachers": [{"id": "math", "rm_url": "http://127.0.0.1:1"}]})

    with pytest.raises(ValueError, match=r"no teacher for domain\(s\) \['code', 'if'\]"):
        ds._build_samples(_args(), str(path), config, _FakeTokenizer(), None)


def test_build_samples_allows_missing_domain_with_default_teacher(tmp_path):
    import slime_mopd.data_source as ds

    path = tmp_path / "train.parquet"
    _write_parquet(path, _rows())
    config = MopdConfig.from_raw(
        {
            "teachers": [{"id": "math", "rm_url": "http://127.0.0.1:1"}],
            "default_teacher": "math",
        }
    )

    samples, counts = ds._build_samples(_args(), str(path), config, _FakeTokenizer(), None)

    assert len(samples) == 3
    assert counts["code"] == 1


def test_build_samples_keeps_message_list_without_chat_template(tmp_path):
    import slime_mopd.data_source as ds

    path = tmp_path / "train.parquet"
    _write_parquet(path, _rows())

    samples, _ = ds._build_samples(_args(apply_chat_template=False), str(path), _config(), _FakeTokenizer(), None)

    assert isinstance(samples[0].prompt, list)
    assert samples[0].prompt[0]["role"] == "system"


def test_sample_list_dataset_shuffle_is_seeded():
    import slime_mopd.data_source as ds

    dataset = ds._SampleListDataset(list(range(6)), seed=1, shuffle=True)
    assert len(dataset) == 6
    dataset.shuffle(0)
    first = list(dataset.samples)
    assert sorted(first) == list(range(6))
    dataset.shuffle(0)  # same epoch is a no-op
    assert dataset.samples == first

    other = ds._SampleListDataset(list(range(6)), seed=1, shuffle=True)
    other.shuffle(0)
    assert other.samples == first


def test_data_source_falls_back_without_mopd_config(monkeypatch):
    clear_config_cache()
    import slime_mopd.data_source as ds

    args = _args()
    data_source = ds.MopdRolloutDataSource(args)

    assert data_source.dataset is None
    assert data_source.buffer == []


def test_data_source_replaces_dataset_and_restores_prompt_data(tmp_path, monkeypatch):
    clear_config_cache()
    import slime_mopd.data_source as ds

    path = tmp_path / "train.parquet"
    _write_parquet(path, _rows())

    captured = {}

    def _fake_base_init(self, args):
        captured["prompt_data"] = args.prompt_data
        self.args = args
        self.buffer = []
        self.dataset = None
        self.epoch_id = 0

    monkeypatch.setattr(ds.RolloutDataSourceWithBuffer, "__init__", _fake_base_init)
    monkeypatch.setattr(ds, "load_tokenizer", lambda *a, **k: _FakeTokenizer())
    monkeypatch.setattr(ds, "load_processor", lambda *a, **k: None)

    args = _args(
        prompt_data=str(path),
        mopd={
            "teachers": [
                {"id": "math", "rm_url": "http://127.0.0.1:1"},
                {"id": "code", "rm_url": "http://127.0.0.1:2"},
                {"id": "if", "rm_url": "http://127.0.0.1:3"},
            ]
        },
    )
    data_source = ds.MopdRolloutDataSource(args)

    # The base class must not parse the file a second time...
    assert captured["prompt_data"] is None
    # ...and the attribute is restored afterwards.
    assert args.prompt_data == str(path)
    assert len(data_source.dataset) == 3
    assert [s.source for s in data_source.dataset.samples] == ["math", "code", "if"]
