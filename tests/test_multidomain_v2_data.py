from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_prepare_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "multidomain_v2" / "prepare_multidomain_v2_data.py"
    spec = importlib.util.spec_from_file_location("prepare_multidomain_v2_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_multidomain_v2 = _load_prepare_module()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_discover_sources_returns_sorted_jsonl_files(tmp_path: Path):
    pool_root = tmp_path / "pool"
    _write_jsonl(pool_root / "structured" / "zeta.jsonl", [{"prompt": [], "label": "3", "metadata": {}}])
    _write_jsonl(pool_root / "tool" / "alpha.jsonl", [{"prompt": [], "label": "1", "metadata": {}}])
    _write_jsonl(pool_root / "math" / "ignored.jsonl", [{"question": "q-ignored", "label": "0"}])
    (pool_root / "code").mkdir(parents=True, exist_ok=True)
    (pool_root / "code" / "ignore.txt").write_text("ignored\n", encoding="utf-8")

    discovered = prepare_multidomain_v2.discover_sources(pool_root)

    assert [path.relative_to(pool_root).as_posix() for path in discovered] == [
        "structured/zeta.jsonl",
        "tool/alpha.jsonl",
    ]


def test_write_dataset_preserves_rows_without_rewriting_fields(tmp_path: Path):
    pool_root = tmp_path / "pool"
    source_a = pool_root / "tool" / "a.jsonl"
    source_b = pool_root / "structured" / "b.jsonl"
    _write_jsonl(
        source_a,
        [
            {
                "prompt": [{"role": "user", "content": "prompt-a"}],
                "label": "label-a",
                "metadata": {"reward_type": "tool_call"},
            }
        ],
    )
    _write_jsonl(
        source_b,
        [
            {
                "prompt": [{"role": "user", "content": "prompt-b"}],
                "label": "label-b",
                "metadata": {"reward_type": "structured_json_schema"},
            }
        ],
    )

    dest = tmp_path / "multidomain_v2_train.jsonl"
    written = prepare_multidomain_v2.write_dataset(
        [source_a, source_b],
        dest,
        skip_samples=1,
        max_samples=1,
    )

    rows = _read_jsonl(dest)

    assert written == 1
    assert len(rows) == 1
    assert rows[0] == {
        "prompt": [{"role": "user", "content": "prompt-b"}],
        "label": "label-b",
        "metadata": {"reward_type": "structured_json_schema"},
    }


def test_write_dataset_keeps_missing_tools_unchanged(tmp_path: Path):
    pool_root = tmp_path / "pool"
    source = pool_root / "stem" / "stem.jsonl"
    _write_jsonl(
        source,
        [
            {
                "prompt": [{"role": "user", "content": "prompt-stem"}],
                "label": "A",
                "metadata": {"reward_type": "stem_mcqa"},
            }
        ],
    )

    dest = tmp_path / "multidomain_v2_train.jsonl"
    written = prepare_multidomain_v2.write_dataset([source], dest)
    rows = _read_jsonl(dest)

    assert written == 1
    assert rows == [
        {
            "prompt": [{"role": "user", "content": "prompt-stem"}],
            "label": "A",
            "metadata": {"reward_type": "stem_mcqa"},
        }
    ]


def test_resolve_named_datasets_expands_requested_pool_sources(tmp_path: Path):
    pool_root = tmp_path / "pool"
    expected_relpaths = [
        "tool/train/toolbench_v1_data_train-00000-of-00004.jsonl",
        "tool/train/toolbench_v1_data_train-00001-of-00004.jsonl",
        "tool/train/toolbench_v1_data_train-00002-of-00004.jsonl",
        "tool/train/toolbench_v1_data_train-00003-of-00004.jsonl",
        "structured/train/jsonschemabench_data_train-00000-of-00001.jsonl",
        "structured/train/nemotron_structured_outputs_structured_outputs_251027_nano_v3_sdg_json_train.jsonl",
    ]
    for relpath in expected_relpaths:
        _write_jsonl(
            pool_root / relpath,
            [{"prompt": [{"role": "user", "content": relpath}], "label": "", "metadata": {}}],
        )

    resolved = prepare_multidomain_v2.resolve_named_datasets(
        pool_root,
        ["toolbench_v1", "jsonschemabench", "nemotron_structured_outputs"],
    )

    assert [path.relative_to(pool_root).as_posix() for path in resolved] == expected_relpaths


def test_resolve_nemotron_knowledge_mcqa_expands_all_train_shards(tmp_path: Path):
    pool_root = tmp_path / "pool"
    expected_relpaths = [
        "stem/train/nemotron_knowledge_mcqa_data_train-00000-of-00004.jsonl",
        "stem/train/nemotron_knowledge_mcqa_data_train-00001-of-00004.jsonl",
        "stem/train/nemotron_knowledge_mcqa_data_train-00002-of-00004.jsonl",
        "stem/train/nemotron_knowledge_mcqa_data_train-00003-of-00004.jsonl",
    ]
    for relpath in expected_relpaths:
        _write_jsonl(
            pool_root / relpath,
            [{"prompt": [{"role": "user", "content": relpath}], "label": "", "metadata": {}}],
        )

    resolved = prepare_multidomain_v2.resolve_named_datasets(pool_root, ["nemotron_knowledge_mcqa"])

    assert [path.relative_to(pool_root).as_posix() for path in resolved] == expected_relpaths
