from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_materialize_train_pool_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "prepare_runtime_dataset.py"
    spec = importlib.util.spec_from_file_location("materialize_train_pool_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


materialize_train_pool = _load_materialize_train_pool_module()


def test_main_uses_jl_runtime_builders_for_math_and_code(tmp_path: Path, monkeypatch):
    pool_root = tmp_path / "pool"
    cache_dir = tmp_path / "cache"
    dest_list = tmp_path / "train.list"
    stem_src = pool_root / "stem" / "train" / "stem.jsonl"
    stem_src.parent.mkdir(parents=True, exist_ok=True)
    stem_src.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "stem"}],
                "label": "A",
                "metadata": {"domain": "stem"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[list[str], list[str], Path]] = []

    def fake_build_mopd_runtime_train_sources(*, pool_root, cache_dir, include_domains, math_train_datasets, code_train_datasets):
        del pool_root
        outputs: list[Path] = []
        if "math" in include_domains:
            math_dest = cache_dir / "math" / "mopd_math_train.normalized.jsonl"
            calls.append((["deepmath", "bigmath"], [], math_dest))
            math_dest.parent.mkdir(parents=True, exist_ok=True)
            math_dest.write_text('{"prompt":[{"role":"user","content":"math"}],"label":"x","metadata":{"domain":"math"}}\n', encoding="utf-8")
            outputs.append(math_dest)
        if "code" in include_domains:
            code_dest = cache_dir / "code" / "mopd_code_train.normalized.jsonl"
            calls.append(([], ["apps", "taco"], code_dest))
            code_dest.parent.mkdir(parents=True, exist_ok=True)
            code_dest.write_text('{"prompt":[{"role":"user","content":"code"}],"label":"x","metadata":{"domain":"code"}}\n', encoding="utf-8")
            outputs.append(code_dest)
        assert math_train_datasets == "deepmath,bigmath"
        assert code_train_datasets == "apps,taco"
        return outputs

    monkeypatch.setattr(materialize_train_pool, "build_mopd_runtime_train_sources", fake_build_mopd_runtime_train_sources)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_runtime_dataset.py",
            "train",
            "--pool-root",
            str(pool_root),
            "--cache-dir",
            str(cache_dir),
            "--manifest-output",
            str(dest_list),
            "--include-domains",
            "stem,code,math",
            "--exclude-patterns",
            "",
            "--stem-train-datasets",
            "",
            "--math-train-datasets",
            "deepmath,bigmath",
            "--code-train-datasets",
            "apps,taco",
        ],
    )

    materialize_train_pool.main()

    manifest_paths = [Path(line.strip()) for line in dest_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert calls == [
        (["deepmath", "bigmath"], [], cache_dir / "math" / "mopd_math_train.normalized.jsonl"),
        ([], ["apps", "taco"], cache_dir / "code" / "mopd_code_train.normalized.jsonl"),
    ]
    assert manifest_paths == [
        cache_dir / "math" / "mopd_math_train.normalized.jsonl",
        cache_dir / "code" / "mopd_code_train.normalized.jsonl",
        cache_dir / "stem" / "train" / "stem.jsonl",
    ]


def test_main_defaults_stem_and_structured_to_nemotron_datasets(tmp_path: Path, monkeypatch):
    pool_root = tmp_path / "pool"
    cache_dir = tmp_path / "cache"
    dest_list = tmp_path / "train.list"

    stem_keep_paths = [
        pool_root / "stem" / "train" / "nemotron_knowledge_mcqa_data_train-00000-of-00004.jsonl",
        pool_root / "stem" / "train" / "nemotron_knowledge_mcqa_data_train-00001-of-00004.jsonl",
        pool_root / "stem" / "train" / "nemotron_knowledge_mcqa_data_train-00002-of-00004.jsonl",
        pool_root / "stem" / "train" / "nemotron_knowledge_mcqa_data_train-00003-of-00004.jsonl",
    ]
    stem_skip = pool_root / "stem" / "train" / "medmcqa_data_train-00000-of-00001.jsonl"
    structured_keep = pool_root / "structured" / "train" / "nemotron_structured_outputs_structured_outputs_251027_nano_v3_sdg_json_train.jsonl"
    structured_skip = pool_root / "structured" / "train" / "jsonschemabench_train-00000-of-00001.jsonl"
    for path in (*stem_keep_paths, stem_skip, structured_keep, structured_skip):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    def fake_build_mopd_runtime_train_sources(*, pool_root, cache_dir, include_domains, math_train_datasets, code_train_datasets):
        del pool_root, math_train_datasets, code_train_datasets
        outputs: list[Path] = []
        for domain in ("math", "code"):
            if domain not in include_domains:
                continue
            dest = cache_dir / domain / f"mopd_{domain}_train.normalized.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": domain}],
                        "label": "x",
                        "metadata": {"domain": domain},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            outputs.append(dest)
        return outputs

    monkeypatch.setattr(materialize_train_pool, "build_mopd_runtime_train_sources", fake_build_mopd_runtime_train_sources)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_runtime_dataset.py",
            "train",
            "--pool-root",
            str(pool_root),
            "--cache-dir",
            str(cache_dir),
            "--manifest-output",
            str(dest_list),
            "--include-domains",
            "stem,structured,code,math",
            "--exclude-patterns",
            "",
        ],
    )

    materialize_train_pool.main()

    manifest_paths = [Path(line.strip()) for line in dest_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert manifest_paths == [
        cache_dir / "math" / "mopd_math_train.normalized.jsonl",
        cache_dir / "code" / "mopd_code_train.normalized.jsonl",
        *(cache_dir / "stem" / "train" / path.name for path in stem_keep_paths),
        cache_dir / "structured" / "train" / structured_keep.name,
    ]
