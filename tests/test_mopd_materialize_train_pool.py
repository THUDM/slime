from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_materialize_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "MOPD" / "materialize_train_pool.py"
    spec = importlib.util.spec_from_file_location("materialize_train_pool_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


materialize_train_pool = _load_materialize_module()


def test_needs_materialize_skips_invalid_leading_lines(tmp_path: Path):
    src = tmp_path / "tool" / "train" / "sample.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "\n".join(
            [
                "not-json",
                '{"prompt":[{"role":"user","content":"hi"}],"supervision_family":"function_call_single"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert materialize_train_pool._file_needs_materialize(src) is True
