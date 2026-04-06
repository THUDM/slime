from __future__ import annotations

import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from common.dataset_selection import resolve_eval_datasets, resolve_train_sources


def test_resolve_train_sources_supports_named_datasets_extras_and_paths(tmp_path: Path) -> None:
    pool_root = tmp_path / "pool"
    named_relpaths = [
        "tool/train/apibench_huggingface_train.jsonl",
        "tool/train/apibench_tensorflow_train.jsonl",
        "tool/train/apibench_torchhub_train.jsonl",
        "tool/train/agent_function_calling_open_dataset_deepnlp_agent_function_call_202510.jsonl",
        "tool/train/agent_function_calling_open_dataset_deepnlp_agent_function_call_202601.jsonl",
    ]
    for relpath in named_relpaths:
        path = pool_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    extra_path = tmp_path / "extra.jsonl"
    extra_path.write_text("{}\n", encoding="utf-8")

    resolved = resolve_train_sources(
        pool_root=pool_root,
        datasets=["apibench"],
        dataset_extras=["agent"],
        paths=[extra_path],
        path_extras=[extra_path],
    )

    assert [path.relative_to(tmp_path).as_posix() for path in resolved] == [
        "pool/tool/train/apibench_huggingface_train.jsonl",
        "pool/tool/train/apibench_tensorflow_train.jsonl",
        "pool/tool/train/apibench_torchhub_train.jsonl",
        "pool/tool/train/agent_function_calling_open_dataset_deepnlp_agent_function_call_202510.jsonl",
        "pool/tool/train/agent_function_calling_open_dataset_deepnlp_agent_function_call_202601.jsonl",
        "extra.jsonl",
    ]


def test_resolve_eval_datasets_supports_profile_defaults_and_runtime_modes(tmp_path: Path) -> None:
    pool_root = tmp_path / "pool"
    eval_relpaths = [
        "structured/eval/ifeval_ifeval_input_data.jsonl",
        "structured/eval/jsonschemabench_test-00000-of-00001.jsonl",
        "structured/eval/ifbench_test_data_train-00000-of-00001.jsonl",
        "stem/eval/mmlu_pro_test-00000-of-00001.jsonl",
        "stem/eval/gpqa_gpqa_main.jsonl",
        "code/livecodebench.jsonl",
    ]
    for relpath in eval_relpaths:
        path = pool_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    resolved = resolve_eval_datasets(
        pool_root=pool_root,
        profile="mdv2",
        dataset_extras=["livecodebench"],
    )

    assert [item.name for item in resolved] == [
        "ifeval",
        "jsonschemabench",
        "ifbench_test",
        "mmlu_pro",
        "gpqa",
        "livecodebench",
    ]
    assert resolved[-1].runtime_mode == "code"
