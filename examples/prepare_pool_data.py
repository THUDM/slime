from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


AVALANCHE_ROOT = Path(__file__).resolve().parents[3]
OPEN_DATA_ROOT = AVALANCHE_ROOT / "data" / "open_data"
POOL_ROOT = AVALANCHE_ROOT / "data" / "pool"

JSONSCHEMABENCH_CONFIGS = (
    "Github_easy",
    "Github_hard",
    "Github_medium",
    "Github_trivial",
    "Github_ultra",
    "Glaiveai2K",
    "JsonSchemaStore",
    "Kubernetes",
    "Snowplow",
    "WashingtonPost",
)
JSONSCHEMABENCH_SPLITS = ("train", "val", "test")


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "apibench_huggingface": {
        "source": OPEN_DATA_ROOT / "tool_call" / "apibench" / "huggingface_train.json",
        "output": POOL_ROOT / "tool" / "train" / "apibench_huggingface_train.jsonl",
        "supervision_family": "function_call_single",
        "reader": "jsonl",
    },
    "apibench_tensorflow": {
        "source": OPEN_DATA_ROOT / "tool_call" / "apibench" / "tensorflow_train.json",
        "output": POOL_ROOT / "tool" / "train" / "apibench_tensorflow_train.jsonl",
        "supervision_family": "function_call_single",
        "reader": "jsonl",
    },
    "apibench_torchhub": {
        "source": OPEN_DATA_ROOT / "tool_call" / "apibench" / "torchhub_train.json",
        "output": POOL_ROOT / "tool" / "train" / "apibench_torchhub_train.jsonl",
        "supervision_family": "function_call_single",
        "reader": "jsonl",
    },
    "xlam_function_calling_60k": {
        "source": OPEN_DATA_ROOT / "tool_call" / "xlam_function_calling_60k" / "xlam-function-calling-60k.parquet",
        "output": POOL_ROOT / "tool" / "train" / "xlam_function_calling_60k_xlam-function-calling-60k.jsonl",
        "supervision_family": "function_call_single",
        "reader": "parquet",
    },
    "agent_function_calling_open_dataset_deepnlp_agent_function_call_202510": {
        "source": OPEN_DATA_ROOT / "tool_call" / "agent_function_calling_open_dataset" / "deepnlp_agent_function_call_202510.json",
        "output": POOL_ROOT / "tool" / "train" / "agent_function_calling_open_dataset_deepnlp_agent_function_call_202510.jsonl",
        "supervision_family": "function_call_single",
        "reader": "jsonl",
    },
    "agent_function_calling_open_dataset_deepnlp_agent_function_call_202601": {
        "source": OPEN_DATA_ROOT / "tool_call" / "agent_function_calling_open_dataset" / "deepnlp_agent_function_call_202601.json",
        "output": POOL_ROOT / "tool" / "train" / "agent_function_calling_open_dataset_deepnlp_agent_function_call_202601.jsonl",
        "supervision_family": "function_call_single",
        "reader": "jsonl",
    },
    "bfcl_v3": {
        "source": OPEN_DATA_ROOT / "tool_call" / "bfcl_v3" / "data" / "train-00000-of-00001.parquet",
        "output": POOL_ROOT / "tool" / "eval" / "bfcl_v3_train-00000-of-00001.jsonl",
        "reader": "parquet",
    },
    "bfcl_v3_multi_turn_base": {
        "source": OPEN_DATA_ROOT / "tool_call" / "bfcl_v3_multi_turn_base" / "data" / "train-00000-of-00001.parquet",
        "output": POOL_ROOT / "tool" / "eval" / "bfcl_v3_multi_turn_base_train-00000-of-00001.jsonl",
        "reader": "parquet",
    },
    "gpqa_main": {
        "source": OPEN_DATA_ROOT / "stem" / "gpqa" / "gpqa_main.csv",
        "output": POOL_ROOT / "stem" / "eval" / "gpqa_gpqa_main.jsonl",
        "reader": "csv",
    },
    "mmlu_pro_test": {
        "source": OPEN_DATA_ROOT / "stem" / "mmlu_pro" / "data" / "test-00000-of-00001.parquet",
        "output": POOL_ROOT / "stem" / "eval" / "mmlu_pro_test-00000-of-00001.jsonl",
        "reader": "parquet",
    },
    "medmcqa_train": {
        "source": OPEN_DATA_ROOT / "stem" / "medmcqa" / "data" / "train-00000-of-00001.parquet",
        "output": POOL_ROOT / "stem" / "train" / "medmcqa_data_train-00000-of-00001.jsonl",
        "reader": "parquet",
    },
    "nemotron_knowledge_mcqa_train_00000": {
        "source": OPEN_DATA_ROOT / "stem" / "nemotron_knowledge_mcqa" / "data" / "train-00000-of-00004.parquet",
        "output": POOL_ROOT / "stem" / "train" / "nemotron_knowledge_mcqa_data_train-00000-of-00004.jsonl",
        "reader": "parquet",
    },
    "nemotron_knowledge_mcqa_train_00001": {
        "source": OPEN_DATA_ROOT / "stem" / "nemotron_knowledge_mcqa" / "data" / "train-00001-of-00004.parquet",
        "output": POOL_ROOT / "stem" / "train" / "nemotron_knowledge_mcqa_data_train-00001-of-00004.jsonl",
        "reader": "parquet",
    },
    "nemotron_knowledge_mcqa_train_00002": {
        "source": OPEN_DATA_ROOT / "stem" / "nemotron_knowledge_mcqa" / "data" / "train-00002-of-00004.parquet",
        "output": POOL_ROOT / "stem" / "train" / "nemotron_knowledge_mcqa_data_train-00002-of-00004.jsonl",
        "reader": "parquet",
    },
    "nemotron_knowledge_mcqa_train_00003": {
        "source": OPEN_DATA_ROOT / "stem" / "nemotron_knowledge_mcqa" / "data" / "train-00003-of-00004.parquet",
        "output": POOL_ROOT / "stem" / "train" / "nemotron_knowledge_mcqa_data_train-00003-of-00004.jsonl",
        "reader": "parquet",
    },
    "ifeval": {
        "source": OPEN_DATA_ROOT / "structured_output" / "ifeval" / "ifeval_input_data.jsonl",
        "output": POOL_ROOT / "structured" / "eval" / "ifeval_ifeval_input_data.jsonl",
        "reader": "jsonl",
    },
    "ifbench_test": {
        "source": OPEN_DATA_ROOT / "structured_output" / "ifbench_test" / "data" / "train-00000-of-00001.parquet",
        "output": POOL_ROOT / "structured" / "eval" / "ifbench_test_data_train-00000-of-00001.jsonl",
        "reader": "parquet",
    },
    "jsonschemabench_train": {
        "source": OPEN_DATA_ROOT / "structured_output" / "jsonschemabench" / "data" / "train-00000-of-00001.parquet",
        "output": POOL_ROOT / "structured" / "train" / "jsonschemabench_train-00000-of-00001.jsonl",
        "reader": "parquet",
    },
    "jsonschemabench_test": {
        "source": OPEN_DATA_ROOT / "structured_output" / "jsonschemabench" / "data" / "test-00000-of-00001.parquet",
        "output": POOL_ROOT / "structured" / "eval" / "jsonschemabench_test-00000-of-00001.jsonl",
        "reader": "parquet",
    },
    "nemotron_structured_outputs": {
        "source": OPEN_DATA_ROOT / "structured_output" / "nemotron_structured_outputs" / "structured_outputs_251027_nano_v3_sdg_json_train.jsonl",
        "output": POOL_ROOT / "structured" / "train" / "nemotron_structured_outputs_structured_outputs_251027_nano_v3_sdg_json_train.jsonl",
        "reader": "jsonl",
    },
}


for split in JSONSCHEMABENCH_SPLITS:
    DATASET_SPECS[f"jsonschemabench_{split}"] = {
        "source": OPEN_DATA_ROOT / "structured_output" / "jsonschemabench" / "data" / f"{split}-00000-of-00001.parquet",
        "output": POOL_ROOT / "structured" / ("train" if split == "train" else "eval") / f"jsonschemabench_{split}-00000-of-00001.jsonl",
        "reader": "parquet",
    }

for config_name in JSONSCHEMABENCH_CONFIGS:
    config_key = config_name.lower()
    for split in JSONSCHEMABENCH_SPLITS:
        DATASET_SPECS[f"jsonschemabench_{config_key}_{split}"] = {
            "source": OPEN_DATA_ROOT / "structured_output" / "jsonschemabench" / config_name / f"{split}-00000-of-00001.parquet",
            "output": POOL_ROOT / "structured" / ("train" if split == "train" else "eval") / f"jsonschemabench_{config_name}_{split}-00000-of-00001.jsonl",
            "reader": "parquet",
        }

DATASET_SPECS.update(
    {
        "ai2_arc_arc_challenge_train": {
            "source": OPEN_DATA_ROOT / "stem" / "ai2_arc" / "ARC-Challenge" / "train-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "train" / "ai2_arc_ARC-Challenge_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "ai2_arc_arc_challenge_test": {
            "source": OPEN_DATA_ROOT / "stem" / "ai2_arc" / "ARC-Challenge" / "test-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "ai2_arc_ARC-Challenge_test-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "ai2_arc_arc_challenge_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "ai2_arc" / "ARC-Challenge" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "ai2_arc_ARC-Challenge_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "ai2_arc_arc_easy_train": {
            "source": OPEN_DATA_ROOT / "stem" / "ai2_arc" / "ARC-Easy" / "train-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "train" / "ai2_arc_ARC-Easy_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "ai2_arc_arc_easy_test": {
            "source": OPEN_DATA_ROOT / "stem" / "ai2_arc" / "ARC-Easy" / "test-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "ai2_arc_ARC-Easy_test-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "ai2_arc_arc_easy_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "ai2_arc" / "ARC-Easy" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "ai2_arc_ARC-Easy_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "ai2_arc_train": {
            "source": [
                OPEN_DATA_ROOT / "stem" / "ai2_arc" / "ARC-Challenge" / "train-00000-of-00001.parquet",
                OPEN_DATA_ROOT / "stem" / "ai2_arc" / "ARC-Easy" / "train-00000-of-00001.parquet",
            ],
            "output": POOL_ROOT / "stem" / "train" / "ai2_arc_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "aqua_rat_train": {
            "source": OPEN_DATA_ROOT / "stem" / "aqua_rat" / "raw" / "train-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "train" / "aqua_rat_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "aqua_rat_test": {
            "source": OPEN_DATA_ROOT / "stem" / "aqua_rat" / "raw" / "test-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "aqua_rat_test-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "aqua_rat_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "aqua_rat" / "raw" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "aqua_rat_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "mmlu_auxiliary_train": {
            "source": OPEN_DATA_ROOT / "stem" / "mmlu" / "auxiliary_train" / "train-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "train" / "mmlu_auxiliary_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "mmlu_dev": {
            "source": OPEN_DATA_ROOT / "stem" / "mmlu" / "all" / "dev-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "mmlu_dev-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "mmlu_test": {
            "source": OPEN_DATA_ROOT / "stem" / "mmlu" / "all" / "test-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "mmlu_test-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "mmlu_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "mmlu" / "all" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "mmlu_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "mmlu_pro_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "mmlu_pro" / "data" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "mmlu_pro_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "mmlu_pro_data_test": {
            "source": OPEN_DATA_ROOT / "stem" / "mmlu_pro" / "data" / "test-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "mmlu_pro_data_test-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "mmlu_pro_data_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "mmlu_pro" / "data" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "mmlu_pro_data_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "medmcqa_test": {
            "source": OPEN_DATA_ROOT / "stem" / "medmcqa" / "data" / "test-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "medmcqa_data_test-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "medmcqa_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "medmcqa" / "data" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "medmcqa_data_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "medmcqa_train_alias": {
            "source": OPEN_DATA_ROOT / "stem" / "medmcqa" / "data" / "train-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "train" / "medmcqa_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "openbookqa_main_train": {
            "source": OPEN_DATA_ROOT / "stem" / "openbookqa" / "main" / "train-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "train" / "openbookqa_main_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "openbookqa_main_test": {
            "source": OPEN_DATA_ROOT / "stem" / "openbookqa" / "main" / "test-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "openbookqa_main_test-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "openbookqa_main_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "openbookqa" / "main" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "openbookqa_main_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "openbookqa_additional_train": {
            "source": OPEN_DATA_ROOT / "stem" / "openbookqa" / "additional" / "train-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "train" / "openbookqa_additional_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "openbookqa_additional_test": {
            "source": OPEN_DATA_ROOT / "stem" / "openbookqa" / "additional" / "test-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "openbookqa_additional_test-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "openbookqa_additional_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "openbookqa" / "additional" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "openbookqa_additional_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "openbookqa_train": {
            "source": [
                OPEN_DATA_ROOT / "stem" / "openbookqa" / "main" / "train-00000-of-00001.parquet",
                OPEN_DATA_ROOT / "stem" / "openbookqa" / "additional" / "train-00000-of-00001.parquet",
            ],
            "output": POOL_ROOT / "stem" / "train" / "openbookqa_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "scienceqa_train": {
            "source": OPEN_DATA_ROOT / "stem" / "scienceqa" / "data" / "train-00000-of-00001-1028f23e353fbe3e.parquet",
            "output": POOL_ROOT / "stem" / "train" / "scienceqa_train-00000-of-00001-1028f23e353fbe3e.jsonl",
            "reader": "parquet",
        },
        "scienceqa_test": {
            "source": OPEN_DATA_ROOT / "stem" / "scienceqa" / "data" / "test-00000-of-00001-f0e719df791966ff.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "scienceqa_data_test-00000-of-00001-f0e719df791966ff.jsonl",
            "reader": "parquet",
        },
        "scienceqa_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "scienceqa" / "data" / "validation-00000-of-00001-6c7328ff6c84284c.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "scienceqa_data_validation-00000-of-00001-6c7328ff6c84284c.jsonl",
            "reader": "parquet",
        },
        "scienceqa_train_alias": {
            "source": OPEN_DATA_ROOT / "stem" / "scienceqa" / "data" / "train-00000-of-00001-1028f23e353fbe3e.parquet",
            "output": POOL_ROOT / "stem" / "train" / "scienceqa_data_train-00000-of-00001-1028f23e353fbe3e.jsonl",
            "reader": "parquet",
        },
        "sciq_train": {
            "source": OPEN_DATA_ROOT / "stem" / "sciq" / "data" / "train-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "train" / "sciq_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "sciq_test": {
            "source": OPEN_DATA_ROOT / "stem" / "sciq" / "data" / "test-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "sciq_data_test-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "sciq_validation": {
            "source": OPEN_DATA_ROOT / "stem" / "sciq" / "data" / "validation-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "eval" / "sciq_data_validation-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "sciq_train_alias": {
            "source": OPEN_DATA_ROOT / "stem" / "sciq" / "data" / "train-00000-of-00001.parquet",
            "output": POOL_ROOT / "stem" / "train" / "sciq_data_train-00000-of-00001.jsonl",
            "reader": "parquet",
        },
        "agieval_dev": {
            "source": OPEN_DATA_ROOT / "stem" / "agieval" / "dev",
            "output": POOL_ROOT / "stem" / "eval" / "agieval_dev.jsonl",
            "reader": "jsonl_dir",
        },
        "agieval_test": {
            "source": OPEN_DATA_ROOT / "stem" / "agieval" / "test",
            "output": POOL_ROOT / "stem" / "eval" / "agieval_test.jsonl",
            "reader": "jsonl_dir",
        },
    }
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return value


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _base_metadata(
    dataset_name: str,
    domain: str,
    record_id: Any,
    reward_type: str | None = None,
    supervision_family: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "dataset_name": dataset_name,
        "domain": domain,
        "record_id": record_id,
    }
    if reward_type:
        metadata["reward_type"] = reward_type
    if supervision_family:
        metadata["supervision_family"] = supervision_family
    metadata.update({key: value for key, value in extra.items() if value not in (None, "", [], {})})
    return metadata


def _stable_record_id(prefix: str, payload: Any) -> str:
    raw = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True)
    return f"{prefix}:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _tool_metadata(dataset_name: str, record_id: Any, supervision_family: str, **extra: Any) -> dict[str, Any]:
    return _base_metadata(dataset_name, "tool", record_id, supervision_family=supervision_family, **extra)


def _ensure_message_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        messages: list[dict[str, Any]] = []
        for item in prompt:
            if isinstance(item, dict):
                messages.append(_json_ready(item))
        return messages
    text = "" if prompt is None else str(prompt).strip()
    return [{"role": "user", "content": text}] if text else []


def _stable_shuffle(options: list[str], key: str) -> list[str]:
    shuffled = list(options)
    random.Random(key).shuffle(shuffled)
    return shuffled


def _format_mcqa_prompt(question: str, options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [question.strip(), "", "Options:"]
    for idx, option in enumerate(options):
        lines.append(f"{letters[idx]}. {option}")
    return "\n".join(lines).strip()


def _answer_index_to_letter(index: Any) -> str:
    try:
        numeric = int(index)
    except (TypeError, ValueError):
        return str(index or "").strip().upper()
    if 0 <= numeric < 26:
        return chr(ord("A") + numeric)
    return str(index or "").strip().upper()


def _question_with_optional_passage(question: str, passage: Any = None) -> str:
    text = str(question or "").strip()
    passage_text = str(passage or "").strip()
    if passage_text:
        return f"{passage_text}\n\n{text}".strip()
    return text


def _choices_dict_to_options(choices: Any) -> list[str]:
    if isinstance(choices, dict):
        texts = list(_json_ready(choices.get("text") or []))
        return [str(text) for text in texts]
    return [str(item) for item in _json_ready(choices or [])]


def _convert_apibench_row(row: dict[str, Any], dataset_name: str, supervision_family: str) -> list[dict[str, Any]]:
    record_id = _stable_record_id(dataset_name, {"api_call": row.get("api_call"), "code": row.get("code")})
    sample = {
        "dataset_name": dataset_name,
        "domain": "tool",
        "record_id": record_id,
        "supervision_family": supervision_family,
        "code": row.get("code"),
        "api_call": row.get("api_call"),
        "provider": row.get("provider"),
        "api_data": _json_ready(row.get("api_data")),
        "metadata": _tool_metadata(dataset_name, record_id, supervision_family),
    }
    return [sample]


def _convert_xlam_row(row: dict[str, Any], dataset_name: str, supervision_family: str) -> list[dict[str, Any]]:
    extra = _json_ready(row.get("extra"))
    record_id = str((extra or {}).get("id") or _stable_record_id(dataset_name, row))
    sample = {
        "dataset_name": dataset_name,
        "domain": "tool",
        "record_id": record_id,
        "supervision_family": supervision_family,
        "messages": _json_ready(row.get("messages") or []),
        "tools": _json_ready(_parse_json_like(row.get("tools")) or []),
        "extra": extra,
        "metadata": _tool_metadata(
            dataset_name,
            record_id,
            supervision_family,
            source_fields={"extra": extra} if extra else None,
        ),
    }
    return [sample]


def _convert_agent_row(row: dict[str, Any], dataset_name: str, supervision_family: str) -> list[dict[str, Any]]:
    trace_id = str(row.get("trace_id") or _stable_record_id(dataset_name, row))
    model = row.get("model")
    session_id = row.get("session_id")
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(row.get("function_calls") or []):
        record_id = f"{trace_id}:{index}"
        source_record_fields = {
            key: _json_ready(value)
            for key, value in item.items()
            if key not in {"messages", "tools"}
        }
        samples.append(
            {
                "dataset_name": dataset_name,
                "domain": "tool",
                "record_id": record_id,
                "supervision_family": supervision_family,
                "trace_id": trace_id,
                "model": model,
                "session_id": session_id,
                "messages": _json_ready(item.get("messages") or []),
                "tools": _json_ready(item.get("tools") or []),
                "metadata": _tool_metadata(
                    dataset_name,
                    record_id,
                    supervision_family,
                    source_fields={
                        "model": model,
                        "session_id": session_id,
                    },
                    source_record_fields=source_record_fields,
                ),
            }
        )
    return samples


def _convert_bfcl_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = str(row.get("id") or _stable_record_id(dataset_name, row))
    parsed_tools = _json_ready(_parse_json_like(row.get("tools")) or [])
    turns = _json_ready(_parse_json_like(row.get("turns")) or [])
    prompt: list[dict[str, Any]] = []
    if isinstance(turns, list) and turns:
        first_turn = turns[0]
        if isinstance(first_turn, list):
            prompt = _ensure_message_prompt(first_turn)
    sample = {
        "dataset_name": dataset_name,
        "domain": "tool",
        "record_id": record_id,
        "id": row.get("id"),
        "multi_turn": row.get("multi_turn"),
        "functions": _json_ready(_parse_json_like(row.get("functions")) or []),
        "tools": parsed_tools,
        "missed_functions": _json_ready(_parse_json_like(row.get("missed_functions")) or {}),
        "initial_config": _json_ready(_parse_json_like(row.get("initial_config")) or {}),
        "involved_classes": _json_ready(_parse_json_like(row.get("involved_classes")) or []),
        "turns": turns,
        "language": row.get("language"),
        "test_category": row.get("test_category"),
        "subset": row.get("subset"),
        "ground_truth": _json_ready(_parse_json_like(row.get("ground_truth")) or []),
        "prompt": prompt,
        "metadata": {
            "dataset_name": dataset_name,
            "domain": "tool",
            "record_id": record_id,
        },
    }
    return [sample]


def _convert_gpqa_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = str(row.get("Record ID") or _stable_record_id(dataset_name, row))
    question = str(row.get("Question") or row.get("Pre-Revision Question") or "").strip()
    answers = [
        str(row.get("Correct Answer") or "").strip(),
        str(row.get("Incorrect Answer 1") or "").strip(),
        str(row.get("Incorrect Answer 2") or "").strip(),
        str(row.get("Incorrect Answer 3") or "").strip(),
    ]
    shuffled = _stable_shuffle(answers, record_id)
    answer_letter = chr(ord("A") + shuffled.index(str(row.get("Correct Answer") or "").strip()))
    sample = {
        "dataset_name": dataset_name,
        "domain": "stem",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": _format_mcqa_prompt(question, shuffled)}],
        "question": question,
        "options": shuffled,
        "answer": answer_letter,
        "metadata": _base_metadata(
            dataset_name,
            "stem",
            record_id,
            reward_type="stem_mcqa",
            source_fields={
                "high_level_domain": row.get("High-level domain"),
                "subdomain": row.get("Subdomain"),
            },
        ),
    }
    return [sample]


def _convert_mmlu_pro_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = str(row.get("question_id") or _stable_record_id(dataset_name, row))
    question = str(row.get("question") or "").strip()
    options = [str(option) for option in _json_ready(row.get("options") or row.get("choices") or [])]
    answer_raw = row.get("answer")
    answer = _answer_index_to_letter(answer_raw) if isinstance(answer_raw, int) else str(answer_raw or "").strip().upper()
    sample = {
        "dataset_name": dataset_name,
        "domain": "stem",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": _format_mcqa_prompt(question, options)}],
        "question": question,
        "options": options,
        "answer": answer,
        "metadata": _base_metadata(
            dataset_name,
            "stem",
            record_id,
            reward_type="stem_mcqa",
            source_fields={
                "category": row.get("category"),
                "src": row.get("src"),
                "cot_content": row.get("cot_content"),
            },
        ),
    }
    return [sample]


def _convert_choice_dict_mcqa_row(
    row: dict[str, Any],
    dataset_name: str,
    question_field: str,
    answer_field: str,
    subset: str | None = None,
) -> list[dict[str, Any]]:
    record_id = str(row.get("id") or _stable_record_id(dataset_name, row))
    question = _question_with_optional_passage(row.get(question_field) or "", row.get("passage"))
    options = _choices_dict_to_options(row.get("choices"))
    sample = {
        "dataset_name": dataset_name,
        "domain": "stem",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": _format_mcqa_prompt(question, options)}],
        "question": question,
        "options": options,
        "answer": str(row.get(answer_field) or "").strip().upper(),
        "metadata": _base_metadata(
            dataset_name,
            "stem",
            record_id,
            reward_type="stem_mcqa",
            source_fields={"subset": subset} if subset else None,
        ),
    }
    return [sample]


def _convert_aqua_rat_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = _stable_record_id(dataset_name, row.get("question"))
    question = str(row.get("question") or "").strip()
    options = [str(option) for option in _json_ready(row.get("options") or [])]
    sample = {
        "dataset_name": dataset_name,
        "domain": "stem",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": _format_mcqa_prompt(question, options)}],
        "question": question,
        "options": options,
        "answer": str(row.get("correct") or "").strip().upper(),
        "metadata": _base_metadata(
            dataset_name,
            "stem",
            record_id,
            reward_type="stem_mcqa",
            source_fields={"rationale": row.get("rationale")},
        ),
    }
    return [sample]


def _convert_scienceqa_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = _stable_record_id(dataset_name, row.get("question"))
    question = str(row.get("question") or "").strip()
    options = [str(choice) for choice in _json_ready(row.get("choices") or [])]
    sample = {
        "dataset_name": dataset_name,
        "domain": "stem",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": _format_mcqa_prompt(question, options)}],
        "question": question,
        "options": options,
        "answer": _answer_index_to_letter(row.get("answer")),
        "metadata": _base_metadata(
            dataset_name,
            "stem",
            record_id,
            reward_type="stem_mcqa",
            source_fields={
                "hint": row.get("hint"),
                "image": _json_ready(row.get("image")),
                "lecture": row.get("lecture"),
                "solution": row.get("solution"),
                "subject": row.get("subject"),
                "task": row.get("task"),
                "topic": row.get("topic"),
                "category": row.get("category"),
                "grade": row.get("grade"),
                "skill": row.get("skill"),
            },
        ),
    }
    return [sample]


def _convert_sciq_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = _stable_record_id(dataset_name, row.get("question"))
    question = str(row.get("question") or "").strip()
    options = _stable_shuffle(
        [
            str(row.get("correct_answer") or "").strip(),
            str(row.get("distractor1") or "").strip(),
            str(row.get("distractor2") or "").strip(),
            str(row.get("distractor3") or "").strip(),
        ],
        record_id,
    )
    answer = chr(ord("A") + options.index(str(row.get("correct_answer") or "").strip()))
    sample = {
        "dataset_name": dataset_name,
        "domain": "stem",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": _format_mcqa_prompt(question, options)}],
        "question": question,
        "options": options,
        "answer": answer,
        "metadata": _base_metadata(
            dataset_name,
            "stem",
            record_id,
            reward_type="stem_mcqa",
            source_fields={"support": row.get("support")},
        ),
    }
    return [sample]


def _convert_agieval_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = _stable_record_id(dataset_name, {"question": row.get("question"), "label": row.get("label")})
    question = _question_with_optional_passage(row.get("question") or "", row.get("passage"))
    options = [str(option) for option in _json_ready(row.get("options") or [])]
    sample = {
        "dataset_name": dataset_name,
        "domain": "stem",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": _format_mcqa_prompt(question, options)}],
        "question": question,
        "options": options,
        "answer": str(row.get("label") or "").strip().upper(),
        "metadata": _base_metadata(
            dataset_name,
            "stem",
            record_id,
            reward_type="stem_mcqa",
            source_fields={
                "explanation": row.get("explanation"),
                "other": _json_ready(row.get("other")),
            },
        ),
    }
    return [sample]


def _convert_medmcqa_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = str(row.get("id") or _stable_record_id(dataset_name, row))
    question = str(row.get("question") or "").strip()
    options = [str(row.get(key) or "") for key in ("opa", "opb", "opc", "opd")]
    answer_index = int(row.get("cop", 0))
    answer = chr(ord("A") + answer_index)
    sample = {
        "dataset_name": dataset_name,
        "domain": "stem",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": _format_mcqa_prompt(question, options)}],
        "question": question,
        "options": options,
        "answer": answer,
        "metadata": _base_metadata(
            dataset_name,
            "stem",
            record_id,
            reward_type="stem_mcqa",
            source_fields={
                "choice_type": row.get("choice_type"),
                "subject_name": row.get("subject_name"),
                "topic_name": row.get("topic_name"),
                "exp": row.get("exp"),
            },
        ),
    }
    return [sample]


def _extract_nemotron_mcqa_prompt(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, dict):
        return _ensure_message_prompt(prompt.get("input"))
    return _ensure_message_prompt(prompt)


def _convert_nemotron_knowledge_mcqa_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = str(row.get("uuid") or _stable_record_id(dataset_name, row))
    prompt = _extract_nemotron_mcqa_prompt(row.get("responses_create_params"))
    sample = {
        "dataset_name": dataset_name,
        "domain": "stem",
        "record_id": record_id,
        "prompt": prompt,
        "answer": str(row.get("expected_answer") or "").strip().upper(),
        "options": _json_ready(row.get("options") or []),
        "responses_create_params": _json_ready(row.get("responses_create_params") or {}),
        "template_metadata": _json_ready(row.get("template_metadata") or {}),
        "reward_profiles": _json_ready(row.get("reward_profiles") or []),
        "metadata": _base_metadata(dataset_name, "stem", record_id, reward_type="stem_mcqa"),
    }
    return [sample]


def _convert_instruction_following_row(row: dict[str, Any], dataset_name: str, reward_type: str) -> list[dict[str, Any]]:
    record_id = str(row.get("key") or _stable_record_id(dataset_name, row))
    prompt_text = str(row.get("prompt") or "").strip()
    sample = {
        "dataset_name": dataset_name,
        "domain": "structured",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": prompt_text}],
        "prompt_text": prompt_text,
        "instruction_id_list": _json_ready(row.get("instruction_id_list") or []),
        "kwargs": _json_ready(row.get("kwargs") or []),
        "metadata": _base_metadata(dataset_name, "structured", record_id, reward_type=reward_type),
    }
    return [sample]


def _convert_jsonschemabench_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = str(row.get("unique_id") or _stable_record_id(dataset_name, row))
    schema_text = str(row.get("json_schema") or "").strip()
    schema = _parse_json_like(schema_text)
    sample = {
        "dataset_name": dataset_name,
        "domain": "structured",
        "record_id": record_id,
        "prompt": [{"role": "user", "content": schema_text}],
        "schema": schema if isinstance(schema, dict) else {},
        "unique_id": row.get("unique_id"),
        "metadata": _base_metadata(dataset_name, "structured", record_id, reward_type="structured_json_schema"),
    }
    return [sample]


def _convert_nemotron_structured_row(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    record_id = _stable_record_id(dataset_name, row)
    schema = _parse_json_like(row.get("schema_str"))
    sample = {
        "dataset_name": dataset_name,
        "domain": "structured",
        "record_id": record_id,
        "prompt": _extract_nemotron_mcqa_prompt(row.get("responses_create_params")),
        "schema": schema if isinstance(schema, dict) else {},
        "responses_create_params": _json_ready(row.get("responses_create_params") or {}),
        "schema_str": row.get("schema_str"),
        "schema_type": row.get("schema_type"),
        "schema_fields_count": row.get("schema_fields_count"),
        "metadata": _base_metadata(dataset_name, "structured", record_id, reward_type="structured_json_schema"),
    }
    return [sample]


def convert_row_for_pool(row: dict[str, Any], dataset_name: str) -> list[dict[str, Any]]:
    if dataset_name.startswith("apibench_"):
        return _convert_apibench_row(row, dataset_name, "function_call_single")
    if dataset_name == "xlam_function_calling_60k":
        return _convert_xlam_row(row, dataset_name, "function_call_single")
    if dataset_name == "agent_function_calling_open_dataset":
        return _convert_agent_row(row, dataset_name, "function_call_single")
    if dataset_name in {"bfcl_v3", "bfcl_v3_multi_turn_base"}:
        return _convert_bfcl_row(row, dataset_name)
    if dataset_name == "gpqa":
        return _convert_gpqa_row(row, dataset_name)
    if dataset_name in {"mmlu_pro", "mmlu"}:
        return _convert_mmlu_pro_row(row, dataset_name)
    if dataset_name == "ai2_arc":
        return _convert_choice_dict_mcqa_row(row, dataset_name, "question", "answerKey")
    if dataset_name == "openbookqa":
        return _convert_choice_dict_mcqa_row(row, dataset_name, "question_stem", "answerKey")
    if dataset_name == "aqua_rat":
        return _convert_aqua_rat_row(row, dataset_name)
    if dataset_name == "scienceqa":
        return _convert_scienceqa_row(row, dataset_name)
    if dataset_name == "sciq":
        return _convert_sciq_row(row, dataset_name)
    if dataset_name == "agieval":
        return _convert_agieval_row(row, dataset_name)
    if dataset_name == "medmcqa":
        return _convert_medmcqa_row(row, dataset_name)
    if dataset_name == "nemotron_knowledge_mcqa":
        return _convert_nemotron_knowledge_mcqa_row(row, dataset_name)
    if dataset_name == "ifeval":
        return _convert_instruction_following_row(row, dataset_name, "instruction_following_strict")
    if dataset_name == "ifbench_test":
        return _convert_instruction_following_row(row, dataset_name, "instruction_following_soft")
    if dataset_name == "jsonschemabench":
        return _convert_jsonschemabench_row(row, dataset_name)
    if dataset_name == "nemotron_structured_outputs":
        return _convert_nemotron_structured_row(row, dataset_name)
    raise ValueError(f"Unsupported dataset for pool conversion: {dataset_name}")


def write_jsonl_rows(dest: Path, rows: Iterable[dict[str, Any]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")
    tmp_path.replace(dest)


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def iter_jsonl_dir_rows(path: Path) -> Iterator[dict[str, Any]]:
    for child in sorted(path.glob("*.jsonl")):
        yield from iter_jsonl_rows(child)


def iter_json_rows(path: Path) -> Iterator[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row


def iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if isinstance(row, dict):
                yield row


def iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        for row in batch.to_pylist():
            if isinstance(row, dict):
                yield row


def _iter_source_rows(path: Path | list[Path], reader: str) -> Iterator[dict[str, Any]]:
    if isinstance(path, list):
        for item in path:
            yield from _iter_source_rows(item, reader)
        return
    if reader == "jsonl":
        yield from iter_jsonl_rows(path)
        return
    if reader == "jsonl_dir":
        yield from iter_jsonl_dir_rows(path)
        return
    if reader == "json":
        yield from iter_json_rows(path)
        return
    if reader == "csv":
        yield from iter_csv_rows(path)
        return
    if reader == "parquet":
        yield from iter_parquet_rows(path)
        return
    raise ValueError(f"Unsupported reader: {reader}")


def _base_dataset_name(dataset_key: str) -> str:
    if dataset_key.startswith("apibench_"):
        return dataset_key
    if dataset_key.startswith("agent_function_calling_open_dataset_"):
        return "agent_function_calling_open_dataset"
    if dataset_key.startswith("nemotron_knowledge_mcqa_train_"):
        return "nemotron_knowledge_mcqa"
    if dataset_key.startswith("bfcl_v3_multi_turn_base"):
        return "bfcl_v3_multi_turn_base"
    if dataset_key.startswith("bfcl_v3"):
        return "bfcl_v3"
    if dataset_key == "gpqa_main":
        return "gpqa"
    if dataset_key.startswith("mmlu_pro_"):
        return "mmlu_pro"
    if dataset_key.startswith("mmlu_"):
        return "mmlu"
    if dataset_key.startswith("medmcqa_"):
        return "medmcqa"
    if dataset_key.startswith("jsonschemabench_"):
        return "jsonschemabench"
    if dataset_key.startswith("ai2_arc_"):
        return "ai2_arc"
    if dataset_key.startswith("aqua_rat_"):
        return "aqua_rat"
    if dataset_key.startswith("openbookqa_"):
        return "openbookqa"
    if dataset_key.startswith("scienceqa_"):
        return "scienceqa"
    if dataset_key.startswith("sciq_"):
        return "sciq"
    if dataset_key.startswith("agieval_"):
        return "agieval"
    return dataset_key


def build_dataset(dataset_key: str) -> int:
    spec = DATASET_SPECS[dataset_key]
    source = spec["source"]
    if isinstance(source, list):
        source = [Path(item) for item in source]
    else:
        source = Path(source)
    rows: list[dict[str, Any]] = []
    base_dataset_name = _base_dataset_name(dataset_key)
    for raw_row in _iter_source_rows(source, spec["reader"]):
        rows.extend(convert_row_for_pool(raw_row, base_dataset_name))
    write_jsonl_rows(Path(spec["output"]), rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pool files directly from open_data.")
    parser.add_argument("--dataset", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.dataset or sorted(DATASET_SPECS)
    for dataset_key in selected:
        if dataset_key not in DATASET_SPECS:
            raise SystemExit(f"Unsupported dataset key: {dataset_key}")
        count = build_dataset(dataset_key)
        print(f"{dataset_key}: {count}")


if __name__ == "__main__":
    main()
