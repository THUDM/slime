from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable


DEFAULT_BFCL_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FC"


def _load_bfcl_backend() -> dict[str, Any]:
    try:
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX
        from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
        from bfcl_eval.eval_checker.eval_runner import load_dataset_entry, runner
    except Exception as exc:
        raise RuntimeError(
            "bfcl-eval is required for official BFCL evaluation. "
            "Install bfcl-eval in the execution environment."
        ) from exc
    return {
        "VERSION_PREFIX": VERSION_PREFIX,
        "MODEL_CONFIG_MAPPING": MODEL_CONFIG_MAPPING,
        "load_dataset_entry": load_dataset_entry,
        "runner": runner,
    }


def _coerce_prompt_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return [dict(item) for item in prompt if isinstance(item, dict)]
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt.strip()}]
    return []


def _prompt_user_text(row: dict[str, Any]) -> str:
    for message in _coerce_prompt_messages(row.get("prompt")):
        if str(message.get("role", "")).lower() == "user":
            return str(message.get("content", "")).strip()
    return ""


def _bfcl_dataset_name(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(row.get("dataset_name") or metadata.get("dataset_name") or "").strip()


def _bfcl_test_category(row: dict[str, Any]) -> str:
    if row.get("test_category"):
        return str(row["test_category"]).strip()
    dataset_name = _bfcl_dataset_name(row)
    if dataset_name == "bfcl_v3_multi_turn_base":
        return "multi_turn_base"
    raise RuntimeError(f"Unable to infer BFCL test category for row: dataset_name={dataset_name!r}")


def _row_native_id(row: dict[str, Any]) -> str:
    if row.get("id") not in (None, ""):
        return str(row["id"])
    record_id = row.get("record_id")
    if record_id not in (None, ""):
        return str(record_id)
    return ""


def _multi_turn_prompt_ids(rows: list[dict[str, Any]], backend: dict[str, Any]) -> list[str]:
    prompt_entries = backend["load_dataset_entry"](
        "multi_turn_base",
        include_prereq=False,
        include_language_specific_hint=False,
    )
    if len(prompt_entries) < len(rows):
        raise RuntimeError(
            f"BFCL multi-turn prompt set too small: need {len(rows)} entries, got {len(prompt_entries)}"
        )
    return [str(entry["id"]) for entry in prompt_entries[: len(rows)]]


def normalize_multi_turn_result(raw_result: Any) -> list[list[str]]:
    if isinstance(raw_result, str):
        text = raw_result.strip()
        return [[text]] if text else []
    if not isinstance(raw_result, list):
        return []

    if all(isinstance(item, str) for item in raw_result):
        return [[item] for item in raw_result if str(item).strip()]

    normalized: list[list[str]] = []
    for turn in raw_result:
        if isinstance(turn, str):
            text = turn.strip()
            if text:
                normalized.append([text])
            continue
        if not isinstance(turn, list):
            continue
        messages = [str(item) for item in turn if str(item).strip()]
        normalized.append(messages)
    return normalized


def build_bfcl_result_entries(rows: list[dict[str, Any]], outputs: list[Any], *, backend: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if len(rows) != len(outputs):
        raise ValueError(f"rows and outputs must have the same length, got {len(rows)} and {len(outputs)}")
    backend = backend or _load_bfcl_backend()

    multi_turn_indices = [idx for idx, row in enumerate(rows) if _bfcl_dataset_name(row) == "bfcl_v3_multi_turn_base"]
    resolved_multi_turn_ids: dict[int, str] = {}
    if multi_turn_indices:
        assigned = _multi_turn_prompt_ids([rows[idx] for idx in multi_turn_indices], backend)
        resolved_multi_turn_ids = dict(zip(multi_turn_indices, assigned))

    entries: list[dict[str, Any]] = []
    for idx, (row, output) in enumerate(zip(rows, outputs)):
        dataset_name = _bfcl_dataset_name(row)
        if dataset_name == "bfcl_v3_multi_turn_base":
            entry_id = resolved_multi_turn_ids.get(idx, "")
        else:
            entry_id = _row_native_id(row)
        if not entry_id:
            raise RuntimeError(f"Missing BFCL entry id for dataset {dataset_name}")

        if dataset_name == "bfcl_v3_multi_turn_base":
            result = normalize_multi_turn_result(output)
        else:
            result = "" if output is None else str(output)
        entries.append({"id": entry_id, "result": result})
    return entries


def _instantiate_handler(model_name: str, backend: dict[str, Any]):
    config = backend["MODEL_CONFIG_MAPPING"].get(model_name)
    if config is None:
        supported = ", ".join(sorted(backend["MODEL_CONFIG_MAPPING"]))
        raise RuntimeError(f"Unsupported BFCL model '{model_name}'. Supported models: {supported}")
    return config.model_handler(
        model_name=config.model_name,
        temperature=0,
        registry_name=model_name,
        is_fc_model=config.is_fc_model,
    )


def _score_file_headers(score_dir: Path, model_name: str, test_categories: list[str], *, version_prefix: str) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    model_score_dir = score_dir / model_name
    if not model_score_dir.exists():
        return headers
    for category in test_categories:
        matches = list(model_score_dir.rglob(f"{version_prefix}_{category}_score.json"))
        if not matches:
            continue
        with matches[0].open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        if not first_line:
            continue
        headers[category] = json.loads(first_line)
    return headers


def summarize_bfcl_scores(headers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0.0
    categories: dict[str, dict[str, Any]] = {}
    for category, header in headers.items():
        total_count = int(header.get("total_count", 0))
        accuracy = float(header.get("accuracy", 0.0))
        categories[category] = {
            "accuracy": accuracy,
            "total_count": total_count,
            "correct_count": int(header.get("correct_count", round(accuracy * total_count))),
        }
        total += total_count
        correct += accuracy * total_count
    overall = correct / total if total else 0.0
    return {"overall_accuracy": overall, "total_count": total, "categories": categories}


def run_bfcl_official_eval(
    rows: list[dict[str, Any]],
    outputs: list[Any],
    *,
    model_name: str = DEFAULT_BFCL_MODEL_NAME,
    result_dir: Path | None = None,
    score_dir: Path | None = None,
    allow_missing: bool = True,
) -> dict[str, Any]:
    if not rows:
        return {"overall_accuracy": 0.0, "total_count": 0, "categories": {}}

    backend = _load_bfcl_backend()
    entries = build_bfcl_result_entries(rows, outputs, backend=backend)
    test_categories = sorted({_bfcl_test_category(row) for row in rows})

    own_tmpdir = None
    if result_dir is None or score_dir is None:
        own_tmpdir = tempfile.TemporaryDirectory(prefix="bfcl_official_eval_")
        base = Path(own_tmpdir.name)
        result_dir = result_dir or (base / "result")
        score_dir = score_dir or (base / "score")

    assert result_dir is not None
    assert score_dir is not None
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    handler = _instantiate_handler(model_name, backend)
    handler.write(entries, result_dir, update_mode=False)
    backend["runner"]([model_name], test_categories, result_dir, score_dir, allow_missing=allow_missing)
    headers = _score_file_headers(score_dir, model_name, test_categories, version_prefix=backend["VERSION_PREFIX"])
    summary = summarize_bfcl_scores(headers)
    summary["model_name"] = model_name
    summary["result_dir"] = str(result_dir)
    summary["score_dir"] = str(score_dir)

    if own_tmpdir is not None:
        summary["_tmpdir"] = own_tmpdir
    return summary


def generate_bfcl_multi_turn_outputs(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    generate_one: Callable[[str], str],
    max_prompt_tokens: int,
) -> list[list[list[str]]]:
    outputs: list[list[list[str]]] = []
    for row in rows:
        messages = _coerce_prompt_messages(row.get("messages") or row.get("prompt"))
        tools = list(row.get("tools") or [])
        ground_truth = list(row.get("ground_truth") or [])
        sample_outputs: list[list[str]] = []
        for _ in ground_truth:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if len(input_ids) > max_prompt_tokens:
                break
            response = generate_one(prompt_text)
            sample_outputs.append([response])
            messages = [*messages, {"role": "assistant", "content": response}]
        outputs.append(sample_outputs)
    return outputs


def summary_to_metrics(eval_name: str, summary: dict[str, Any]) -> dict[str, float]:
    metrics = {f"eval/{eval_name}": float(summary.get("overall_accuracy", 0.0))}
    for category, payload in (summary.get("categories") or {}).items():
        metrics[f"eval/{eval_name}/by_category/{category}/score"] = float(payload.get("accuracy", 0.0))
        metrics[f"eval/{eval_name}/by_category/{category}/count"] = float(payload.get("total_count", 0))
    return metrics
