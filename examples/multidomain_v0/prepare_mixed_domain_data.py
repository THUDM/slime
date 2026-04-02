#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class SourceSpec:
    source: Path
    dataset_format: str
    ratio: float
    domain: str


def normalize_message(message: dict[str, Any]) -> dict[str, str]:
    role = str(message.get("role", "user"))
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        content = "\n".join(part for part in parts if part)
    return {"role": role, "content": "" if content is None else str(content)}


def normalize_json_like_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def normalize_tool_definition(tool: dict[str, Any]) -> dict[str, Any]:
    if "function" in tool and isinstance(tool["function"], dict):
        function = tool["function"]
        return {
            "type": str(tool.get("type", "function")),
            "function": {
                "name": str(function.get("name", "")),
                "description": str(function.get("description", "")),
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            },
        }

    return {
        "type": str(tool.get("type", "function")),
        "function": {
            "name": str(tool.get("name", "")),
            "description": str(tool.get("description", "")),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        },
    }


def normalize_tool_definitions(raw_tools: Any) -> list[dict[str, Any]]:
    raw_tools = normalize_json_like_value(raw_tools)
    if not isinstance(raw_tools, list):
        return []
    return [normalize_tool_definition(tool) for tool in raw_tools if isinstance(tool, dict)]


def normalize_ground_truth_calls(raw_calls: Any) -> list[dict[str, Any]]:
    raw_calls = normalize_json_like_value(raw_calls)
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    if not isinstance(raw_calls, list):
        return []

    normalized: list[dict[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            normalized.append(
                {
                    "name": str(function.get("name", "")),
                    "arguments": normalize_json_like_value(function.get("arguments", {})),
                }
            )
            continue
        normalized.append(
            {
                "name": str(call.get("name", "")),
                "arguments": normalize_json_like_value(call.get("arguments", call.get("parameters", {}))),
            }
        )
    return normalized


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
        return

    if path.suffix == ".json":
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            rows: list[dict[str, Any]] = []
            try:
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                    else:
                        raise ValueError(f"Unsupported JSONL row in {path}")
            except Exception:
                raise exc
            if rows:
                for row in rows:
                    yield row
                return
            raise exc
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(payload, dict):
            yield payload
            return
        raise ValueError(f"Unsupported JSON payload: {path}")

    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise RuntimeError("Reading parquet sources requires pyarrow.") from exc
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                if isinstance(row, dict):
                    yield row
        return

    raise ValueError(f"Unsupported source format: {path}")


def iter_selected_samples(
    samples: Iterable[dict[str, Any]],
    skip_samples: int,
    max_samples: int | None,
) -> Iterator[dict[str, Any]]:
    seen = 0
    yielded = 0
    for sample in samples:
        if seen < skip_samples:
            seen += 1
            continue
        if max_samples is not None and yielded >= max_samples:
            break
        yielded += 1
        yield sample


def build_choice_prompt(question: str, choices: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    options_block = "\n".join(f"{label}. {text}" for label, text in choices)
    prompt = f"{question}\n{options_block}"
    return [{"role": "user", "content": prompt}]


def dataset_domain(dataset_format: str) -> str:
    if dataset_format in {
        "nemotron_knowledge_mcqa",
        "mmlu_pro",
        "ai2_arc",
        "scienceqa",
        "medmcqa",
        "openbookqa",
        "sciq",
    }:
        return "stem"
    if dataset_format in {"apigen_mt_5k", "xlam_function_calling_60k"}:
        return "tool"
    if dataset_format in {"nemotron_structured_outputs", "ifeval"}:
        return "structured"
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def make_sample(prompt: list[dict[str, str]], metadata: dict[str, Any], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "label": "",
        "metadata": metadata,
        "tools": tools or [],
    }


def convert_nemotron_knowledge_mcqa_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = [normalize_message(message) for message in row.get("responses_create_params", {}).get("input") or [] if isinstance(message, dict)]
    return [
        make_sample(
            prompt=prompt,
            metadata={
                "domain": "stem",
                "dataset_name": "nemotron_knowledge_mcqa",
                "reward_type": "stem_mcqa",
                "answer": str(row.get("expected_answer", "")).strip(),
                "record_id": row.get("uuid"),
            },
        )
    ]


def convert_mmlu_pro_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    options = [(CHOICE_LABELS[index], str(option)) for index, option in enumerate(row.get("options") or [])]
    return [
        make_sample(
            prompt=build_choice_prompt(str(row.get("question", "")), options),
            metadata={
                "domain": "stem",
                "dataset_name": "mmlu_pro",
                "reward_type": "stem_mcqa",
                "answer": str(row.get("answer", "")).strip(),
                "record_id": row.get("question_id"),
            },
        )
    ]


def convert_ai2_arc_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    choices = row.get("choices") or {}
    items = list(zip(choices.get("label") or [], choices.get("text") or []))
    return [
        make_sample(
            prompt=build_choice_prompt(str(row.get("question", "")), [(str(label), str(text)) for label, text in items]),
            metadata={
                "domain": "stem",
                "dataset_name": "ai2_arc",
                "reward_type": "stem_mcqa",
                "answer": str(row.get("answerKey", "")).strip(),
                "record_id": row.get("id"),
            },
        )
    ]


def convert_scienceqa_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    choices = row.get("choices") or []
    answer_index = int(row.get("answer", 0))
    options = [(CHOICE_LABELS[index], str(choice)) for index, choice in enumerate(choices)]
    return [
        make_sample(
            prompt=build_choice_prompt(str(row.get("question", "")), options),
            metadata={
                "domain": "stem",
                "dataset_name": "scienceqa",
                "reward_type": "stem_mcqa",
                "answer": CHOICE_LABELS[answer_index],
            },
        )
    ]


def convert_medmcqa_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    options = [
        ("A", str(row.get("opa", ""))),
        ("B", str(row.get("opb", ""))),
        ("C", str(row.get("opc", ""))),
        ("D", str(row.get("opd", ""))),
    ]
    answer_index = int(row.get("cop", 0))
    return [
        make_sample(
            prompt=build_choice_prompt(str(row.get("question", "")), options),
            metadata={
                "domain": "stem",
                "dataset_name": "medmcqa",
                "reward_type": "stem_mcqa",
                "answer": CHOICE_LABELS[answer_index],
                "record_id": row.get("id"),
            },
        )
    ]


def convert_openbookqa_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    choices = row.get("choices") or {}
    items = list(zip(choices.get("label") or [], choices.get("text") or []))
    return [
        make_sample(
            prompt=build_choice_prompt(str(row.get("question_stem", "")), [(str(label), str(text)) for label, text in items]),
            metadata={
                "domain": "stem",
                "dataset_name": "openbookqa",
                "reward_type": "stem_mcqa",
                "answer": str(row.get("answerKey", "")).strip(),
                "record_id": row.get("id"),
            },
        )
    ]


def convert_sciq_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    choices = [
        str(row.get("correct_answer", "")),
        str(row.get("distractor1", "")),
        str(row.get("distractor2", "")),
        str(row.get("distractor3", "")),
    ]
    rng = random.Random(str(row.get("question", "")))
    rng.shuffle(choices)
    answer_index = choices.index(str(row.get("correct_answer", "")))
    options = [(CHOICE_LABELS[index], choice) for index, choice in enumerate(choices)]
    return [
        make_sample(
            prompt=build_choice_prompt(str(row.get("question", "")), options),
            metadata={
                "domain": "stem",
                "dataset_name": "sciq",
                "reward_type": "stem_mcqa",
                "answer": CHOICE_LABELS[answer_index],
            },
        )
    ]


def convert_apigen_mt_5k_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    tools = normalize_tool_definitions(row.get("tools") or [])
    conversations = row.get("conversations") or []
    history: list[dict[str, str]] = []
    system_prompt = row.get("system")
    if system_prompt:
        history.append({"role": "system", "content": str(system_prompt)})

    samples: list[dict[str, Any]] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("from", "")).lower()
        value = turn.get("value", "")
        if speaker in {"human", "user"}:
            history.append({"role": "user", "content": "" if value is None else str(value)})
            continue
        if speaker in {"gpt", "assistant"}:
            history.append({"role": "assistant", "content": "" if value is None else str(value)})
            continue
        if speaker == "observation":
            history.append({"role": "tool", "content": "" if value is None else str(value)})
            continue
        if speaker != "function_call":
            continue

        ground_truth = normalize_ground_truth_calls(value)
        if not ground_truth:
            continue
        samples.append(
            make_sample(
                prompt=list(history),
                metadata={
                    "domain": "tool",
                    "dataset_name": "apigen_mt_5k",
                    "reward_type": "tool_call",
                    "ground_truth": ground_truth,
                    "tools": tools,
                    "parser_type": "qwen25",
                },
                tools=tools,
            )
        )

    return samples


def convert_xlam_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or []
    tools = normalize_tool_definitions(row.get("tools") or [])
    samples: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or str(message.get("role", "")) != "assistant":
            continue
        ground_truth = normalize_ground_truth_calls(message.get("tool_calls"))
        if not ground_truth:
            continue
        prompt = [normalize_message(previous) for previous in messages[:index] if isinstance(previous, dict)]
        samples.append(
            make_sample(
                prompt=prompt,
                metadata={
                    "domain": "tool",
                    "dataset_name": "xlam_function_calling_60k",
                    "reward_type": "tool_call",
                    "ground_truth": ground_truth,
                    "tools": tools,
                    "parser_type": "qwen25",
                    "record_id": (row.get("extra") or {}).get("id"),
                },
                tools=tools,
            )
        )
    return samples


def convert_nemotron_structured_outputs_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = [normalize_message(message) for message in row.get("responses_create_params", {}).get("input") or [] if isinstance(message, dict)]
    schema = normalize_json_like_value(row.get("schema_str")) or {}
    return [
        make_sample(
            prompt=prompt,
            metadata={
                "domain": "structured",
                "dataset_name": "nemotron_structured_outputs",
                "reward_type": "structured_json_schema",
                "schema": schema,
                "schema_type": row.get("schema_type"),
            },
        )
    ]


def convert_ifeval_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_text = str(row.get("prompt", ""))
    return [
        make_sample(
            prompt=[{"role": "user", "content": prompt_text}],
            metadata={
                "domain": "structured",
                "dataset_name": "ifeval",
                "reward_type": "ifeval",
                "record_id": row.get("key"),
                "prompt_text": prompt_text,
                "instruction_id_list": row.get("instruction_id_list") or [],
                "kwargs": row.get("kwargs") or [],
            },
        )
    ]


def convert_row(row: dict[str, Any], dataset_format: str) -> list[dict[str, Any]]:
    if dataset_format == "nemotron_knowledge_mcqa":
        return convert_nemotron_knowledge_mcqa_row(row)
    if dataset_format == "mmlu_pro":
        return convert_mmlu_pro_row(row)
    if dataset_format == "ai2_arc":
        return convert_ai2_arc_row(row)
    if dataset_format == "scienceqa":
        return convert_scienceqa_row(row)
    if dataset_format == "medmcqa":
        return convert_medmcqa_row(row)
    if dataset_format == "openbookqa":
        return convert_openbookqa_row(row)
    if dataset_format == "sciq":
        return convert_sciq_row(row)
    if dataset_format == "apigen_mt_5k":
        return convert_apigen_mt_5k_row(row)
    if dataset_format == "xlam_function_calling_60k":
        return convert_xlam_row(row)
    if dataset_format == "nemotron_structured_outputs":
        return convert_nemotron_structured_outputs_row(row)
    if dataset_format == "ifeval":
        return convert_ifeval_row(row)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def iter_converted_samples(source: Path, dataset_format: str) -> Iterator[dict[str, Any]]:
    for row in iter_rows(source):
        for converted in convert_row(row, dataset_format):
            yield converted


def proportional_counts(total: int, ratios: Sequence[float]) -> list[int]:
    if total <= 0:
        return [0 for _ in ratios]
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("source ratios must sum to a positive value")

    exact_counts = [(total * ratio) / ratio_sum for ratio in ratios]
    counts = [math.floor(value) for value in exact_counts]
    remainder = total - sum(counts)
    ranked = sorted(
        range(len(ratios)),
        key=lambda index: (exact_counts[index] - counts[index], ratios[index], -index),
        reverse=True,
    )
    for index in ranked[:remainder]:
        counts[index] += 1
    return counts


def build_weighted_schedule(weights: Sequence[float]) -> list[int]:
    decimals = [Decimal(str(weight)) for weight in weights]
    max_places = max(max(-value.as_tuple().exponent, 0) for value in decimals)
    scale = 10**max_places
    scaled_weights = [max(1, int(value * scale)) for value in decimals]
    total = sum(scaled_weights)
    current = [0 for _ in scaled_weights]
    schedule: list[int] = []
    for _ in range(max(1, total)):
        best_index = 0
        best_value: int | None = None
        for index, weight in enumerate(scaled_weights):
            current[index] += weight
            if best_value is None or current[index] > best_value:
                best_index = index
                best_value = current[index]
        current[best_index] -= total
        schedule.append(best_index)
    return schedule


def prepare_single_source_samples(spec: SourceSpec, skip_samples: int, max_samples: int | None) -> Iterator[dict[str, Any]]:
    return iter_selected_samples(iter_converted_samples(spec.source, spec.dataset_format), skip_samples, max_samples)


def count_converted_samples(spec: SourceSpec) -> int:
    return sum(1 for _ in iter_converted_samples(spec.source, spec.dataset_format))


def max_total_for_ratios(capacities: Sequence[int], ratios: Sequence[float]) -> int:
    if not capacities:
        return 0
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("source ratios must sum to a positive value")
    limits = []
    for capacity, ratio in zip(capacities, ratios):
        if ratio <= 0:
            continue
        limits.append(math.floor(capacity * ratio_sum / ratio))
    return max(0, min(limits)) if limits else 0


def allocate_sample_counts(capacities: Sequence[int], ratios: Sequence[float], total_target: int | None) -> list[int]:
    capacities = [max(0, int(capacity)) for capacity in capacities]
    if total_target is None:
        total_target = max_total_for_ratios(capacities, ratios)
    total_target = min(int(total_target), sum(capacities))
    if total_target <= 0:
        return [0 for _ in capacities]

    remaining = capacities[:]
    allocated = [0 for _ in capacities]
    active = [index for index, capacity in enumerate(remaining) if capacity > 0]
    remaining_total = total_target

    while remaining_total > 0 and active:
        proposed = proportional_counts(remaining_total, [ratios[index] for index in active])
        progress = 0
        next_active: list[int] = []
        for local_index, index in enumerate(active):
            take = min(remaining[index], proposed[local_index])
            allocated[index] += take
            remaining[index] -= take
            progress += take
            if remaining[index] > 0:
                next_active.append(index)
        if progress == 0:
            break
        remaining_total -= progress
        active = next_active

    return allocated


def next_domain_sample(
    domain_name: str,
    domain_sources: list[int],
    source_schedules: dict[str, list[int]],
    iterators: list[Iterator[dict[str, Any]]],
    active_sources: list[bool],
    remaining_counts: list[int],
) -> dict[str, Any] | None:
    schedule = source_schedules[domain_name]
    while True:
        emitted = False
        for local_index in schedule:
            source_index = domain_sources[local_index]
            if not active_sources[source_index] or remaining_counts[source_index] <= 0:
                continue
            try:
                sample = next(iterators[source_index])
            except StopIteration:
                active_sources[source_index] = False
                continue
            emitted = True
            remaining_counts[source_index] -= 1
            return sample
        if not emitted:
            return None


def prepare_mixed_samples(
    specs: Sequence[SourceSpec],
    skip_samples: int,
    max_samples: int | None,
) -> Iterator[dict[str, Any]]:
    capacities = [count_converted_samples(spec) for spec in specs]
    ratios = [spec.ratio for spec in specs]
    skip_counts = allocate_sample_counts(capacities, ratios, skip_samples)
    remaining_capacities = [max(0, capacity - skip) for capacity, skip in zip(capacities, skip_counts)]
    target_counts = allocate_sample_counts(remaining_capacities, ratios, max_samples)

    domains: list[str] = []
    domain_to_indices: dict[str, list[int]] = {}
    domain_weights: dict[str, float] = {}
    for index, spec in enumerate(specs):
        if spec.domain not in domain_to_indices:
            domains.append(spec.domain)
            domain_to_indices[spec.domain] = []
            domain_weights[spec.domain] = 0.0
        domain_to_indices[spec.domain].append(index)
        domain_weights[spec.domain] += spec.ratio

    total_target = sum(target_counts)
    remaining_counts = target_counts[:]
    iterators = [
        iter(prepare_single_source_samples(spec, skip_counts[index], None))
        for index, spec in enumerate(specs)
    ]
    active_sources = [True for _ in specs]
    source_schedules = {
        domain: build_weighted_schedule([specs[index].ratio for index in domain_to_indices[domain]])
        for domain in domains
    }
    domain_remaining = {
        domain: sum(remaining_counts[index] for index in domain_to_indices[domain])
        for domain in domains
    }
    yielded = 0

    for domain in sorted(domains, key=lambda name: domain_weights[name], reverse=True):
        if domain_remaining[domain] <= 0:
            continue
        sample = next_domain_sample(domain, domain_to_indices[domain], source_schedules, iterators, active_sources, remaining_counts)
        if sample is None:
            continue
        yielded += 1
        domain_remaining[domain] -= 1
        yield sample
        if yielded >= total_target:
            return

    domain_schedule = build_weighted_schedule([domain_weights[domain] for domain in domains])
    while True:
        emitted = False
        for domain_index in domain_schedule:
            domain = domains[domain_index]
            if domain_remaining[domain] <= 0:
                continue
            sample = next_domain_sample(domain, domain_to_indices[domain], source_schedules, iterators, active_sources, remaining_counts)
            if sample is None:
                continue
            emitted = True
            yielded += 1
            domain_remaining[domain] -= 1
            yield sample
            if yielded >= total_target:
                return
        if not emitted:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert mixed-domain datasets into slime jsonl format.")
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--dataset-format", action="append", required=True)
    parser.add_argument("--source-ratio", action="append", type=float, required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (len(args.source) == len(args.dataset_format) == len(args.source_ratio)):
        raise ValueError("source, dataset-format, and source-ratio counts must match")

    specs = [
        SourceSpec(Path(source), dataset_format, float(ratio), dataset_domain(dataset_format))
        for source, dataset_format, ratio in zip(args.source, args.dataset_format, args.source_ratio)
    ]

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(specs) == 1:
        samples = prepare_single_source_samples(specs[0], args.skip_samples, args.max_samples)
    else:
        samples = prepare_mixed_samples(specs, args.skip_samples, args.max_samples)

    with dest.open("w", encoding="utf-8") as fout:
        for sample in samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
