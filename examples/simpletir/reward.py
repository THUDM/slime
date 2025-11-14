from __future__ import annotations

from typing import Iterable, Optional

from slime.utils.types import Sample

from .dataset import SimpleTIRRecord, get_dataset, get_dataset_from_path
from .reward_score import compute_reward_async

try:
    from slime.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
    from slime.rollout.rm_hub.math_utils import grade_answer_verl
except ImportError:  # pragma: no cover - defensive
    extract_boxed_answer = None
    grade_answer_verl = None


def _resolve_record(args, sample: Sample, *, split: str) -> SimpleTIRRecord:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    dataset_path = (
        metadata.get("simpletir_dataset_path")
        or metadata.get("simpletir_dataset")
        or metadata.get("simpletir_dataset_spec")
    )
    dataset_split = metadata.get("simpletir_split") or metadata.get("split") or split
    if dataset_path:
        dataset = get_dataset_from_path(dataset_path, split=dataset_split)
    else:
        dataset = get_dataset(args, split=split)

    record_index = metadata.get("simpletir_record_index", metadata.get("index"))
    if record_index is not None:
        try:
            return dataset.get_record(extra_index=int(record_index))
        except KeyError:
            pass
    row_id = sample.index % len(dataset)
    return dataset.get_record(row_id=row_id)


def compute_rule_reward(record: SimpleTIRRecord, sample: Sample) -> Optional[float]:
    """Compute rule-based reward using the math answer checker when possible."""
    if record.reward_style != "rule" or record.ground_truth is None:
        return None

    if extract_boxed_answer is None or grade_answer_verl is None:
        return None

    predicted = sample.response or ""
    if extract_boxed_answer is not None:
        extracted = extract_boxed_answer(sample.response)
        if extracted:
            predicted = f"\\boxed{{{extracted}}}"
    try:
        return float(int(grade_answer_verl(predicted, record.ground_truth)))
    except Exception:  # noqa: BLE001
        return 0.0


async def async_reward(args, sample_or_samples: Sample | Iterable[Sample], *, split: str = "train"):
    """Async reward hook compatible with ``custom_rm_path`` in slime."""
    if isinstance(sample_or_samples, Sample):
        sample = sample_or_samples
        record = _resolve_record(args, sample, split=split)
        reward = compute_rule_reward(record, sample)
        if reward is not None:
            return reward

        if record.ground_truth is None:
            return 0.0

        try:
            result = await compute_reward_async(
                record.data_source or "",
                sample.response,
                record.ground_truth,
                extra_info=record.extra_info,
            )
        except (ValueError, RuntimeError):
            return 0.0

        extra = result.get("extra_info") if isinstance(result, dict) else None
        if extra:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            if metadata is None:
                metadata = {}
            metadata.setdefault("reward_extra_info", extra)
            sample.metadata = metadata

        score = result["score"] if isinstance(result, dict) else float(result)
        return float(score)

    rewards: list[float] = []
    for sample in sample_or_samples:
        record = _resolve_record(args, sample, split=split)
        reward_val = compute_rule_reward(record, sample)
        if reward_val is not None:
            rewards.append(reward_val)
            continue

        if record.ground_truth is None:
            rewards.append(0.0)
            continue

        try:
            result = await compute_reward_async(
                record.data_source or "",
                sample.response,
                record.ground_truth,
                extra_info=record.extra_info,
            )
        except (ValueError, RuntimeError):
            rewards.append(0.0)
            continue

        extra = result.get("extra_info") if isinstance(result, dict) else None
        if extra:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            if metadata is None:
                metadata = {}
            metadata.setdefault("reward_extra_info", extra)
            sample.metadata = metadata

        score = result["score"] if isinstance(result, dict) else float(result)
        rewards.append(float(score))
    return rewards
