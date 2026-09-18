"""Policy age in weight-update units (fresh samples have age one)."""

from slime.utils.types import Sample


def fully_async_metrics_enabled(args) -> bool:
    return "fully_async" in (getattr(args, "rollout_function_path", None) or "")


def sample_staleness(sample: Sample, current_weight_version: int) -> int | None:
    versions = sample.weight_versions or []
    if not versions or any(not str(v).isascii() or not str(v).removeprefix("-").isdigit() for v in versions):
        return None
    numeric_versions = [int(v) for v in versions]
    if max(numeric_versions) > current_weight_version:
        return None
    return current_weight_version + 1 - min(numeric_versions)


def compute_staleness_metrics(samples: list[Sample], current_weight_version: int | None) -> dict[str, float | int]:
    if current_weight_version is None:
        return {}
    ages = [sample_staleness(sample, current_weight_version) for sample in samples]
    known = [age for age in ages if age is not None]
    metrics: dict[str, float | int] = {"staleness/unknown_count": len(ages) - len(known)}
    if known:
        metrics["staleness/mean"] = sum(known) / len(known)
        metrics["staleness/max"] = max(known)
    return metrics
