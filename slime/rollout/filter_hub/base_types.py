from collections import defaultdict
from dataclasses import dataclass


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None = None
    # Keep a rejected group when dropping it would leave too few candidates to
    # fill the rollout batch. This avoids launching another oversampling round.
    keep_when_insufficient: bool = False


def should_drop_dynamic_filter_output(
    output: DynamicFilterOutput,
    *,
    remaining_batch_size: int,
    target_data_size: int,
) -> bool:
    if output.keep:
        return False
    if output.keep_when_insufficient and remaining_batch_size <= target_data_size:
        return False
    return True


def call_dynamic_filter(fn, *args, **kwargs):
    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(*args, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        return {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
