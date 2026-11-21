from collections import defaultdict
from dataclasses import dataclass, replace


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None = None
    # Keep a rejected group when dropping it would leave too few candidates to
    # fill the rollout batch. This avoids launching another oversampling round.
    keep_when_insufficient: bool = False
    # Sum and count of trajectory rewards before the hook runs. Carry only
    # these scalars back from workers, including when rejected data is released.
    raw_reward_stats: tuple[float, int] | None = None


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


def call_dynamic_filter(fn, args, samples, **kwargs):
    reward_stats = None
    if fn is not None or getattr(args, "rollout_sample_filter_path", None) is not None:
        # Context-window segments share one outcome; count the terminal reward
        # once per trajectory, before a custom filter can mutate it.
        trajectories = [sample[-1] if isinstance(sample, list) else sample for sample in samples]
        if all(sample.reward is not None for sample in trajectories):
            rewards = [sample.get_reward_value(args) for sample in trajectories]
            reward_stats = (float(sum(rewards)), len(rewards))

    output = fn(args, samples, **kwargs) if fn is not None else True

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return replace(output, raw_reward_stats=reward_stats)


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._raw_reward_sum = 0.0
        self._raw_reward_count = 0
        self._raw_rewards_complete = True

    def on_group_completed(self, output: DynamicFilterOutput):
        if output.raw_reward_stats is None:
            # Older restored results may lack pre-filter observations. Do not
            # label a partial mean as the reward of the whole candidate batch.
            self._raw_rewards_complete = False
            return
        reward_sum, reward_count = output.raw_reward_stats
        self._raw_reward_sum += reward_sum
        self._raw_reward_count += reward_count

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        if self._raw_rewards_complete and self._raw_reward_count:
            metrics["rollout/raw_reward_before_filters"] = self._raw_reward_sum / self._raw_reward_count
        return metrics
