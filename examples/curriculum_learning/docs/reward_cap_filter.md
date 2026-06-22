# Per-Data-Source Reward Cap Filter

## Overview

This feature enables fine-grained quality control during dynamic sampling by setting different reward caps for different data sources. Sample groups that exceed their source's average reward cap are dropped during rollout, allowing you to be more selective with easier datasets while being more lenient with harder ones.

## How It Works

During rollout generation with dynamic sampling:
1. For each sample group, check if all rewards are 0.0 - if so, drop the group (no learning signal)
2. Compute the average reward across all samples inside a group
3. Compare against the configured cap for this group's data source
4. Drop the group if avg_reward > cap
5. Otherwise, keep the group for training

## Configuration

### Basic Configuration (Static Caps)

Add these fields to your training config YAML:

```yaml
prompt_data_source_names:
  - dapo_math_17k
  - verinstruct

data_source_reward_caps:
  - 0.3  # dapo_math_17k: drop groups if average reward >0.3
  - 0.5  # verinstruct: drop groups if average reward >0.5

over_sampling_batch_size: 256
dynamic_sampling_filter_path: examples.curriculum_learning.reward_cap_filter.check_reward_cap_per_source
```

**Important**: The order in `data_source_reward_caps` must match `prompt_data_source_names`.

### Dynamic (Step-wise) Caps

Reward caps can change over time based on rollout step for curriculum learning.
Each cap can be:
1. **Constant**: `0.5`
2. **Lambda function**: `"lambda step: 0.1 + 0.5 * min(step / 200, 1.0)"`
3. **Function path**: `"examples.curriculum_learning.reward_cap_scheduler.increasing_cap"`

Example with dynamic caps:

```yaml
data_source_reward_caps:
  - "lambda step: 0.1 + 0.4 * min(step / 200, 1.0)"
  - "examples.curriculum_learning.reward_cap_scheduler.decreasing_cap"
```

**Available scheduler functions** (see [reward_cap_scheduler.py](../reward_cap_scheduler.py)):
- `increasing_cap`: 0.1 → 0.9 over 200 steps (start strict, become lenient)
- `decreasing_cap`: 0.9 → 0.1 over 200 steps (start lenient, become strict)
- `exponential_increase`: smooth exponential growth over 200 steps
- `exponential_decrease`: smooth exponential decay over 200 steps
- `level_increasing`: discrete increasing levels (0.5 → 0.6 → 0.7 → 0.8 over 200 steps)
- `level_decreasing`: discrete decreasing levels (0.8 → 0.7 → 0.6 → 0.5 over 200 steps)

## Monitoring

The filter logs drop reasons that appear in WandB metrics:
- `rollout/dynamic_filter/drop_all_zero_{source}`: Count of groups dropped because all rewards were 0.0
- `rollout/dynamic_filter/drop_reward_cap_{source}`: Count of groups dropped for exceeding reward cap

Per-source metrics from `custom_log.py` help you monitor:
- `rollout/{source}/raw_reward/mean`: Average reward per source
- `rollout/{source}/sample_count`: Number of samples kept per source
- `rollout/{source}/filter_ratio_all_zero`: Ratio of groups dropped due to all zero rewards (filtered / final_samples)
- `rollout/{source}/filter_ratio_reward_cap`: Ratio of groups dropped due to reward cap (filtered / final_samples)
- `rollout/{source}/reward_cap`: **Current reward cap value** for this source (curriculum progression curve for dynamic caps)
- `rollout/{source}/weight`: Current sampling weight for this source

## Implementation Details

**File**: [reward_cap_filter.py](../reward_cap_filter.py)

The filter is stateless and makes independent decisions for each sample group:
- No state tracking across groups
- Simple comparison: `avg_reward > cap`
- Uses pre-computed rewards (no recalculation needed)
- Asserts all samples in group have same data source
- Works with existing dynamic sampling infrastructure

**Integration**: Uses the standard dynamic sampling filter interface:
- Input: `(args, samples: list[Sample]) -> DynamicFilterOutput`
- Output: `DynamicFilterOutput(keep=bool, reason=str)`

## Combining with Other Filters

This filter can work alongside other filters like `check_reward_nonzero_std`. To combine filters, you would need to create a composite filter function that calls both.

## Visualizing Curriculum Progression

When using dynamic caps, WandB will log the curriculum curves:
- **`rollout/{source}/weight`**: Shows how sampling weights change over time
- **`rollout/{source}/reward_cap`**: Shows how reward caps evolve over time

Plot these together to visualize your curriculum strategy:
- Increasing `reward_cap` = gradually accepting more samples (relaxing filter)
- Decreasing `reward_cap` = gradually accepting fewer samples (tightening filter)
- Combined with `weight` changes = full curriculum visualization

## Tips

1. **Start conservative**: Use higher caps (0.6-0.8) initially to ensure you have enough training data
2. **Monitor drop rates**: Check `rollout/{source}/filter_ratio_reward_cap` metrics to ensure you're not dropping too many groups
3. **Adjust per source**: Easier datasets typically need lower caps, harder datasets need higher caps
4. **Balance with weights**: Combine with dynamic data source weights for full curriculum learning control
5. **Dynamic caps**: Use step-wise caps to gradually adjust filtering strictness as model improves
