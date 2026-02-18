# Dynamic Data Source Weights

## Overview
The `MultipleWeightedRolloutDataSourceWithBuffer` class allows you to specify different weights for each data source that can change dynamically based on the rollout step. This enables curriculum learning strategies where the emphasis on different data sources evolves during training.

## Configuration Format

Each data source can have its weight specified in three ways:

### 1. Constant Weight
```yaml
data_source_weights:
  - 0.5  # constant
  - 0.5  # constant
```

### 2. Lambda Function
```yaml
data_source_weights:
  - "lambda step: 0.9 - 0.8 * min(step / 200, 1.0)"  # decreasing
  - "lambda step: 0.1 + 0.8 * min(step / 200, 1.0)"  # increasing
```

### 3. Function from File
```yaml
data_source_weights:
  - "examples.curriculum_learning.weight_scheduler.easy_to_hard"
  - "examples.curriculum_learning.weight_scheduler.hard_to_easy"
```

### 4. Mixed Approaches
```yaml
data_source_weights:
  - 0.3  # constant for source 1
  - "lambda step: 0.5 + step/2000"  # lambda for source 2
  - "examples.curriculum_learning.weight_scheduler.hard_to_easy"  # function for source 3
```

## Available Scheduler Functions

See `weight_scheduler.py` for predefined functions:

- **easy_to_hard**: Decreasing weight (0.9 → 0.1 over 200 steps)
- **hard_to_easy**: Increasing weight (0.1 → 0.9 over 200 steps)
- **exponential_decay**: Smooth exponential decrease
- **exponential_warmup**: Smooth exponential increase
- **level_increasing**: Discrete increasing levels (0.5 → 0.6 → 0.7 → 0.8 over 200 steps)
- **level_decreasing**: Discrete decreasing levels (0.8 → 0.7 → 0.6 → 0.5 over 200 steps)

## Logging

The `custom_rollout_log_function` automatically logs the weight for each data source at each rollout step:

- `rollout/{source_name}/weight` - Current weight for each data source
- `rollout/{source_name}/sample_count` - Number of samples from each source
- `rollout/{source_name}/response_len/*` - Response length statistics per source
- `rollout/{source_name}/raw_reward/*` - Reward statistics per source

These curves allow you to visualize how the curriculum learning schedule is progressing.

## Creating Custom Weight Functions

To create your own weight scheduler:

```python
# In weight_scheduler.py or your own file

def my_custom_schedule(rollout_step: int) -> float:
    """
    Custom weight schedule for one data source.

    Args:
        rollout_step: Current training rollout step

    Returns:
        Single float weight for this data source
    """
    # Your custom logic here
    if rollout_step < 100:
        return 0.9
    elif rollout_step < 500:
        return 0.5
    else:
        return 0.1
```

Then use it in your config:
```yaml
data_source_weights:
  - "path.to.your.module.my_custom_schedule"
  - 0.5  # other source with constant weight
```

## Architecture

- **data_source.py**: `MultipleWeightedRolloutDataSourceWithBuffer` class
  - Manages multiple data sources with independent weights
  - Parses weight configurations (constant/lambda/function)
  - Tracks rollout step for dynamic weight computation
  - Distributes samples according to computed weights

- **weight_scheduler.py**: Pre-defined weight scheduler functions
  - Each function takes `rollout_step` and returns a single float
  - Can be referenced from config files

- **custom_log.py**: Custom logging function
  - Logs per-source metrics and weight curves
  - Parses and evaluates weight functions for logging

## Notes

- Weights are normalized automatically (sum to 1.0)
- Each data source maintains its own buffer independently, which also supports partial rollouts based on data-source separate buffers.
- Sample group indices are kept globally unique across sources
- Checkpoint saving/loading preserves rollout step and source states
