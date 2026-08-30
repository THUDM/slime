# Code Structure

This document describes the code structure and implementation details of the curriculum learning system.

## File Structure

```
examples/curriculum_learning/
├── README.md                          # Usage guide
├── RESULTS.md                         # Experiment results
├── data_source.py                     # Multi-source data manager
├── weight_scheduler.py                # Pre-defined weight functions
├── reward_cap_scheduler.py            # Pre-defined reward cap functions
├── reward_cap_filter.py               # Dynamic sampling filter
├── reward.py                          # Custom reward functions
├── custom_log.py                      # Per-source logging
├── run_math_if_qwen3_8b.sh           # Launch script
├── configs/
│   ├── math_if_train_config.yaml     # Training configuration
│   └── math_if_eval_config.yaml      # Evaluation configuration
├── docs/
│   ├── CODE_STRUCTURE.md             # This file - code structure documentation
│   ├── dynamic_weights.md            # Dynamic weights documentation
│   └── reward_cap_filter.md          # Reward filtering documentation
└── data/
    ├── download_data.md              # Data download instructions
    ├── dapo-math-17k.jsonl          # (download) Math training data
    ├── verinstruct.jsonl            # (download) IF training data
    ├── aime-2024.jsonl              # (download) Math eval data
    └── IFBench_eval.jsonl           # (download) IF eval data
```

## Core Components

### 1. data_source.py

**Class: `MultipleWeightedRolloutDataSourceWithBuffer`**

Manages sampling from multiple datasets with dynamic weights.

**Key Methods:**
- `get_samples()` - Evaluates weights, distributes samples across sources, tags with metadata
- `add_samples()` - Routes samples back to source buffers for partial rollout
- `save()/load()` - Checkpoint support with per-source buffers and global state

**Weight Parsing:** Supports constants, lambda strings, and function paths.

### 2. weight_scheduler.py

Pre-defined weight scheduler functions (signature: `(rollout_step: int) -> float`).

| Function | Range | Description |
|----------|-------|-------------|
| `decreasing_weight` | 0.9 → 0.1 | Linear decrease over 200 steps |
| `increasing_weight` | 0.1 → 0.9 | Linear increase over 200 steps |
| `exponential_decay` | 0.9 → 0.1 | Smooth exponential decrease |
| `exponential_warmup` | 0.1 → 0.9 | Smooth exponential increase |
| `level_increasing` | 0.5 → 0.8 | Discrete levels: 0.5, 0.6, 0.7, 0.8 |
| `level_decreasing` | 0.8 → 0.5 | Discrete levels: 0.8, 0.7, 0.6, 0.5 |

### 3. reward_cap_scheduler.py

Pre-defined reward cap scheduler functions (signature: `(rollout_step: int) -> float`).

| Function | Range | Description |
|----------|-------|-------------|
| `increasing_cap` | 0.1 → 0.9 | Linear increase (relax filtering) |
| `decreasing_cap` | 0.9 → 0.1 | Linear decrease (tighten filtering) |
| `exponential_increase` | 0.0 → 0.9 | Smooth exponential growth |
| `exponential_decrease` | 0.9 → 0.1 | Smooth exponential decay |
| `level_increasing` | 0.5 → 0.8 | Discrete levels: 0.5, 0.6, 0.7, 0.8 |
| `level_decreasing` | 0.8 → 0.5 | Discrete levels: 0.8, 0.7, 0.6, 0.5 |

### 4. reward_cap_filter.py

**Function: `check_reward_cap_per_source`**

Filters sample groups based on per-source reward caps:
1. Gets cap function for the sample's data source
2. Evaluates cap at current rollout step
3. Drops if all rewards are 0.0 or average reward exceeds cap
4. Returns `DynamicFilterOutput` with keep/drop decision

### 5. reward.py

**Function: `async_rm_math_if`**

Supports multiple reward types:
- **deepscaler** - Math reasoning (1.0 for correct, 0.0 for incorrect)
- **format_verify** - Instruction-following constraints (fraction satisfied)
- **ifbench** - IFBench evaluation

### 6. custom_log.py

**Function: `custom_rollout_log_function`**

Tracks per-source metrics:
- Weight curves, sample counts, rewards, response lengths
- Filter ratios (all-zero, reward cap) for dynamic sampling
- Groups samples by `data_source` metadata for independent tracking

## Training Loop

1. **Sample Selection** - Compute weights, distribute samples, tag metadata
2. **Rollout Generation** - Generate responses, compute rewards
3. **Dynamic Filtering** - Over-sample, drop by reward caps, keep `rollout_batch_size` groups
4. **Training** - Train policy on filtered samples
5. **Logging** - Track per-source metrics and curriculum progression
6. **Buffer Management** - Route samples back to source buffers

## Key Implementation Details

**Weight Normalization:** Weights automatically sum to 1.0.

**Sample Distribution:** Proportional allocation with rounding, remainder to last source.

**Group Index Management:** Per-source offsets ensure globally unique indices.

**Metadata Flow:** Samples carry `data_source`, `data_source_weight`, `rollout_step`, `rm_type` for routing and tracking.

**Checkpointing:** Per-source buffers + global state (offsets, rollout step, source names).

## Extending the System

**Add Weight Scheduler:**
```python
def my_scheduler(rollout_step: int) -> float:
    return weight_value  # Your logic here
```

**Add Reward Cap Scheduler:**
```python
def my_cap_scheduler(rollout_step: int) -> float:
    return cap_value  # Your logic here
```

**Add Reward Function:**
```python
# In reward.py
elif rm_type == "my_custom_reward":
    return reward_value
```

## Implementation Notes

- Weight/cap functions evaluated once per `get_samples()` call
- Cap functions cached after first parse
- Per-source buffers enable independent recycling, thus naturally supports slime's partial rollout implementation
- Checkpoint saves are per-source
