---
name: add-dynamic-filter
description: Guide for adding dynamic/filter hooks in slime rollout pipeline. Use when user wants sample-group selection during rollout, buffer filtering before training, or per-sample masking/processing hooks.
---

# Add Dynamic Filter

Add filtering hooks in rollout and buffer stages while preserving sample-group contracts.

## When to Use

Use this skill when:

- User asks to filter sample groups during dynamic sampling
- User asks to customize buffer extraction strategy
- User asks to mask/remove some rollout samples before training
- User asks to process all generated samples for logging/analysis

## Step-by-Step Guide

### Step 1: Pick the Correct Hook

- Dynamic sampling filter: `--dynamic-sampling-filter-path`
- Buffer filter: `--buffer-filter-path`
- Per-sample rollout filter: `--rollout-sample-filter-path`
- All-samples post process: `--rollout-all-samples-process-path`

### Step 2: Implement the Function Signature

Dynamic sampling filter (called in `slime/rollout/sglang_rollout.py`):

```python
def filter_function(args, samples, **kwargs):
    # return DynamicFilterOutput or bool
```

Preferred return type:

```python
from slime.rollout.filter_hub.base_types import DynamicFilterOutput

return DynamicFilterOutput(keep=True, reason=None)
```

Buffer filter (called in `slime/rollout/data_source.py`):

```python
def buffer_filter(args, rollout_id, buffer, num_samples):
    return selected_groups
```

Rollout sample filter:

```python
def rollout_sample_filter(args, samples):
    # modify sample.remove_sample in-place where needed
```

All-samples process:

```python
def process_all_samples(args, all_samples, data_source):
    ...
```

### Step 3: Preserve Group Structure

- Keep `list[list[Sample]]` structure intact where required.
- Do not flatten sample groups unless downstream path expects flattened samples.
- For sample removal, prefer `sample.remove_sample=True` over deleting objects.

### Step 4: Wire and Validate

Example wiring:

```bash
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
--buffer-filter-path <module>.buffer_filter
--rollout-sample-filter-path <module>.rollout_sample_filter
--rollout-all-samples-process-path <module>.process_all_samples
```

Run targeted validation:

```bash
rg --files tests | rg 'rollout|debug_rollout'
pytest "$(rg --files tests | rg 'rollout|debug_rollout' | head -n 1)" -vv
```

### Step 5: Measure Side Effects

- Check final sample count remains aligned with `rollout_batch_size` expectations.
- Verify drop reasons are surfaced in rollout metrics when dynamic filter is used.
- Validate training still receives valid loss masks/rewards after filtering.

## Common Mistakes

- Returning wrong container type for buffer filter
- Dropping samples by deletion instead of mask flagging
- Losing sample-group alignment in group-RM setup
- Adding expensive logic in hot filtering paths

## Reference Locations

- Dynamic filter types: `slime/rollout/filter_hub/base_types.py`
- Dynamic filter example: `slime/rollout/filter_hub/dynamic_sampling_filters.py`
- Rollout generation hook points: `slime/rollout/sglang_rollout.py`
- Buffer filter hook point: `slime/rollout/data_source.py`
