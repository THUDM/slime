---
name: add-rollout-function
description: Guide for adding a new rollout function in slime and wiring it through --rollout-function-path. Use when user wants to implement custom rollout data generation logic, custom train/eval rollout outputs, or migrate from the default sglang rollout path.
---

# Add Rollout Function

Implement a custom rollout function and integrate it safely with slime training/eval flow.

## When to Use

Use this skill when:

- User asks to add a new rollout task or rollout generation function
- User asks to replace default `slime.rollout.sglang_rollout.generate_rollout`
- User asks to customize train/eval data generation behavior

## Step-by-Step Guide

### Step 1: Choose the Right Starting Point

Start from one of these references:

- Async RL-style rollout: `slime/rollout/sglang_rollout.py`
- Simple SFT-style rollout: `slime/rollout/sft_rollout.py`

If the task needs engine-based async generation and rewards, use the sglang path as base.
If the task is file/buffer-driven and simple, use sft path as base.

### Step 2: Create the New Rollout Module

Create a new file, for example: `slime/rollout/<your_rollout>.py`

Required callable signature:

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    ...
```

Return types are defined in `slime/rollout/base_types.py`.

### Step 3: Implement Train and Eval Branches Explicitly

- For training (`evaluation=False`), return `RolloutFnTrainOutput(samples=..., metrics=...)`
- For evaluation (`evaluation=True`), return `RolloutFnEvalOutput(data=..., metrics=...)`

Minimal skeleton:

```python
from slime.rollout.base_types import RolloutFnTrainOutput, RolloutFnEvalOutput


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        result = {
            "custom_eval": {
                "rewards": [],
                "truncated": [],
                "samples": [],
            }
        }
        return RolloutFnEvalOutput(data=result)

    groups = data_source.get_samples(args.rollout_batch_size)
    # fill Sample fields needed by training: tokens/response_length/reward/status (+ loss_mask when needed)
    return RolloutFnTrainOutput(samples=groups)
```

### Step 4: Keep Data Contract Compatible

For each generated sample, ensure required training fields are populated consistently with your objective:

- `tokens`
- `response_length`
- `reward` (or reward dict if your setup uses keyed rewards)
- `status`

If partial rollout or masking logic is involved, keep `loss_mask` semantics consistent with existing behavior.

### Step 5: Wire Through Arguments

Set your function path via CLI:

```bash
--rollout-function-path slime.rollout.<your_rollout>.generate_rollout
```

The default and signature expectation are documented in:

- `slime/utils/arguments.py`
- `docs/en/get_started/customization.md`

### Step 6: Validate End-to-End

- Run a closest existing script with your new function path.
- Run at least one targeted test under `tests/`.
- Confirm both training generation and eval path (if eval is enabled) execute without schema mismatch.

Suggested commands:

```bash
rg --files tests | rg 'rollout|debug_rollout|quick_start'
pytest "$(rg --files tests | rg 'rollout|debug_rollout|quick_start' | head -n 1)" -vv
```

## Common Mistakes

- Returning raw Python lists/dicts with mismatched schema in custom path
- Implementing only training branch and forgetting evaluation branch
- Generating samples without required fields (`tokens`, `response_length`, `reward`, `status`)
- Using blocking-heavy logic in high-frequency rollout paths without batching/concurrency control

## Reference Locations

- Default rollout: `slime/rollout/sglang_rollout.py`
- Simple custom example: `slime/rollout/sft_rollout.py`
- Output dataclasses: `slime/rollout/base_types.py`
- Wiring/loading: `slime/ray/rollout.py`
- Argument definition: `slime/utils/arguments.py`
- Customization docs: `docs/en/get_started/customization.md`
