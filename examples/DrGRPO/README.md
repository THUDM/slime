# Dr.GRPO Custom Reducer

This example demonstrates how to use a custom reducer function for Dr.GRPO algorithm.

## Overview

By default, slime divides the policy gradient loss by the number of effective tokens in each sample. This custom implementation allows you to divide by a constant value (default: 1000) instead.

## Usage

Use `--custom-pg-loss-reducer-function-path` to apply the custom reducer **only to pg_loss**, while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean:

```bash
--custom-pg-loss-reducer-function-path examples.Dr.GRPO.custom_reducer.get_pg_loss_reducer
```

## Customization

You can modify the `DIVISOR` constant in `custom_reducer.py` to use a different value:

```python
# In custom_reducer.py
DIVISOR = 1000.0  # Change this to your desired constant
```

## How It Works

The custom function has the same signature as the default `get_sum_of_sample_mean`:

```python
def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
```

Instead of dividing by `loss_mask_i.sum()` (the number of effective tokens), it divides by the constant `DIVISOR`.

## Example

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --custom-pg-loss-reducer-function-path examples.Dr.GRPO.custom_reducer:get_pg_loss_reducer
   # ... other arguments
)
```

