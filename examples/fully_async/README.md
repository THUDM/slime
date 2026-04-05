# Fully Asynchronous Rollout Example

This example shows a simple way to make rollout generation **fully asynchronous**: a single global worker is created once and then keeps running in the background, continuously pulling prompts and launching generation tasks. Training only needs to fetch already finished results. This removes the per‑step wait that happens in the normal synchronous style.

## Files
* `fully_async_rollout.py`: global async worker + `generate_rollout_fully_async` entry, including off-policy buffer management.
* `run-qwen3.5-4b-off-policy-benchmark.sh`: multi-mode off-policy benchmark script supporting one-step-off baseline, fully async, staleness-backpressure, and window-evict modes.

## Prerequisite
First set up model & environment following the Qwen3.5-4B example.

## Quick Start

**Off-policy benchmark (4 modes):**
```bash
# One-step off-policy async baseline (default rollout, no fully async worker)
MODE=one_step_off      bash examples/fully_async/run-qwen3.5-4b-off-policy-benchmark.sh

# Fully async, no staleness control
MODE=fully_async       bash examples/fully_async/run-qwen3.5-4b-off-policy-benchmark.sh

# Fully async + staleness backpressure + partial rollout
MODE=staleness_partial bash examples/fully_async/run-qwen3.5-4b-off-policy-benchmark.sh

# Fully async + version-window eviction + partial rollout
MODE=window_partial    bash examples/fully_async/run-qwen3.5-4b-off-policy-benchmark.sh
```

You should see log lines like:
```
Creating new global async worker...
Continuous async rollout worker started
```

## How It Works

* First call: create `AsyncRolloutWorker` (thread + asyncio loop).
* Loop keeps up to `--rollout-batch-size` tasks in flight using `generate_and_rm_group`.
* Completed groups are pushed into a `CompletedSampleRecord` store; caller drains until it has enough samples.
* Worker is stopped automatically at process exit.

### Why do we need both staleness control and version-window eviction?

The two existing async modes (`one_step_off` and `fully_async`) both lack the ability to control off-policy staleness and neither supports **partial rollout**:

- **`one_step_off`**: uses the default `sglang_rollout` path. Although `sglang_rollout.py` internally implements `abort()` and partial-rollout recycling, the original `train_async.py` did not have `before_weight_update` / `after_weight_update` lifecycle hooks, so the training loop never notified the rollout module before a weight sync. In-flight tasks were simply drained to completion, making partial rollout impossible.
- **`fully_async`**: the original async worker had no concept of policy version tracking, no staleness budget, no `abort()` call, and no weight-update hooks. The worker ran continuously without any coordination with weight updates, so partial rollout was equally unsupported.

The first fix is `staleness_partial`: it adds policy-version tracking, a stale-sample budget, and the lifecycle hooks needed by partial rollout. The staleness/backpressure idea here is close to the fully async design used in VERL.

With the new lifecycle hooks (`before_weight_update` / `after_weight_update`) wired into `train_async.py` and `RolloutManager`, the async worker can now abort in-flight tasks before each weight update, recycle partially generated samples back to the data buffer, and mask off-policy tokens during training.

However, staleness backpressure still has two practical limitations:

1. If rollout throughput is lower than training consumption throughput, pausing new scheduling can introduce rollout bubbles and make the rollout side fall further behind.
2. When partial rollout is enabled, a common strategy is to prioritize recycled samples so they are resumed first. That improves reuse of partial work, but it also means a single trajectory may span many policy versions, so the `version span` can still lag by much more than 1 even if the stale backlog is bounded.

That is why `window_evict` is introduced after staleness control. If you want to strictly cap the allowed version lag, for example keep it `<= 1`, while also avoiding rollout pauses when rollout is faster than training, `window_evict` is a better fit. Its sliding-version-window eviction behavior is mainly inspired by MiniMax Forge.

### Off-Policy Buffer Policies

In fully async mode, the rollout worker runs continuously and may produce samples generated under an older policy version. Two buffer policies control how these **stale (off-policy) samples** are managed:

#### Buffer Policy Comparison

| Feature | `legacy_backpressure` | `window_evict` |
|---------|----------------------|----------------|
| **Scheduling** | Pauses when stale budget reached | Never pauses, always scheduling |
| **Sample Eviction** | No eviction  | Actively evicts out-of-window samples |
| **GPU Utilization** | May have idle periods | Always high utilization |
| **Version Lag Control** | Soft control (backlog ratio) | Hard control (window width W) |
| **Partial Rollout Span** | May span many versions | Bounded to ≤ W+1 versions |
| **Key Parameter** | `--staleness-threshold` | `--fully-async-version-window` |

#### `legacy_backpressure` (default; used by `staleness_partial`)

Pause scheduling new rollout tasks when the number of stale samples reaches a configurable budget:

```
budget = rollout_batch_size × update_weights_interval × (1 + staleness_threshold)
```

The worker resumes after the trainer consumes enough samples to bring the stale count below the budget. This is the simpler staleness-control mode, but pausing can leave rollout GPUs idle and it does not strictly bound per-trajectory version span under partial rollout.

#### `window_evict` (used by `window_partial`)

Keep rollout scheduling active at all times. Instead of pausing, evict completed samples whose policy version falls outside a sliding window `[current_version - W, current_version]`. This trades sample efficiency (some generated samples are discarded) for higher GPU utilization and a stricter bound on allowed version lag.

Key parameters:
- `--fully-async-version-window W`: window width (default 1).
- `--fully-async-max-completed-samples N`: hard cap on buffered samples.
- `--fully-async-eviction-policy`: `drop_oldest_version` (default) or `drop_oldest_fifo`.

### Partial Rollout & Off-Policy Masking

When `--partial-rollout` is enabled, in-flight rollout tasks are **aborted** before each weight update rather than drained to completion. The partially generated samples are returned to the data buffer and re-scheduled under the new policy.

Combined with `--mask-offpolicy-in-partial-rollout`, any trajectory whose generation spans multiple policy versions will have its off-policy tokens masked during training loss computation, ensuring that only on-policy tokens contribute to gradient updates.

### Lifecycle Hooks

The training loop (`train_async.py`) calls `RolloutManager.before_weight_update` / `after_weight_update` around each weight sync. These hooks are forwarded to module-level functions in the rollout module (`before_weight_update`, `after_weight_update` in `fully_async_rollout.py`), enabling the async worker to:
1. Pause scheduling and drain/abort in-flight tasks before weights change.
2. Update the internal policy version, evict out-of-window samples, and resume after weights are synced.
3. Report per-interval staleness and eviction metrics to wandb.

## New CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--staleness-threshold` | float | None | Max stale backlog ratio. Enables backpressure when set. |
| `--fully-async-buffer-policy` | str | `legacy_backpressure` | Buffer policy: `legacy_backpressure` or `window_evict`. |
| `--fully-async-version-window` | int | 1 | Policy-version window width for `window_evict`. |
| `--fully-async-max-completed-samples` | int | auto | Hard cap on completed samples in memory. |
| `--fully-async-eviction-policy` | str | `drop_oldest_version` | Overflow eviction strategy for `window_evict`. |
| `--fully-async-debug-version-tracking` | flag | False | Print per-batch version summaries for debugging. |

## Wandb Metrics

When enabled, the following metric groups are logged under a dedicated `fully_async/step` axis:

- `fully_async/count/*`: stale samples processed, consumed, recycled, dropped.
- `fully_async/partial/*`: partial rollout ratio and max version span.
- `fully_async/window/*`: completed store size, eligible samples, eviction counts.

## Config Differences (2 Key Points)
To enable the fully async pattern there are only two changes compared to a normal run:

1. Use the async training driver: `train_async.py` (not `train.py`).
2. Set the rollout function path:
	```bash
	--rollout-function-path fully_async_rollout.generate_rollout_fully_async
	```

Why is it still "fully" async although `train_async.py` itself schedules rollouts step‑by‑step?

Because the real generation work is done by a **persistent background worker** created in `generate_rollout_fully_async`. Each call from `train_async.py` only drains already completed samples from the worker's output queue; the worker has been continuously generating since the first call. Thus rollout production (model inference) and training consume happen in parallel with minimal waiting.
