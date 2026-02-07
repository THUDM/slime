# Fully Asynchronous Colocate Rollout

This example demonstrates **fully-async generation on shared GPUs** (colocate mode): training and inference share the same GPUs, with asynchronous rollout generation that overlaps with the collection process.

## The Problem

In standard colocate mode, rollout generation is **synchronous**: the training loop waits for all samples to be generated before proceeding. GPU utilization during generation is limited by the sequential nature of batch collection.

## The Solution

We combine two ideas:

1. **Fully-async generation**: A persistent background worker continuously generates samples, maximizing GPU utilization during inference.
2. **Colocate offload/onload**: Training and inference share GPUs. When enough samples are collected, we offload inference and train; when training finishes, we offload training and resume inference.

### Handling Timeouts

The main challenge is that **training can take hundreds of seconds**. During this time, sglang engines are offloaded and any in-flight HTTP requests would fail or time out.

Our solution: **abort all in-flight requests before offloading**. The `pause()` method:
1. Sets a pause flag so the background worker stops submitting new tasks.
2. Calls sglang's `/abort_request` API to cancel all pending generation requests.
3. Waits for all `asyncio` tasks to settle (they return quickly after abort).
4. Successfully completed samples are kept; aborted ones are returned to the data buffer for retry.

This ensures there are zero dangling requests when we offload the engines.

## Files

| File | Description |
|------|-------------|
| `fully_async_colocate_rollout.py` | Background async worker with pause/resume + rollout entry point |
| `run-qwen3-4b-fully_async_colocate.sh` | Example launch script for Qwen3-4B |

This example reuses the standard `train.py` — the colocate offload/onload cycle is already handled by `--colocate --offload` flags.

## Prerequisites

Set up model & environment following the [Qwen3-4B example](../../docs/en/examples/qwen3-4B.md).

## Quick Start

```bash
cd slime
bash examples/fully_async_colocate/run-qwen3-4b-fully_async_colocate.sh
```


## Key Differences from `fully_async`

| Aspect | `fully_async` | `fully_async_colocate` |
|--------|---------------|------------------------|
| GPU sharing | Separate GPUs for train/inference | Same GPUs (colocate) |
| Worker lifecycle | Always running | Pause during train, resume after |
| Offloading | N/A | Offload rollout before train, offload train before rollout |
| Request management | Worker runs through training | Abort in-flight requests before offload |
| Training driver | `train_async.py` | `train_fully_async_colocate.py` |
| `--colocate` flag | Not required | Required |


## Limitations

- No evaluation mode during async rollout (eval happens synchronously between steps).

## Acknowledgements

This fully-async colocate approach was proposed and developed by the Roll team. We thank them for their contribution to improving GPU utilization in shared training/inference scenarios.
