# Fully-Async Rollout Example

End-to-end demo of slime's fully-async rollout path. A background asyncio
worker keeps a fixed pool of in-flight generations across rollout boundaries,
so the next training step doesn't wait for the slowest in-flight sample.
The worker itself lives in `slime.rollout.fully_async_rollout`; this
directory is just the launch script + CI test.

## Files

* `run-qwen2.5-0.5B-fully_async.sh` — single-node, 4-GPU, three-rollout demo
  with Qwen2.5-0.5B-Instruct on dapo-math-17k. Fast enough to be the CI
  smoke test for the fully-async path.
* `run-qwen3.5-9B-fully_async.sh` — single-node, 8-GPU, three-rollout demo
  with Qwen3.5-9B on dapo-math-17k.

The same script doubles as `tests/test_qwen2.5_0.5B_fully_async_short.py` in
CI.

## Prerequisites

```
/root/models/Qwen2.5-0.5B-Instruct/            # HF checkpoint
/root/models/Qwen2.5-0.5B-Instruct_torch_dist/ # tools/convert_hf_to_torch_dist.py
/root/datasets/dapo-math-17k/dapo-math-17k.jsonl
```

## Run

```bash
cd slime
bash examples/fully_async/run-qwen2.5-0.5B-fully_async.sh
```

You should see:

```
fully-async rollout 0: target=8 queue_warm=0
fully-async rollout 0: done in ...s, queue_left=...
```

## How To Plug Your Own Generate Into This

Fully-async uses the standard `python3 train.py` entrypoint. Select the
rollout implementation with:

```
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
```

For custom per-sample logic, use slime's standard plug-in points — they
work unchanged under fully-async:

```
--custom-generate-function-path your.module.generate     # (args, sample, sampling_params) -> Sample | list[Sample]
--custom-rm-path                your.module.reward      # (args, sample | list[Sample]) -> float | list[float]
```

See `examples/coding_agent_rl/` for a non-trivial example that plugs in a
multi-turn agent (Claude Code in a Docker-Proxy sandbox) this way.

## Worker Internals (Very Short)

* First call: create a process-wide `AsyncRolloutWorker` (thread + asyncio
  loop). The worker is shared across all subsequent `generate_rollout`
  calls so its queue stays warm.
* Loop keeps up to `args.sglang_server_concurrency` tasks in flight using
  `generate_and_rm_group`.
* Completed groups land on an output queue; each `generate_rollout` call
  drains until it has `rollout_batch_size` groups and returns them sorted
  by `sample.index`.
* Groups containing an `ABORTED` sample are pushed back into
  `data_buffer.add_samples` instead of being shipped to training.
* Evaluation (`--eval-interval`) runs on the worker's event loop, sharing the
  engines with the in-flight trajectories. Leave `--eval-function-path` unset:
  pointing it at `slime.rollout.sglang_rollout.generate_rollout` would run
  evaluation on a second event loop against the same loop-bound
  `GenerateState`.
* Worker is stopped automatically at process exit via `atexit`.

## Trajectories Across Weight Updates

A weight update pauses the SGLang engines, which aborts every in-flight
request. The aborted sample keeps its partial response and per-token rollout
log-probs; when its group is picked up again, generation continues from that
prefix with the remaining token budget under the new weights. A finished
sample can therefore contain tokens from several policy versions, recorded in
`Sample.weight_versions` and summarized by the `rollout/staleness/*` metrics
(`multi_version_frac` is the fraction of such samples). Either correct those
tokens with importance sampling (`--use-tis`, e.g. the FlashREINFORCE trust
region `slime.backends.megatron_utils.loss.binary_kl_trust_region_function`)
or keep only the latest segment with
`--partial-rollout --mask-offpolicy-in-partial-rollout`.

This continuation applies to slime's `generate` and `generate_streaming`, for
text and multimodal samples (a resumed multimodal request resends the
processor-expanded prompt ids plus the partial response with its images).

A custom multi-turn generate function should produce each model turn with
`slime.rollout.sglang_rollout.generate_turn(args, sample, sampling_params)`.
It sends `sample.tokens` (prompt, earlier turns and tool tokens appended with
`trainable=False`), appends the turn with its log-probs, weight versions and
routed experts (`--use-rollout-routing-replay`), and resends a turn cut by a
weight update with its partial response, so the turn continues under the new
weights while the environment and the earlier turns stay as they are.
Restarting the whole trajectory on an abort instead can livelock once
trajectories outlast the interval between weight updates. The pooled worker
(below) drops a rollout that comes back `ABORTED` three times in a row without
new tokens, which is how it treats a generate function reporting a failure.

## Forming Batches Like molt

`--fully-async-pool-size N` switches to `PooledRolloutWorker`, which forms
batches the way NVIDIA molt's asynchronous trainer does:

* The pool holds `N` rollout groups, counting the ones still generating and
  the finished ones not yet in a batch.
* A batch is formed only while one of `--fully-async-max-queued-batches`
  slots is free (default 1). Training frees a slot when it takes a batch,
  before training on it, so at most that many batches are formed ahead.
* While forming a batch, the worker refills the pool to `N` and takes one
  finished group, until the batch holds `rollout_batch_size` groups. When
  several groups have finished, it takes them in a random order fixed at
  dispatch (molt's `ray.wait` returns an arbitrary ready one). Between
  batches nothing is dispatched, so when training is the bottleneck the pool
  fills with finished groups and the engines idle.
* A group aborted by a weight update is resent at once and keeps its place in
  the pool: SGLang holds the request until the engines continue.
* `--fully-async-drain-each-epoch` stops refilling once every prompt of the
  epoch is dispatched, drains the pool into a smaller last batch, and starts
  the next epoch with the following batch. The smaller batch is trained as one
  update: its loss is averaged over its own samples and the learning-rate
  schedule advances one step.
* Batch formation pauses while an evaluation runs.

Keep `--sglang-server-concurrency` times the number of engines at least `N`,
so that the request semaphore does not cap the pool.
[`examples/flash_reinforce`](../flash_reinforce/README.md) uses this mode.

## Limitations

* Distributed fully-async (`--rollout-data-transport straw`) does not support
  evaluation or `--fully-async-pool-size` yet.
* Ordering across rollouts is best-effort — within a rollout, groups are
  sorted by index before being handed to training.
