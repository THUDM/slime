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

* With the default object-store data source, the first call creates a
  manager-local `AsyncRolloutWorker` (thread + asyncio loop). The data source
  owns it across subsequent calls so its queue stays warm.
* With `--rollout-data-transport straw`, queue readers select
  `DistributedRollout`: one generation actor on each eligible Ray node with
  CPU resources. These actors request generations from the SGLang engines;
  they do not load additional model replicas.
* The concurrency budget is `sglang_server_concurrency` times the logical
  engine count, divided into prompt-group slots. Distributed workers share
  those slots and need at least one group per worker node.
* Completed groups land on an output queue; each `generate_rollout` call
  drains until it has `rollout_batch_size` groups and returns them sorted
  by `sample.index`.
* Groups containing an `ABORTED` sample are pushed back into
  `data_buffer.add_samples` instead of being shipped to training.
* The data source coordinates pause, checkpoint and close. Admission pauses
  around weight synchronization; in-flight work is preserved before resuming.

## Partial rollout and durable storage

Partial rollout is also useful with synchronous training. With
`--partial-rollout`, the standard sampler retains unfinished prefixes after
enough complete groups are accepted, then resumes them in a later rollout.
Oversampling supplies spare groups; for example use a target batch of 8 groups
and `--over-sampling-batch-size 16`. Completed groups form the training batch;
unfinished groups are saved for continuation.

Async workers rebuffer aborted groups through their own lifecycle. A continued
response can contain several weight versions. Optional
`--mask-offpolicy-in-partial-rollout` excludes existing prefix tokens from the
loss, but does not remove prefix-distribution mismatch.

straw persists tasks, partial/ready groups and packed tensors on shared storage.
For multi-node execution, configure a verified JuiceFS mount at the same path
on all nodes, along with `--rollout-data-dir`, `--rollout-queue-run-id`,
`--rollout-storage-profile juicefs` and `--rollout-storage-declaration`.
Recovery needs a completed model/optimizer and rollout checkpoint plus the
referenced data pool and compatible worker topology. See the
[scheduling tutorial](https://github.com/THUDM/slime/blob/main/docs/en/advanced/rollout-scheduling.md)
and [straw guide](https://github.com/THUDM/slime/blob/main/docs/en/advanced/straw.md).

## Limitations

* No evaluation mode (would conflict with the continuous-running model).
* Ordering across rollouts is best-effort — within a rollout, groups are
  sorted by index before being handed to training.
* Distributed fully async does not support `--rollout-all-samples-process-path`.
* straw does not support `--buffer-filter-path`; its persisted queue has its
  own completed/partial/fresh ordering and staleness priority.
* Restoring queue state does not restore GPU KV caches or guarantee identical
  future random generations. Custom generation hooks must preserve the
  continuation and abort contracts.
