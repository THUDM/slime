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

Two pieces flip the standard pipeline into fully-async:

1. Use the async training driver: `python3 train_async.py` (not `train.py`).
2. Set the rollout function path:
   ```
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   ```

For custom per-sample logic, use slime's standard plug-in points — they
work unchanged under fully-async:

```
--custom-generate-function-path your.module.generate     # (args, sample, sampling_params) -> Sample | list[Sample]
--custom-rm-path                your.module.reward      # (args, sample | list[Sample]) -> float | list[float]
```

See `examples/swe_codex/` for a non-trivial example that plugs in a
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
* Worker is stopped automatically at process exit via `atexit`.

## Limitations

* No evaluation mode (would conflict with the continuous-running model).
* Ordering across rollouts is best-effort — within a rollout, groups are
  sorted by index before being handed to training.
* TODO: partial-rollout-style resume for `ABORTED` trajectories is not
  yet wired; for now the trajectory is re-queued and starts over.
