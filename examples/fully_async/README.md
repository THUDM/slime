## Fully Asynchronous Rollout Example

This example shows a simple way to make rollout generation **fully asynchronous**: a single global worker is created once and then keeps running in the background, continuously pulling prompts and launching generation tasks. Training only needs to fetch already finished results. This removes the per‑step wait that happens in the normal synchronous style.

### Files
* `fully_async_rollout.py`: global async worker + `generate_rollout_fully_async` entry.
* `run-qwen3-4b-fully_async.sh`: example launch script with Qwen3‑4B.

### Prerequisite
First set up model & environment following the Qwen3-4B example.

### Quick Start
```bash
cd slime
bash examples/fully_async/run-qwen3-4b-fully_async.sh
```
You should see log lines like:
```
Creating new global async worker...
Continuous async rollout worker started
```

### How It Works (Very Short)
* First call: create `AsyncRolloutWorker` (thread + asyncio loop).
* Loop keeps up to `--rollout-batch-size` tasks in flight using `generate_and_rm_group`.
* Completed groups are pushed into a queue; caller drains until it has enough samples.
* Worker is stopped automatically at process exit.

### Limitations
* No evaluation mode.
* Ordering is best effort (sorted at the end by index).
* Minimal error handling.

### Config Differences (2 Key Points)
To enable the fully async pattern there are only two changes compared to a normal run:

1. Use the async training driver: `train_async.py` (not `train.py`).
2. Set the rollout function path:
	```bash
	--rollout-function-path fully_async_rollout.generate_rollout_fully_async
	```

Why is it still "fully" async although `train_async.py` itself schedules rollouts step‑by‑step?

Because the real generation work is done by a **persistent background worker** created in `generate_rollout_fully_async`. Each call from `train_async.py` only drains already completed samples from the worker's output queue; the worker has been continuously generating since the first call. Thus rollout production (model inference) and training consume happen in parallel with minimal waiting.
