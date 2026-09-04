# Trainer Communication Timeline

slime can write a low-overhead JSONL timeline for training and weight synchronization. It is disabled by default and does not change collective ordering or add synchronization to the training path.

Enable it with a per-rank path:

```bash
python train.py \
  ... \
  --communication-timeline /path/to/traces/slime-{role}-{rank}.jsonl
```

The equivalent environment variable is `SLIME_COMMUNICATION_TIMELINE`. Paths support `{rank}`, `{trainer_rank}`, `{local_rank}`, `{pid}`, `{hostname}`, `{role}`, and `{world_size}`. If a multi-rank path has no rank placeholder, slime adds a `.rank-N` suffix. Actor, critic, and disk-orchestrator roles also get separate suffixes when `{role}` is omitted. This works for any world size; there is no topology-specific rank layout. slime generates one shared `run_id` for the actor group; use `--communication-timeline-run-id` or `SLIME_COMMUNICATION_TIMELINE_RUN_ID` to supply one explicitly.

## Built-in phases

The trainer records:

- `train_forward_backward`: the Megatron forward/backward schedule;
- `grad_sync`: Megatron's final gradient synchronization callback;
- `optimizer_step`;
- `weight_convert`: work performed while producing each HF weight bucket;
- `weight_bucket_ready` and `weight_bucket_send`;
- `engine_bucket_receive`: the trainer's observation that transfer has completed;
- `engine_load_weights`: time until the engine update request returns;
- `weight_sync_complete`.

`engine_bucket_receive` is a trainer-side observation, not an engine-side timestamp. The `observation` metadata says which boundary produced it.

Every record carries the common fields `global_step`, `rollout_id`, `weight_version`, `bucket_id`, `trainer_rank`, `engine_id`, `message_bytes`, and `transport`. Fields that do not apply at a boundary are `null`. `sequence_id` is process-local and monotonic, while `logical_operation_id` combines the available lifecycle identifiers with the operation name.

CUDA spans use events on the current framework stream. Events are queried without blocking during normal execution and drained at process shutdown. These timestamps bracket framework eligibility and observed completion; they do not claim to be the exact start of an NCCL kernel on an internal ProcessGroupNCCL stream. Each span also emits an NVTX range named `slime.comm/<operation>` so Nsight Systems can provide exact kernel correlation.

## Add semantic phases from custom code

slime deliberately does not patch Megatron's internal MoE implementation. A custom Megatron hook or plugin can add `ep_dispatch` and `ep_combine` without changing the schema:

```python
from slime.observability.communication_timeline import communication_context, communication_phase

with communication_context(global_step=step, rollout_id=rollout_id):
    with communication_phase(
        "ep_dispatch",
        message_bytes=dispatch_bytes,
        transport="nccl",
        layer=layer_id,
    ):
        dispatch_tokens()
```

Use `communication_event(...)` for an instantaneous boundary and call `mark_consumer()` on the object yielded by `communication_phase(...)` when the first consumer is observed.

This timeline is process-oriented and complements the per-sample [Trace Viewer](./trace.md). For kernel-level analysis, correlate its NVTX ranges with [Profiling](./profiling.md).
