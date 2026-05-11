# Mooncake DataProto Rollout Transfer

slime can transfer rollout data through Mooncake instead of Ray object references. This is useful when the rollout producer and actor consumer run on different nodes and Mooncake Store is configured for the cluster transport.

The default transfer backend remains Ray. Enable Mooncake DataProto transfer explicitly:

```bash
python3 train.py \
  --transfer-backend mooncake_dataproto \
  --mooncake-dataproto-store-init-kwargs '{"setup_method":"setup"}'
```

## What is transferred

The Mooncake path keeps slime's rollout data layout unchanged:

- per-rank rollout partitions are still selected by slime before actor consumption;
- tensor fields such as `tokens` and `loss_masks` are stored as Mooncake remote tensor batches;
- non-tensor rollout fields and metadata stay in the `DataProto` wrapper;
- cleanup keys are tracked in metadata and removed after actor-side materialization.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--transfer-backend` | `ray` | Set to `mooncake_dataproto` to enable Mooncake rollout transfer. |
| `--mooncake-dataproto-store-init-kwargs` | `null` | JSON arguments used to initialize the Mooncake store. Use `{"setup_method":"setup"}` for real Mooncake Store setup and `{"setup_method":"setup_dummy"}` for local unit tests. |
| `--mooncake-dataproto-hard-pin` | `true` | Hard-pin remote tensor data to the producer segment when publishing tensor batches. |

For performance runs, configure Mooncake Store with the production transport, for example RDMA, and keep buffer registration or prewarm costs separate from online transfer latency.
