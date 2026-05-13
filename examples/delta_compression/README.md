# Delta-Compression Weight Sync

This example demonstrates non-colocated weight sync with **delta compression**: instead of broadcasting every parameter on every sync, slime broadcasts the sparse-encoded difference between the current weights and the last sync's weights, and the SGLang receiver applies it additively (`param += delta`).

For non-colocated runs the wire shrinks by ~30× (typical 2–3% density at 355B), and the broadcast that previously dominated the sync phase becomes a small fraction of it. Colocated runs share GPU memory via CUDA IPC and have no wire — delta compression buys nothing there and is rejected at argparse time.

## Files

- `run-glm4.7-355B-A32B-delta.sh`: non-colocated GLM-4.7-355B-A32B launcher with delta-compression flags set.

## Usage

1. Set up the same checkpoint and dataset paths as a standard non-colocated run (see [docs/en/examples/glm4.7-355B-A32B.md](../../docs/en/examples/glm4.7-355B-A32B.md)).

2. Launch:

```bash
ACTOR_NUM_NODES=4 ROLLOUT_NUM_GPUS=64 MASTER_ADDR=<head-ip> \
    bash examples/delta_compression/run-glm4.7-355B-A32B-delta.sh
```

The flags that switch the run into delta mode are grouped in `DELTA_ARGS` near the bottom of the script:

```bash
--update-weight-mode delta            # default 'full'; first sync is always full
--delta-compression sparse_indices    # 'sparse_indices' / 'sparse_bitmask' / 'dense'
--delta-dtype fp32                    # subtraction and apply happen at this dtype
--delta-full-interval 10000           # full sync every N successful deltas
```

And one receiver-side flag in `SGLANG_ARGS`:

```bash
--sglang-update-weight-delta-chunk-bytes $((2 * 1024 * 1024 * 1024))
```

See [docs/en/advanced/delta-compression.md](../../docs/en/advanced/delta-compression.md) for the wire protocol, encoding choice, and precision behaviour.

## When to use which encoding

`sparse_indices` and `sparse_bitmask` are the two compressed wire formats. With `n = numel`, `k = nnz`, `v = value dtype bytes`:

- `sparse_indices` wire = `k * (4 + v)` (int32 indices + values)
- `sparse_bitmask` wire = `ceil(n/8) + k * v` (1 bit per element + values)

Break-even density is `k/n = 1/32 ≈ 3.125%` regardless of `v`. Below 3% pick `sparse_indices`; above 3% pick `sparse_bitmask`. `dense` is for debugging.
