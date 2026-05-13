# Delta-Compression Weight Sync

This example demonstrates non-colocated weight sync with **delta compression**: instead of broadcasting every parameter on every sync, slime broadcasts the sparse-encoded difference between the current weights and the last sync's weights, and the SGLang receiver applies it additively (`param += delta`).

For non-colocated runs the wire shrinks roughly in proportion to the delta's density, which is typically a few percent during RL fine-tuning at conservative learning rates. The broadcast that previously dominated the sync phase becomes a small fraction of it. Colocated runs share GPU memory via CUDA IPC and have no wire — delta compression buys nothing there and is rejected at argparse time.

## Files

- `run-glm4.7-355B-A32B-delta.sh`: 16-node (8 actor + 8 rollout) GLM-4.7-355B-A32B launcher with delta-compression flags set.

## Usage

Set up the same checkpoint and dataset paths as a standard non-colocated GLM-4.7 run (see [docs/en/examples/glm4.7-355B-A32B.md](../../docs/en/examples/glm4.7-355B-A32B.md)), then launch:

```bash
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

## Results

W&B traces comparing delta-compression against the full-sync baseline on the run above.

![Raw reward](./raw_reward.png)

*Raw reward over training steps — delta and full match.*

![Train/rollout logprob abs diff](./train_rollout_logprob_abs_diff.png)

*Absolute logprob difference between train and rollout — delta and full match.*

![Update weights time](./update_weights_time.png)

*Per-step weight-update wall-clock — delta is substantially faster.*

## When to use which encoding

`sparse_indices` and `sparse_bitmask` are the two compressed wire formats. With `n = numel`, `k = nnz`, `v = value dtype bytes`:

- `sparse_indices` wire = `k * (4 + v)` (int32 indices + values)
- `sparse_bitmask` wire = `ceil(n/8) + k * v` (1 bit per element + values)

Break-even density is `k/n = 1/32 ≈ 3.125%` regardless of `v`. Below 3% pick `sparse_indices`; above 3% pick `sparse_bitmask`. `dense` is for debugging.
