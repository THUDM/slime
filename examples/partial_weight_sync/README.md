# Partial Weight Sync (delta / selective)

This example demonstrates non-colocated weight sync with **partial-update modes**: instead of broadcasting every parameter on every sync, slime broadcasts only the changed-position payload, and the SGLang receiver applies it without rebroadcasting the unchanged majority of the weights. Two sub-modes:

- **`delta`** — broadcast `(current − snapshot)` sparse-encoded; receiver applies additively (`param += delta`). Inspired by [Cursor Composer 2](https://cursor.com/resources/Composer2.pdf) and [Fireworks AI — Frontier RL Is Cheaper Than You Think](https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think).
- **`selective`** — broadcast new values at changed positions only (with NaN as the "unchanged" sentinel); receiver overwrites those positions, leaves others alone. Lossless by construction (no arithmetic), wire ~½ the size of fp32 delta. Inspired by [arXiv:2509.19128](https://arxiv.org/abs/2509.19128).

For non-colocated runs the wire shrinks roughly in proportion to the change density, which is typically a few percent during RL fine-tuning at conservative learning rates. The broadcast that previously dominated the sync phase becomes a small fraction of it. Colocated runs share GPU memory via CUDA IPC and have no wire — partial-update modes buy nothing there and are rejected at argparse time.

## Files

- `run-glm4.7-355B-A32B-partial.sh`: 16-node (8 actor + 8 rollout) GLM-4.7-355B-A32B launcher with partial-update flags set.

## Usage

Set up the same checkpoint and dataset paths as a standard non-colocated GLM-4.7 run (see [docs/en/examples/glm4.7-355B-A32B.md](../../docs/en/examples/glm4.7-355B-A32B.md)), then launch:

```bash
bash examples/partial_weight_sync/run-glm4.7-355B-A32B-partial.sh
```

The script has two pre-built `PARTIAL_ARGS` blocks; the delta block is active by default and the selective block is commented out. Comment one out to switch.

**Delta mode:**

```bash
PARTIAL_ARGS=(
   --update-weight-mode delta
   --update-weight-partial-encoding sparse_indices
   --update-weight-delta-dtype fp32
   --update-weight-base-sync-interval 9999
)
```

**Selective mode:**

```bash
PARTIAL_ARGS=(
   --update-weight-mode selective
   --update-weight-partial-encoding sparse_indices
   --update-weight-base-sync-interval 9999
)
```

Notes:
- `--update-weight-delta-dtype` is delta-only (silently ignored in selective mode — no arithmetic happens there).
- `--update-weight-base-sync-interval` defaults to `9999` — effectively disables periodic base syncs because both modes are lossless under their defaults (delta with fp32 math, selective by construction). Set lower (e.g. `30`) if you want to verify correctness against periodic full broadcasts, or if your workload has a custom base-sync requirement.
- `--update-weight-partial-encoding` accepts `sparse_indices` / `sparse_bitmask` / `dense`.

And one receiver-side flag in `SGLANG_ARGS`:

```bash
--sglang-update-weight-partial-chunk-bytes $((2 * 1024 * 1024 * 1024))
```

See [docs/en/advanced/partial-weight-sync.md](../../docs/en/advanced/partial-weight-sync.md) for the wire protocol, encoding choice, and precision behaviour.

## Results

### Delta mode

W&B traces comparing `delta` mode against the full-sync baseline on the run above.

![Raw reward](./raw_reward.png)

*Raw reward over training steps — delta and full match.*

![Train/rollout logprob abs diff](./train_rollout_logprob_abs_diff.png)

*Absolute logprob difference between train and rollout — delta and full match.*

![Update weights time](./update_weights_time.png)

*Per-step weight-update wall-clock — delta is substantially faster.*

### Selective mode

W&B traces comparing `selective` mode against the full-sync baseline.

<!-- TODO: add raw_reward_selective.png / train_rollout_logprob_abs_diff_selective.png / update_weights_time_selective.png and any commentary on per-sync density + wall-clock vs delta. -->

*Placeholder — selective experiment numbers and traces pending.*

## Reading the curves

The reward / logprob-diff curves track each other closely between modes, but they don't sit pixel-on-pixel. That divergence is **not** evidence that partial-update modes lose information — both modes shipped here are mathematically lossless under their respective recipes:

- **`delta` with `--update-weight-delta-dtype fp32`**: every bf16 value is exactly representable in fp32; the subtraction is exact within fp32; the receiver's in-place `bf16 += fp32` add casts back identically to what a full sync would store. Receiver state matches a full-sync reference bit-for-bit per step.
- **`selective`**: no arithmetic at all — the receiver overwrites changed positions with the trainer's exact bf16 values. Lossless by construction.

The small curve-to-curve divergence comes from **non-determinism elsewhere in the training/rollout stack** (cuBLAS reductions, FlashAttention split-K, NCCL all-reduce ordering, dynamic-batch token assignment). Two identically-configured *full*-sync runs would diverge the same way. What's "matching" between partial and full here is the trajectory, not the bits.

## Why `sparse_indices` for this run

Per-sync weight-change density during RL fine-tuning at conservative learning rates is typically a few percent — see for instance [arXiv:2602.03839](https://arxiv.org/pdf/2602.03839), which reports that only on the order of 1% of weights change per RL update. Our own logs on the GLM-4.7-355B run measured roughly **2–3% density per sync**.

The break-even density between the two sparse encodings is independent of the value dtype. With `n = numel`, `k = nnz`, `v = value-dtype bytes`:

```
sparse_indices wire = k * (4 + v)         (int32 indices + values)
sparse_bitmask wire = ceil(n / 8) + k * v (1 bit per element + values)
```

Equal when `4k = n/8`, i.e. `k/n = 1/32 ≈ 3.125%`. Below 3.125% `sparse_indices` is smaller; above, `sparse_bitmask` wins. Our 2–3% observed density sits below the break-even — hence `sparse_indices` is the right pick for this workload. (`dense` is the no-compression option, kept around for debugging the additive / selective apply path independently of the sparse encoding.)

## Composes with any communication optimization in slime

This feature only changes *what bytes get shipped*; it does not touch the NCCL broadcast itself, the Ray lock around it, the bucket scheduling, or any send/receive layer. So any future slime improvement to the weight-update communication path — better compute/broadcast overlap, NIC-level optimizations, pipeline-parallel sends, deduplicated metadata — stacks additively on top of the speedups shown above. Both `delta` and `selective` inherit those gains for free.

## Two modes, one feature

`delta` and `selective` are both *lossless* partial-update modes. They differ only in what they put on the wire and how the receiver applies it:

- **`delta`** keeps an arithmetic path (`receiver += sender's delta`). Wire-values portion is 4 bytes/element at fp32. Pick this when you want the arithmetic semantics (e.g. for compatibility with future ideas that compose with additive apply).
- **`selective`** carries new values directly (~½ the values-wire at bf16) and applies them by selective overwrite. No arithmetic on either side, so the receiver is bit-exact with the trainer regardless of dtype. Pick this when wire size is the binding constraint.

Both are exposed so you can pick the trade-off that fits your run.
