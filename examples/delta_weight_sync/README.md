# Delta Weight Sync

Non-colocated weight sync that ships only changed positions + values instead of every parameter. Two transports over one wire format and one receiver-side decoder:

- **Disk** (the point) — write per-flush safetensors to a shared filesystem; one HTTP push per sync. Designed for **training/inference disaggregation** across datacenters where bandwidth between trainer and rollout is on the order of 100s of MB/s.
- **NCCL** (the baseline) — broadcast each per-flush bucket directly. Used intra-datacenter to validate that the wire encoding and apply logic are correct, separate from any shared-FS variable.

Both modes are lossless by construction (selective overwrite via NaN sentinel; no arithmetic).

## Files

- `run-glm4.7-355B-A32B-delta.sh`: 16-node (8 actor + 8 rollout) GLM-4.7-355B-A32B launcher. Disk transport active by default; NCCL block commented below it.

## Usage

```bash
bash examples/delta_weight_sync/run-glm4.7-355B-A32B-delta.sh
```

**Disk (default):**

```bash
DELTA_ARGS=(
   --update-weight-mode delta
   --update-weight-transport disk
   --update-weight-encoding deltas_zstd
   --update-weight-disk-dir /shared/fs/delta-updates
)
```

**NCCL (baseline):**

```bash
DELTA_ARGS=(
   --update-weight-mode delta
   --update-weight-transport nccl
   --update-weight-encoding indices
)
```

Receiver-side byte cap (both transports):

```bash
--sglang-update-weight-delta-chunk-bytes $((2 * 1024 * 1024 * 1024))
```

See [docs/en/advanced/delta-weight-sync.md](../../docs/en/advanced/delta-weight-sync.md) for the wire protocol, encoding choice, and design.

## Results

W&B traces comparing delta sync against the full-sync baseline on GLM-4.7-355B-A32B / DAPO-Math-17k.

![Raw reward](./raw_reward.png)

![Train/rollout logprob abs diff](./train_rollout_logprob_abs_diff.png)

![Update weights time](./update_weights_time.png)

> **Note on the small curve-to-curve gap.** RL training is inherently non-deterministic (cuBLAS reductions, FlashAttention split-K, NCCL all-reduce ordering, dynamic-batch token assignment). Two identically-configured *full*-sync runs would diverge the same way. Delta sync's selective overwrite is bit-exact with full sync per step (no arithmetic, no drift); the trajectory matches, the bits don't.

![Update weights density](./update_weights_density.png)

*Per-sync change density (`perf/update_weights_density`) — fraction of weight positions that moved between consecutive syncs. Sync 0 is omitted: it's the snapshot-seeding pass with density = 1.0, which would compress the y-axis.*

## Why these encoding defaults

Per-sync change density during RL fine-tuning at conservative LRs sits around **2-3%** ([arXiv:2602.03839](https://arxiv.org/pdf/2602.03839) reports ~1% on a related setup; we measured ~2-3% on this run). Below the 3.125% break-even point, gap-encoded positions are smaller than absolute indices — the disk default `deltas_zstd` adds zstd L1 on top to squeeze the gap byte stream further (~35-40%), which is the right tradeoff when shared-FS bandwidth is ≤ 300 MB/s. Intra-datacenter NCCL has no bandwidth pressure, so `indices` (lowest compute, biggest payload) is the cleaner default there.
