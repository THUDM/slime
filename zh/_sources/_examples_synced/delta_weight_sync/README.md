# Delta Weight Sync

Non-colocated weight sync that ships only the **changed bytes** between two syncs instead of a
full checkpoint, for training/inference disaggregation across clusters or datacenters. The
trainer publishes per-tensor deltas to a shared filesystem as a canonical HF checkpoint
directory; each engine's `/pull_weights` applies them into a host-local checkpoint on every
host it spans, and the engines reload through the ordinary `update_weights_from_disk` path —
slime only ever talks to one endpoint per engine.

See [Delta Weight Sync](../../docs/en/advanced/delta-weight-sync.md) for the full mechanism,
encodings, integrity checks, and shared-filesystem visibility hooks.

## Try it

`run-glm4.7-30B-A3B-delta.sh` runs the disk delta path on GLM-4.7-Flash, non-colocated across a
2-node (16-GPU) Ray cluster. See its header for prerequisites.

## Minimal flags

Add to a non-colocated training run (the trainer and engines only need to share the filesystem
at `--update-weight-disk-dir`):

```bash
--update-weight-mode delta \
--update-weight-transport disk \
--update-weight-disk-dir   /shared/fs/delta-updates \
--update-weight-local-checkpoint-dir /local/nvme/rollout-ckpt \
--update-weight-delta-encoding xor \
--update-weight-delta-checksum xxh3-128
```

- `--update-weight-disk-dir` — shared directory the trainer writes deltas to and the hosts read.
- `--update-weight-local-checkpoint-dir` — host-local full HF checkpoint the delta patches in
  place; materialized from the engine's model path on the first `/pull_weights`.
- `--update-weight-delta-encoding` — `xor` (smallest/fastest) or `overwrite` (idempotent).
- `--update-weight-delta-checksum` — `xxh3-128` (default), `blake3`, or `adler32`.

For object-store-backed volumes that need an explicit commit/refresh to make writes visible
across hosts, supply `--custom-update-weight-post-write-path` (trainer side) /
`--sglang-custom-pull-weights-pre-read-hook` (engine side) — no vendor-specific code lives in slime
or sglang; see the doc.
