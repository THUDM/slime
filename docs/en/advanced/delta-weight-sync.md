# Delta Weight Sync

Delta weight sync keeps non-colocated rollout engines up to date by shipping only the bytes
that changed between two syncs, instead of a full checkpoint each time. It targets large-model
training/inference disaggregation across clusters or datacenters, where writing the whole actor
every sync is the dominant cost.

It is **disk-transport only** and reloads through the **ordinary** `update_weights_from_disk`
endpoint, so the inference engine needs no delta-specific support.

## Configuration

```bash
--update-weight-mode delta
--update-weight-transport disk
--update-weight-disk-dir /shared/fs/delta-updates
--update-weight-local-checkpoint-dir /local/nvme/rollout-ckpt
--update-weight-delta-encoding xor          # or: overwrite
--update-weight-delta-checksum xxh3-128     # or: blake3, adler32
```

| Flag | Role |
|---|---|
| `--update-weight-disk-dir` | Shared filesystem directory the trainer publishes deltas to and the rollout hosts read from. |
| `--update-weight-local-checkpoint-dir` | Host-local (e.g. NVMe) full HF checkpoint that the delta is applied into in place. Each host materializes it from `--hf-checkpoint` at engine start. |
| `--update-weight-delta-encoding` | On-disk delta encoding: `xor` (default) or `overwrite`. |
| `--update-weight-delta-checksum` | Per-tensor integrity checksum: `xxh3-128` (default), `blake3`, or `adler32`. |

Deltas are always zstd-compressed (level 1); profiling showed it dominates lz4 / gzip / snappy / brotli on both wire size and decompress speed for this data, so it is not a knob.

## How it works

1. **Seed.** On the first sync the trainer captures a CPU snapshot of every parameter — seeded
   from `--hf-checkpoint`, which is exactly what each rollout host materializes its local
   checkpoint from. Nothing is published; this snapshot is the base the next sync diffs against.
2. **Publish.** On every later sync the trainer diffs each gathered HF tensor against the
   snapshot, encodes and compresses the change, and writes a new version directory
   `weight_v{N:06d}/` under `--update-weight-disk-dir`. The directory is a canonical HF
   checkpoint — `model-NNNNN.safetensors` files holding the compressed diff tensors plus a
   `model.safetensors.index.json` (tensor name → file) carrying the apply metadata — so the
   artifact is portable, not tied to the trainer's parallelism layout. The snapshot is then
   advanced to the new values for the next diff.
3. **Apply.** Each rollout host applies the new version's delta into its local checkpoint in
   place. The apply is parallelized across tensors and verified per-tensor (see Integrity).
4. **Reload.** The engines reload the patched local checkpoint through the vanilla
   `update_weights_from_disk` path — they never see the delta format.

Because the snapshot is seeded from `--hf-checkpoint` (the engine's actual base) rather than
from the current GPU weights, the scheme is correct for any model even where the Megatron→HF
round-trip is not byte-exact (e.g. trimmed vocab-padding rows in the embedding / LM head).

## Encodings

Both encodings are byte-level and dtype-blind, so the same path works for quantized checkpoints.
The engine reads the choice from each version's index metadata.

- **`xor`** (default): writes `new ^ old`. Smallest wire and fastest to apply (sequential,
  cache-friendly; the unchanged bytes are zeros the compressor crushes). It is an involution,
  so it must be applied **exactly once** against the correct base — applying it twice reverts.
- **`overwrite`**: writes the changed positions and their new absolute values. Larger on the
  wire and a less cache-friendly scattered apply, but **idempotent**: re-applying it (or
  finishing a partially-applied delta) converges to the same state regardless of how many times
  it runs. Use it when re-applicability matters more than wire size.

## Integrity

The trainer stores a per-tensor checksum of each tensor's new state in the version. After
applying, every host recomputes the checksum and **raises on any mismatch**, so a corrupt delta
or a wrong base fails loud instead of serving bad weights. The apply also refuses to run out of
order: a version only applies on top of its declared base version.

`--update-weight-delta-checksum` selects the algorithm. The checksum is not the apply bottleneck
(the apply is decompress + XOR bound), so this is a digest-property choice, not a speed one:
`xxh3-128` (default) is the widest fast non-cryptographic digest; `blake3` is cryptographic, for
untrusted storage; `adler32` is for interop with systems that expect it.

## Shared-filesystem visibility hooks

On a POSIX shared filesystem (NFS, Lustre, …) no extra step is needed. Object-store-backed
volumes that need an explicit commit/refresh to make writes visible across hosts can supply two
optional hooks, loaded by import path — no vendor-specific code lives in slime:

- `--custom-delta-pre-push-path`: called after a version's files are written, before the engines
  are told to read it (e.g. commit the volume). Signature: `hook(args, version_dir, rollout_engines)`.
- `--custom-delta-pre-read-path`: called on each rollout host before it reads the delta directory
  (e.g. refresh the volume). Signature: `hook(delta_dir, target_version)`.
