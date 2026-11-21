# Delta Weight Sync

Delta weight sync keeps non-colocated rollout engines up to date by shipping only the bytes
that changed between two syncs, instead of a full checkpoint each time. It targets large-model
training/inference disaggregation across clusters or datacenters, where writing the whole actor
every sync is the dominant cost.

It is **disk-transport only**. The trainer publishes each sync as a canonical HF checkpoint
directory; the engine's `/pull_weights` endpoint (shipped in slime's sglang patch) fans the
apply out to **every host the engine spans** and verifies it, then the engine reloads the
patched local checkpoint through the **ordinary** `update_weights_from_disk` endpoint. slime
only ever talks to one endpoint per engine, so multi-node serving and external rollout engines
need nothing extra on the slime side.

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
| `--update-weight-local-checkpoint-dir` | Host-local (e.g. NVMe) full HF checkpoint that `/pull_weights` keeps in sync. The initial full publication establishes its baseline; later deltas apply in place, and a new full publication replaces it. |
| `--update-weight-delta-encoding` | On-disk delta encoding: `xor` (default) or `overwrite`. |
| `--update-weight-delta-checksum` | Per-tensor integrity checksum: `xxh3-128` (default), `blake3`, or `adler32`. |

Deltas are always zstd-compressed (level 1); profiling showed it dominates lz4 / gzip / snappy / brotli on both wire size and decompress speed for this data, so it is not a knob.

## How it works

1. **Seed.** The first sync, including after trainer recovery or an engine replacement,
   publishes a full HF checkpoint from the trainer's current weights. It captures the CPU
   snapshot from those exact tensors and reloads every engine before generation resumes.
   The version exceeds every existing version directory, including an update whose reply
   was lost. Existing versions are retained; restarting does not delete the stream.
2. **Publish.** On every later sync the trainer diffs each gathered HF tensor against the
   snapshot, encodes and compresses the change, and writes a new version directory
   `weight_v{N:06d}/` under `--update-weight-disk-dir`. The directory is a canonical HF
   checkpoint — `model-NNNNN.safetensors` files holding the compressed diff tensors plus a
   `model.safetensors.index.json` (tensor name → file) carrying the apply metadata — so the
   artifact is portable, not tied to the trainer's parallelism layout. The snapshot is then
   advanced to the new values for the next diff.
3. **Pull.** The trainer calls `/pull_weights` on each engine. Inside the engine the request is
   broadcast to every rank on every node; each host applies the new version's delta into its
   local checkpoint in place (a per-host file lock collapses co-located ranks to one apply).
   The apply is parallelized across tensors and verified per-tensor (see Integrity); the call
   only reports success once **every host** holds a checksum-verified checkpoint.

   `/pull_weights` is not delta-specific: each published version is self-describing, and a
   version that is an ordinary full HF checkpoint (no delta metadata in its index) is pulled by
   copying it as-is — resetting the chain, so a fresh host joining late seeds from the newest
   full version instead of replaying every delta, and older deltas can be pruned. slime's
   full-mode disk sync uses exactly this when `--update-weight-local-checkpoint-dir` is set.
4. **Reload.** The engines reload the patched local checkpoint through the vanilla
   `update_weights_from_disk` path — the weight-loading code never sees the delta format.

The snapshot and the engines start from the same published HF bytes, including when the
trainer restored a newer model than `--hf-checkpoint`. Megatron→HF conversion differences
(such as trimmed embedding / LM-head padding) therefore do not corrupt the next delta.

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
applying, every host recomputes the checksum and **raises on any mismatch** — the failure
propagates through the `/pull_weights` response, so a corrupt delta or a wrong base fails loud
instead of serving bad weights. The apply also refuses to run out of order: a version only
applies on top of its declared base version.

`--update-weight-delta-checksum` selects the algorithm. The checksum is not the apply bottleneck
(the apply is decompress + XOR bound), so this is a digest-property choice, not a speed one:
`xxh3-128` (default) is the widest fast non-cryptographic digest; `blake3` is cryptographic, for
untrusted storage; `adler32` is for interop with systems that expect it.

## Shared-filesystem visibility hooks

On a POSIX shared filesystem (NFS, Lustre, …) no extra step is needed. Object-store-backed
mounts that need an explicit publish/refresh to make writes visible across hosts can supply two
optional hooks, loaded by import path — no vendor-specific code lives in slime or sglang:

- `--custom-update-weight-post-write-path` (slime, trainer side): called after a version's files are
  written, before the engines are told to read it (e.g. upload pending writes to the backing object store).
  Signature: `hook(args, version_dir, rollout_engines)`.
- `--sglang-custom-pull-weights-pre-read-hook` (sglang server arg, engine side): called on each host
  inside the engine before `/pull_weights` reads the delta directory (e.g. refresh the mount's view).
  Signature: `hook(delta_dir, target_version)`.
