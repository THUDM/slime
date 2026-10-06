# Persistent rollout with straw

[straw](https://github.com/zhuzilin/straw) provides filesystem-based queues and
packed tensor storage for slime. It persists prompt tasks, partial rollouts,
completed samples and training batches. Generation and training processes read
and write payloads in parallel across machines; Ray carries control messages
and references.

The supported shared-filesystem target is JuiceFS. All nodes must mount the
storage pool at the same absolute path. Other network filesystems, including
NFS, require separate verification of locking, visibility and durability.

## Enable it

The standard slime installation and Docker build include `straw-queue`. To
install it in an existing environment, run this on every rollout and training
node, using the same version throughout the job:

```bash
pip install 'straw-queue>=0.1.2'
```

Add these arguments to your training command:

```bash
--rollout-data-transport straw \
--rollout-data-dir /shared/juicefs/jobs/my-run/rollout_data \
--rollout-queue-run-id my-run \
--rollout-storage-profile juicefs \
--rollout-storage-declaration /shared/juicefs/deployment.json
```

The deployment declaration describes verified mount settings; it does not
configure the mount. It includes `direct_mount: true`, `writeback: false`,
`open_cache: 0`, `readdir_cache: false`, `client_version` and
`durability_description`. See straw's
[filesystem contract](https://github.com/zhuzilin/straw/blob/main/docs/FILESYSTEM.md).
Use the default `local` profile for local POSIX development and testing.

If `--rollout-data-dir` is omitted, a fresh run uses `<save>/rollout_data`; without
`--save`, an explicit shared directory is required. Checkpoint recovery can infer
the pool from the saved checkpoint.

Storage and execution mode are independent. The defaults are Ray `object-store`
transport and synchronous rollout. For distributed fully async execution, also add:

```bash
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
```

Fully async runs one generation process per Ray node with CPU resources and
shares the configured concurrency across them. Faster workers can supply a
training batch without waiting for every worker. Producers pause admission
during weight synchronization and persist in-flight results before resuming.

The default object-store transport works without straw installed. Selecting
straw without the package fails at startup with the installation command.

## Data flow and scheduling

1. Workers request prompt groups. The data source reads the dataset and saves
   both the groups and its updated cursor to the queue.
2. Workers generate and score samples. Returned partial groups are persisted
   for continuation; completed, accepted groups become available for training.
3. The batch builder applies reward/conversion hooks and divides selected
   samples among training ranks. After training finishes, slime acknowledges
   consumption so storage can be reclaimed when no other references need it.

Rollout and training data share the same storage pool. Large R3 and SC tensors
can be reused across these stages without duplicating their bytes. Queue tasks
use renewable leases to assign work to workers; unfinished work from a lost
worker becomes available again after lease expiry.

Scheduling prioritizes **completed groups → partial groups → fresh prompts**.
Within each stage, groups with older generated-token weight versions come first,
with FIFO ordering for ties. Groups without a numeric version come last in that
stage. Staleness is the current serving version minus the oldest generated-token
version, so fresh samples have staleness 0. Scheduling does not automatically
discard stale samples.

`--buffer-filter-path` is unsupported with straw. `--buffer-sort-by-staleness`
applies to the in-memory data source; straw uses the ordering above. Reward and
sample-selection hooks are available; distributed fully async does not support
`--rollout-all-samples-process-path`. See [customization](../get_started/customization.md)
for custom rollout functions and queue readers.

## Packed storage and supported data

Packing is enabled whenever Straw transport is selected. Each writer appends
multiple samples and tensors to the same file (a pack), reducing small-file
metadata overhead on shared storage. slime uses Straw's default target size,
currently **1 GiB**. Use `--rollout-queue-segment-mib` to override it in MiB.

The target size controls file rotation; readers can access each publication
without waiting for the pack to fill. A publication larger than the target
stays intact, and explicit sealing may leave smaller packs.

Immutable tensor references let rollouts, training batches, and checkpoints
share data. Updates write new records while retained references continue to
identify their original contents.

The sample codec supports nested lists, tuples and dictionaries; scalar and
byte values; NumPy arrays; PyTorch tensors; PIL images; and Sample fields,
including supported custom fields. R3 routes and SC top-k/ragged top-p data
are stored as typed tensors. Unsupported Python objects and cyclic structures
raise an error when published. Stored records and tensor reads are checksummed.

Use `--use-rollout-routing-replay` for R3 and `--use-score-centering` for SC.
With straw transport, their tensors are persisted with the sample group. R3
training reads only the rows assigned to the current CP/TP rank. Large replay
tensors stay lazy during batch conversion, although selected Sample metadata
and ordinary fields still occupy manager memory.

| Option | Default | Purpose |
|---|---|---|
| `--rollout-queue-segment-mib` | Unset; uses Straw's default (currently `1024`) | Override the target pack rotation size in MiB; explicit values must be positive. |
| `--rollout-io-concurrency` | `4` | Bound concurrent serialization and filesystem I/O submissions |
| `--rollout-queue-lease-seconds` | `300` | Worker lease duration; active readers renew it |

## Online GC

Online GC is disabled by default. Enable it with `--rollout-queue-online-gc`.
slime reports when tasks and training batches are finished or discarded. straw
reclaims a sealed pack only after all tasks, readers, checkpoints and archives
have released their references. One live record keeps the whole pack alive;
active writers and journal history also consume space.

Retained checkpoints and archives protect their payloads even with GC enabled,
so rollback does not require disabling GC. Retention must also be released
when data is no longer needed; deleting an index file alone does not release
its storage ownership. Do not manually delete pack files from an active pool.
For offline removal, stop all coordinators, writers and readers first.

A GC failure is reported by subsequent queue operations and at shutdown.
Disk exhaustion requires freeing unneeded retention or expanding storage;
it does not reset the queue automatically. Live-pack and journal compaction are
not implemented.

## Recovery and checkpoints

Save model and rollout state together through normal training checkpoints.
They contain the dataset cursor, sample/group counters, pending and partial
inputs, ready groups, and training progress. Payloads already in straw are
referenced rather than copied into each checkpoint. Retaining a checkpoint
keeps its referenced data available.

This section covers restarting with a new serving cluster. Stop the entire
previous job first, including remote workers. The save-directory lock rejects
concurrent coordinators, but does not stop orphaned readers. There is no automatic
coordinator failover.

If only Megatron training failed and Ray and serving are still running, follow
the [fault-tolerance guide](fault-tolerance.md) to resubmit training while retaining
the serving cluster and queue controller.

### Resume or select a step

To resume the latest completed checkpoint, use the same logical directories:

```bash
--rollout-data-transport straw \
--load /shared/checkpoints/run \
--save /shared/checkpoints/run \
--save-interval 1
```

Whenever `--save` is set, Megatron requires a positive `--save-interval`.
The example saves after each rollout; choose the interval for your workload.

To restore the state saved after rollout 7, add `--ckpt-step 7`. The next rollout
is 8. Keep the dataset, model/tokenizer configuration, straw run ID, storage
profile and fully async worker topology consistent with the checkpoint.
Optimizer and training RNG state must be present for training recovery.

Restoration creates an isolated queue sharing the checkpoint's immutable
payloads. Pending/partial inputs and ready-group order come from that checkpoint,
with the dataset cursor restored to the saved position. Later samples from the
source run are excluded. Further writes leave the source checkpoint intact.

When `--save` already contains a run, outputs go to a unique `branches/<id>`
directory. `rollout/current.json` tracks the active branch, so subsequent restarts
can keep using the same logical `--load` and `--save`. If `--load` is omitted and
`--save` has an active branch, it is resumed automatically. Startup logs the
resolved paths. Give each run a fresh path for immutable `.straw.json` debug archives.

| Selection | Arguments |
|---|---|
| Step in the current branch's history | `--load /shared/checkpoints/run --ckpt-step 7` |
| A particular branch | `--load /shared/checkpoints/run/branches/<id> --ckpt-step 7` |
| An exact checkpoint | `--load /shared/checkpoints/run/rollout/committed_7.json` |
| Separate output directory | `--save /shared/checkpoints/another-run` |

You can also edit `latest_checkpointed_iteration.txt` in the logical save
directory or current branch to select an earlier step. Explicit `--ckpt-step`
or an exact commit file takes precedence. Automatic selection uses completed
model-and-queue checkpoints and does not follow a newer incomplete model save.
It follows branch ancestry only up to each branch point; select an exact commit
file to load an abandoned future from another branch.

### Missing state and recovery limits

If the model exists but no queue snapshot was saved, recovery starts an empty
queue. When `rollout/global_dataset_state_dict_<step>.pt` exists, its dataset
cursor is restored; otherwise the dataset starts at offset 0 with a warning.
No pending, partial or ready data is taken from another step. Missing models,
incomplete or corrupt snapshots, missing payloads and unsupported snapshot
versions raise errors instead of silently starting empty.

Before any model checkpoint exists, restarting the original run with the same
`--save` and model/input configuration can recover persisted rollout work, but
only before the first training batch has been planned. After batch planning,
recovery requires a matching model/optimizer and rollout checkpoint.

Recovery uses the original straw pool; copying a checkpoint directory alone
does not copy its payloads. The recovery boundary is a completed rollout
training batch, not an optimizer microstep. GPU KV caches and generation RNG
state are not restored, so newly generated tokens and random hooks may differ.

When extending `--num-rollout`, use Megatron's
`--use-checkpoint-opt-param-scheduler` if you want to retain the saved optimizer
schedule. Queue restoration does not override optimizer settings.

## Debug archives and sample lookup

Both `.pt` and `.straw.json` work with the debug save/load flags:

```bash
--save-debug-rollout-data '/shared/debug/rollout_{rollout_id}.straw.json'
# In a separate train-only job, without SGLang:
--load-debug-rollout-data '/shared/debug/rollout_{rollout_id}.straw.json'
```

| Format | Contents | Storage requirement |
|---|---|---|
| `.pt` | Sample data and materialized tensors | Self-contained file |
| `.straw.json` | Immutable index with sample/task keys | Keep the referenced straw pool available |

An indexed archive retains its data independently of queue consumption and GC.
It shares existing tensors when saved with straw transport; otherwise it creates
a `straw-data` pool alongside the index. There is one index per rollout, with
samples packed together. Evaluation uses `eval_<id>` in place of the rollout ID.
Copying only the JSON index does not copy the payloads. For train-only replay
with straw transport automatically reuses the archive’s storage pool and run,
overriding `--rollout-data-dir` and `--rollout-queue-run-id`. Each replay creates
an isolated queue in that writable pool; the source queue is left untouched.
Full replay (without subsampling) reuses the archived Sample and tensor records
without republishing them.
All rollout archives in one replay must belong to the same pool and run.
`--load-debug-rollout-data-subsample` also applies to archives.

```python
from slime.data.archive import RolloutArchive
from slime.observability.rollout_data_utils import load_debug_rollout_data

with RolloutArchive('/shared/debug/rollout_7.straw.json') as archive:
    print(archive.keys())  # (sample key, optional task key), in archive order
    samples = archive.load_samples(sample_key='sample:42')
    group = archive.load_samples(task_key='prompt:21')
    archive.export_pt('/shared/debug/rollout_7.pt')

# Convert a .pt dump to an indexed archive with an adjacent straw-data pool.
samples = load_debug_rollout_data('/shared/debug/rollout_7.pt', rollout_id=7)
RolloutArchive.save('/shared/debug/imported_7.straw.json', samples, rollout_id=7)
```

Keys select data as saved in that archive. Missing keys raise `KeyError`.
Lookups return lists because compact trajectories can share a sample index.
Samples without an index use `position:<ordinal>`; samples without queue
provenance have no task key.

Closing an archive closes its reader while retaining its data. After all readers
finish, open the archive and call `archive.release()` to release its retention.
Other checkpoints, queues and archives retain their own references. An exported
`.pt` remains readable after straw reclaims the payloads.
