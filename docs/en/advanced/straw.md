# Persistent rollout with straw

[straw](https://github.com/zhuzilin/straw) provides filesystem-based durable
queues and packed tensor storage for slime. With `--rollout-data-transport straw`,
prompt tasks, partial rollouts, accepted results and training batches have
persistent identities. Multiple generation and training processes access their
payloads directly through a shared filesystem; Ray schedules actors and carries
control messages and small references.

The current shared-filesystem target is JuiceFS. Each node must mount the same
pool at the same absolute path. Other network filesystems, including NFS, need
separate qualification of cross-client locks, visibility and durability.

## What changes in the architecture

| Concern | Ray object-store transport (default) | straw transport |
|---|---|---|
| Data source | `RolloutDataSourceWithBuffer` | `QueueDataSource` backed by durable prompt tasks |
| Payload exchange | Ray object references | `RecordSetRef` / tensor references into shared pack files |
| Worker assignment | In-memory producer state | Leases, attempts, durable continuations and accepted receipts |
| Rollout-to-training conversion | `BatchBuilder` applies hooks and DP partitioning | The same conversion, with a persisted selection plan and ready batch |
| Data lifetime | Process/object-store ownership | Explicit task, consumer, reader and checkpoint ownership |
| Cleanup | Object-store lifetime | Optional online GC of sealed packs with no remaining owners |

There are two logical stages: rollout work/results, and ready training batches.
They share the same straw storage pool, so a tensor retained by either stage or
a checkpoint remains live. These stages do not require two copies of each tensor.
The slime controller coordinates both stages; straw implements the storage,
journal, task acceptance and ownership protocol in Rust.

```{mermaid}
flowchart LR
    D[Dataset producer] -->|cursor + task submission| Q[Durable prompt tasks]
    Q -->|lease + input reference| W[Distributed generation workers]
    W -->|durable partial continuation| Q
    W -->|accepted group reference| R[Rollout results]
    R --> B[BatchBuilder: selection plan and DP shards]
    B -->|ready batch reference| T[Training ranks]
    T -->|training completed| C[Queue controller: progress and ownership]
    C --> G[straw online GC]
    P[(Shared JuiceFS pack files)] -. payload reads and writes .-> W
    P -. shared tensor references .-> B
    P -. rank-local reads .-> T
```

One non-restarting Ray actor owns the coordinator for the run. It serializes
queue transitions and dataset allocation; payload writes happen in the producing
processes. This permits parallel I/O across machines while retaining one owner
for each queue journal. There is no automatic coordinator election or failover.

## End-to-end data flow

1. **Submit prompts.** The dataset cursor and prompt tasks commit together.
   Workers claim leases and load prompt groups from the shared pool. A lost
   worker does not permanently abandon a preallocated range of dataset indices.
2. **Generate and continue.** Distributed fully async generation uses one worker
   per eligible Ray node. Workers generate, score and filter complete groups;
   partial groups are saved as durable continuations. R3 routes and SC rows must
   cover the saved token prefix. An incomplete server capture retries from the
   previous consistent input.
3. **Accept results.** Payloads become durable before their references are
   accepted. Receipts identify accepted groups, allowing recovery after an RPC
   reply is lost. Reassigned tasks receive new attempts; stale attempts cannot
   accept a replacement result.
4. **Build the training batch.** `BatchBuilder` persists the selected inputs and
   conversion configuration before calling reward/conversion hooks. It publishes
   all DP shards before marking one batch ready. Training readers check batch,
   plan and rank identity. The manager still materializes the selected samples
   for conversion, so its host memory remains a capacity consideration.
5. **Acknowledge completion.** After all rank training calls return, the manager
   reports `training_completed`. This advances runtime consumption and frees
   ready-batch capacity. It is distinct from a durable model/optimizer checkpoint.

Ray remains responsible for scheduling, RPC and failures of execution processes.
slime owns Sample schemas, rewards, filters, batching and training. straw owns
bytes, references, persistence and storage lifetime. Custom rollout functions
can still return Sample lists; the manager publishes a compatibility collection.
See [customization](../get_started/customization.md).

## Queue scheduling

`QueueReader` is a worker's queue client: it acquires and returns groups and renews
active leases. `QueueDataSource` adds job-level reader creation, consumer lifecycle
and source checkpoints. It inherits the same sample-access methods; it does not
implement another buffer. With `--rollout-data-transport straw`, the source is
selected automatically. Explicit `--data-source-path` values should use
`slime.rollout.queue_data_source.QueueDataSource`.

`QueueReader` has no local continuation buffer. `add_samples()` publishes valid
continuations, then `yield_tasks()` atomically replaces their inputs and returns
them to pending in one WAL transaction per bounded batch. The old leases end;
any worker can acquire the saved inputs with a new lease. Readers keep only
active leases and executing samples locally. Closing a reader does not rewrite
returned groups.

The persistent order is **completed delivery → partial → fresh prompt**. Within
a stage, the oldest numeric value in a group's `weight_versions` comes first
(higher staleness), then FIFO submission/return order. Groups without numeric versions sort last
within the stage. Comparing stored versions avoids rewriting queue priorities
on every model update. This ordering does not drop stale samples automatically.
Staleness is `current serving weight version - oldest generated-token weight version`: fresh samples have staleness 0.
There is one indexed queue, not three independently managed queues.

`--buffer-filter-path` is rejected for straw. `--buffer-sort-by-staleness` remains
an option for the in-memory data source; straw always applies the order above.
Reward and sample-selection hooks remain available. Already accepted groups
explicitly returned to the source get separate delivery tasks; accepted history
is immutable, and consuming/discarding a delivery also acknowledges its earlier
accepted versions. The fully async scheduler still keeps a bounded ready-results
window, with its own checkpoint state and durable accepted-result references.

Source snapshots now store the shared pending tasks, their exact immutable input
references and scheduling metadata once; worker snapshots store reader metadata.
The reader snapshot format is version 3; old reader-local v1/v2 buffer snapshots
require migration and are rejected explicitly. This requires an updated
`straw-queue` build with `yield_tasks()` and persistent scheduling keys; the
original PyPI 0.1.0 wheel does not provide these APIs.

## Shared tensors, R3 and SC

Each writer process reuses a pack writer. The slime adapter's default rotation
target is **256 MiB**, configured by `--rollout-queue-segment-mib`. Records and
manifests are embedded in packs; there is no file per sample or tensor. This
avoids the small-file metadata traffic that limits shared-filesystem throughput.
Queue and ownership metadata use append logs.

Completed R3 routes and SC tensors are published with their sample group after
custom sample hooks. Later rollout/continuation/training publications reuse
immutable tensor dependencies. Updates produce new records while earlier
references remain valid: this is tensor-level sharing and copy-on-write, not
GPU shared memory or page-level copy-on-write. R3 training reads only the rows
assigned to its CP/TP rank. Large bundles are split within the native write
budget; SC also works without R3.

Use `--rollout-data-transport straw` to persist R3/SC tensors. Object-store
transport keeps them in memory. There is no separate spill hook, per-sample
tensor file, or spill cleanup step. References from another straw storage pool
are copied into the destination pool before dependent results are accepted.
Debug dumps embed tensor contents so they remain readable after queue GC;
enabling dumps therefore adds host-memory and I/O overhead.

## Sample encoding and integrity

The adapter stores samples as a `slime.v1` JSON record `{version: 1, tree: ...}`
with explicit tags for containers, Sample fields, NumPy arrays, images and
rollout references. It preserves nested groups, supported dynamic fields,
multimodal inputs, R3, SC top-k and ragged top-p. Unsupported Python types and
cycles fail explicitly. The payload protocol uses no pickle and never imports
classes named by stored data.

Tensor nodes carry shape, dtype, semantic kind, lazy/validated flags and a
`{dependency: index, ordinal: index}` pointer into the enclosing manifest's
dependencies. Physical locations belong to those manifests, so publishing
another sample or training batch can share the same immutable tensor records.
All tensor dependencies must be published before the enclosing sample is accepted.

straw's `tensor.v1` stores contiguous, row-major, little-endian typed bytes.
Lazy readers verify metadata and the checksum of each chunk covering requested
rows; they do not verify untouched chunks. Full inspection verifies the whole
segment. Tensor descriptors are restored in batches to reuse shared pack indices.

## Enable it

`requirements.txt` includes `straw-queue`, so the standard Docker build and
normal slime installation install it automatically. For an existing environment,
install it on every generation and training node, using the same version throughout the job:

```bash
pip install straw-queue
```

Without straw installed, the default Ray `object-store` transport still works
and startup logs the installation command. Selecting `--rollout-data-transport straw`
without the package fails at startup with the same command.

The Python import name is `straw`. Add these arguments to your existing training
command, using a fresh shared directory:

```bash
--rollout-data-transport straw \
--rollout-data-dir /shared/juicefs/jobs/my-run/rollout_data \
--rollout-queue-run-id my-run \
--rollout-storage-profile juicefs \
--rollout-storage-declaration /shared/juicefs/deployment.json
```

The declaration records `direct_mount: true`, `writeback: false`,
`open_cache: 0`, `readdir_cache: false`, the deployed `client_version` and a
`durability_description`. It declares verified deployment settings; it does
not discover or configure the mount. See straw's
[filesystem contract](https://github.com/zhuzilin/straw/blob/main/docs/FILESYSTEM.md).
The default `local` storage profile represents the POSIX development contract;
using it on a shared mount does not verify that service's durability.

Storage and execution mode are separate choices. The default transport remains
`object-store`; the default rollout entrypoint remains synchronous. To select
distributed fully async execution with straw, also add:

```bash
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
```

If `--rollout-data-dir` is omitted, straw mode uses `<save>/rollout_data`; without
`--save`, an explicit shared directory is required. `--rollout-io-concurrency`
bounds off-event-loop serialization and I/O (default 4). The adapter uses worker
processes and bounded threads; straw does not create an I/O process pool.
`--rollout-queue-max-pending` and `--rollout-queue-max-inflight` bound task
admission (both default 65,536); they do not cap retained bytes or file count.

During weight synchronization, distributed producers pause new admissions and
allow in-flight work to finish and persist. They resume if another training
rollout remains. Weight synchronization itself remains a separate subsystem.
On shutdown, all generation workers share a five-minute deadline to finish
durable writes. Exceeding it reports an error rather than successful shutdown.

## Online GC and ownership

Online GC is disabled by default. Enable `--rollout-queue-online-gc` to let straw
reclaim unused sealed packs during a job. slime provides completion and discard
signals through straw APIs. A pack is deletable only after all task, publication,
queue, reader and checkpoint owners release it. A lease timeout alone does not
prove that a reader has stopped. A live record retains its entire pack; active
writers, retained checkpoints and WAL history still consume space.

GC failures stop the background loop, propagate the original cause through
later coordinator operations and surface at shutdown. They do not stop work
already running on remote workers. Retained data stays available for inspection.
Capacity exhaustion must be handled with
backpressure or an explicit operator decision; do not reset a live queue or
unlink pack files to make room. Offline removal requires all participants to
stop. Storage quotas and application retention policies remain necessary.

Processed positions and filter decisions are recorded explicitly, so gaps in
acceptance order do not hold already-consumed capacity. A separate bounded
control-task allowance lets the manager commit a collection under backpressure.

## Recovery and checkpoints

| Failure or restart | Supported behavior and boundary |
|---|---|
| Worker lost before acceptance | Recover the latest durable continuation; reassign with a new attempt |
| Acceptance committed but reply lost | Recover the accepted receipt without a second logical acceptance |
| Whole job stops before its first batch is planned | `--rollout-queue-resume` recovers accepted groups and durable prefixes with the same initial model and rollout configuration |
| Restart after batch planning | Requires matching model/optimizer and rollout checkpoints; queue-only recovery fails explicitly |

Resume requires the entire previous job, including its coordinator and readers,
to have stopped, and the same root and run ID. The flag is an operator assertion,
not automatic fencing. Unpublished inference may run again; SGLang GPU KV cache
is not restored.

Checkpoints contain `<save>/rollout/queue_state_<rollout_id>.json` for the source
and registered scheduler consumers, and `builder_state_<rollout_id>.json` for
the training consumer view. Their references retain the matching storage graph.
Continuations are persisted incrementally, so checkpoints reference
existing data instead of rewriting every returned token. Restoring a checkpoint
does not erase later accepted history from the WAL.

Partial R3/SC tensors remain lazy references in continuation snapshots and are
loaded when generation appends new rows. Restoring source and builder state
preserves saved warm groups and behavior-policy versions, and records which
later outputs are excluded from the restored branch. Restored pending tasks retain
their storage independently of filtering. The complete source snapshot,
including consumers that have not restarted yet, is retained and durably
recorded before old references are released. Failed saves keep the old references;
background GC starts only after restored consumer state commits.

The dataset producer cursor is not yet part of the source checkpoint, and
restoring a saved pending snapshot does not exclude every later-created task.
Complete rollback to an arbitrary older training step remains future work.
The adapter does not yet establish a joint final manifest for model, optimizer,
RNG and queue state. Persistent rollout is therefore not a claim of exact
optimizer recovery or bit-identical training after an arbitrary crash.

## Validation and remaining limits

CPU coverage includes codecs, R3/SC transport, lazy CP/TP reads, worker failures,
continuations, checkpoint views and GC ownership. straw separately checks its
Rust/Python core, bounded ownership model and native crash boundaries. Physical
multi-client filesystem checks and real GPU training complement these tests.
`tests/test_straw_fully_async_recovery.py` starts two local Ray nodes, interrupts
the entire driver/coordinator/generation process tree with SIGKILL, and starts a
new job against the same straw pool without a graceful checkpoint. It checks
accepted-result replay, unchanged token/logprob/R3/SC prefixes, new leases and
single reward evaluation, with online GC both disabled and enabled. Inference
and rewards are CPU fixtures; this test does not claim optimizer recovery.
It runs automatically in `cpu-unittest`; see [CI setup](../developer_guide/ci.md).

Run the distributed rollout and interruption recovery tests locally with:

```bash
PYTHONPATH=. python tests/test_distributed_rollout.py
PYTHONPATH=. python tests/test_straw_fully_async_recovery.py
```

The main deployment limits are shared-storage bandwidth, durability latency,
coordinator throughput and manager conversion memory. Live-pack compaction,
journal compaction and automatic coordinator failover are not implemented.
See [usage](../get_started/usage.md#persistent-rollout-queue-and-distributed-fully-async)
for the rollout controls and straw's
[verification guide](https://github.com/zhuzilin/straw/blob/main/docs/VERIFICATION.md)
for the storage checks and their scope.
