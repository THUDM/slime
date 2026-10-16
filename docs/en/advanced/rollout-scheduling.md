# Synchronous, partial, and distributed fully async rollout

These are three separate decisions: **when training consumes a batch**, **whether unfinished generation can continue**, and **where rollout state lives**. The {ref}`lab <lab>` shows their combined timeline and exports their actual flags.

| Choice | Generation at a training boundary | Where unfinished work goes |
|---|---|---|
| Synchronous baseline | Finish the target batch, train, update weights | No retained partial work in the lab baseline |
| Synchronous + partial rollout | Oversample; once enough complete groups are accepted, stop the remaining work and train | Save prefixes, resume them in a later rollout |
| Fully async + object-store | A manager-local background worker keeps a queue across calls | Returned aborted groups go back to the data buffer |
| Fully async + straw | Independent generation processes on eligible Ray nodes feed the global batch collector | Persistent queue tasks, partial groups, ready groups and packed tensors |

Separate training/serving GPUs permit overlap. Fully async still pauses admission around weight publication and coordinates in-flight work; it does not remove every synchronization point. straw can also be used with synchronous execution.

(partial)=
## What partial rollout saves

Suppose group A finishes quickly, while group B has generated only a prefix. Once the target number of **complete accepted groups** is ready, the synchronous sampler can abort remaining requests, train, update weights, and later continue B from its prefix. Partial rollout preserves generated work; it does not make an unfinished response a complete rewarded training sample.

```{mermaid}
sequenceDiagram
    participant S as SGLang
    participant Q as Buffer / straw
    participant T as Trainer
    S->>S: Generate oversampled groups under v0
    S->>T: Target batch of complete accepted groups
    S->>Q: Unfinished B: tokens + sampler metadata
    T->>S: Train, then publish v1
    Q->>S: Resume B from its saved prefix
    S->>T: Completed B, possibly spanning v0 and v1
```

The lab exports `--partial-rollout`; for the synchronous recipe it also sets `--over-sampling-batch-size 16` with a target `--rollout-batch-size 8`. Oversampling supplies spare groups so slow unfinished groups can be retained. The standard sampler checks completion at group boundaries; a slow member can keep its group unfinished. Without excess in-flight work there may be little partial work to save.

For a response whose first $k$ tokens were generated under version $v_0$ and remaining tokens under $v_1$, its behavior distribution is

$$
q(y\mid x)=\prod_{t=1}^{k}q_{v_0}(y_t\mid x,y_{<t})
\prod_{t=k+1}^{T}q_{v_1}(y_t\mid x,y_{<t}).
$$

Each sampled token's log-prob belongs to its actual generating policy. R3 routes and SC sampler heads must follow the same token alignment. A token-level importance ratio uses $p_\theta(y_t\mid h_t)/q_{v(t)}(y_t\mid h_t)$; it cannot substitute one current sampler for the entire mixed-version response. See the [importance-weight derivation](policy-mismatch.md#tis).

`--mask-offpolicy-in-partial-rollout` sets loss masks of an existing prefix to zero on continuation. If $m_t=0$ for $t\le k$ and $m_t=1$ otherwise, the token objective becomes $\sum_t m_t\ell_t$ (with the selected loss normalization). This removes those old tokens' loss contribution. Later tokens still condition on the old prefix, so masking does not restore a fully on-policy trajectory distribution. It also reduces the amount of learning signal per generated token.

Custom agent hooks must preserve continuation tokens, masks and required metadata, and implement compatible abort behavior. An external tool side effect is not automatically resumable merely because its text transcript was saved. The source of truth is [the standard sampler](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py) and [hook contracts](../get_started/customization.md).

(distributed)=
## From a warm local queue to distributed producers

Both async paths use `slime.rollout.fully_async_rollout.generate_rollout_fully_async`. The default object-store data source selects a background worker in the manager process. Selecting straw gives the data source queue readers and selects `DistributedRollout` instead. Each eligible, alive Ray node with CPU resources receives one generation actor. These CPU producers send requests to SGLang engines; they are not additional model replicas.

```{mermaid}
flowchart LR
    Q[straw prompt / partial tasks] --> W0[Node 0 producer]
    Q --> W1[Node 1 producer]
    Q --> WN[Node N producer]
    W0 --> S[SGLang GPU engines]
    W1 --> S
    WN --> S
    S --> R[Persist completed groups + tensors]
    R --> B[Global batch collector]
    B --> T[Megatron]
    T -->|weights + admission coordination| S
```

Fast workers can provide the next batch without a barrier requiring every worker to finish. Bounded completed prefetch applies backpressure when training falls behind. Weight-update coordination pauses admission and preserves results before resuming. Aborted groups are rebuffered rather than treated as completed training data. Storage and continuation remain distinct: the synchronous partial flag controls retaining unfinished work at its batch boundary; the async workers have their own aborted-group handling.

Let $C$ be `sglang_server_concurrency`, $E$ the number of logical rollout engines, $G$ responses per prompt, and $W$ eligible worker nodes. The distributed implementation divides

$$K=\left\lfloor\frac{CE}{G}\right\rfloor$$

group slots across workers, with quotient/remainder allocation. Each worker must receive at least one group, so $CE\ge WG$. This is a concurrency budget, not throughput; engine latency, agent tools, reward work and storage matter. Engine counting follows slime's serving configuration, including PD groups, rather than raw GPU count. See [worker selection](https://github.com/THUDM/slime/blob/main/slime/rollout/fully_async_rollout.py) and [distributed scheduler](https://github.com/THUDM/slime/blob/main/slime/rollout/fully_async_distributed.py).

## Staleness and throughput

If version $v$ is currently served and a group's oldest generated token used version $u$, its staleness is $v-u$. A freshly completed response can still be stale if its prefix began earlier. straw prioritizes completed groups, then partial groups, then fresh prompts; within a stage, older generated-token versions come first, with FIFO ties. Ordering does not automatically discard stale samples.

If producers sustainably generate $\lambda$ accepted groups/s and training consumes $\mu$, throughput cannot exceed $\min(\lambda,\mu)$. With $\lambda>\mu$, an unbounded backlog grows at roughly $\lambda-\mu$; backpressure is needed. In a stable queue, Little's-law accounting gives $L=\lambda W_q$ for mean queued group count $L$ and time in that queue $W_q$. More producers are useful only while the rest of the system can consume their output.

Watch `staleness/mean`, `staleness/max`, accepted-group throughput, reward, masking/clipping rates and training/rollout log-prob difference. Off-policy corrections do not guarantee that arbitrary lag is harmless. The [systems cost derivation](rl-systems.md#async) and [PipelineRL paper](https://arxiv.org/abs/2509.19128) provide related pipelining background; slime's implementation above defines the actual behavior.

## Persistence and recovery with straw

The lab asks for a shared JuiceFS directory, run ID, and deployment declaration, and exports:

```bash
--rollout-data-transport straw \
--rollout-data-dir /shared/juicefs/jobs/my-run/rollout_data \
--rollout-queue-run-id my-run \
--rollout-storage-profile juicefs \
--rollout-storage-declaration /shared/juicefs/deployment.json
```

Install the same `straw-queue` version on all nodes; mount the verified storage at the same absolute path. The declaration records mount/durability properties, it does not create the mount. straw stores prompt tasks, partial/ready groups, training batches and packed tensors. Leases allow unfinished work from lost workers to become available again. R3 and SC tensor records can be shared across stages without copying their bytes.

The short straw recipe saves every round. Recovery needs a completed **model/optimizer plus rollout-state checkpoint** and the referenced storage pool. It does not restore GPU KV cache or guarantee identical future random generations. Keep dataset, tokenizer, run ID, storage settings and fully async worker topology compatible. Stop the previous job before a whole-job restart; coordinator failover is not automatic. For retained serving, follow [fault tolerance](fault-tolerance.md). Full setup, leases, GC and branch/rollback semantics are in [the straw guide](straw.md) and [straw's filesystem contract](https://github.com/zhuzilin/straw/blob/main/docs/FILESYSTEM.md).

The continuous async path does not support inline evaluation or `--rollout-all-samples-process-path`; evaluate separately. straw does not support `--buffer-filter-path`, and uses its own persisted ordering. See [debugging](../developer_guide/debug.md) for replay and [profiling](../developer_guide/profiling.md) before changing concurrency.
