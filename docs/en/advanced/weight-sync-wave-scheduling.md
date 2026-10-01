# Weight-Sync Engine Waves

Large rollout deployments can make every SGLang engine receive or load the same
weight bucket at once. That maximizes fan-out, but can also create a short burst
of GPU, PCIe, NVLink, network, or storage traffic. slime can bound this fan-out
without adding a hardware-specific time delay:

```bash
--update-weight-max-inflight-engine-groups 2
```

An engine group here means one logical SGLang engine. Its tensor-, pipeline-, or
multi-node workers remain one indivisible group. The scheduler walks the resolved
engine list in stable order and admits at most the configured number at once.
For four engines and a limit of three, the waves are therefore `(0, 1, 2)` and
`(3)`; no particular world size or number of engines is assumed.

The default is `0`, which keeps the existing all-at-once behavior. A value at
least as large as the engine count is equivalent to the default. Negative values
are rejected.

## Supported transports

- Non-colocated NCCL weight sync creates one process group per logical engine
  only when a real bound is requested. Broadcasts for the engines in one wave
  are launched asynchronously, and the next wave is not admitted until both the
  NCCL work and engine-side load requests complete. The default continues to use
  the original aggregate process group.
- Colocated tensor/IPC sync coordinates the same deterministic waves across all
  trainer ranks. A trainer control-group barrier closes each wave before source
  buffers can be reused.
- Full and delta disk sync apply the bound to checkpoint pulls and engine reload
  requests.
- Quantized post-load processing uses the same engine-group bound.

The existing pause, flush, and weight-version boundaries are unchanged: an engine
is not resumed early merely because its own wave completed. Disk-backed checkpoint
pulls may run before the pause, but the serving weights are not changed by that
prefetch. Consequently, wave admission does not introduce a new mixed-version
serving interval.

## Choosing a value

Start with `0` as the throughput baseline, then compare small limits such as `1`,
`2`, and `4` on the deployment topology that will run the job. Select the largest
value that avoids the observed traffic or memory burst without increasing total
weight-sync time or tail latency. This option limits concurrent engine groups;
`--update-weight-buffer-size` independently controls the size of each weight
bucket.

## Production-callsite confirmation

### Combining waves with bucket credits

The integrated updater supports engine-group waves together with
`--update-weight-max-inflight-buckets` and `--update-weight-max-inflight-bytes`.
One logical bucket reservation spans all its engine waves. A wave must finish
device transport, engine-load acknowledgement and staging release before the
next wave is observed; the logical credit is returned only after the last wave.
The final weight version is published after all consumers resume.

When the engine population is split into multiple waves, buffered buckets are
drained one at a time. The bucket window still bounds admitted logical bytes,
but does not multiply the active engine-group cap or promise inter-bucket
overlap. A single aggregate communicator retains the asynchronous bucket-window
path. Logical bytes are not a physical CUDA-memory cap or measured wire bytes.
Transport credit uses a host-visible stream fence after Work.wait; that wait
alone is only a CUDA dependency. Failed loads poison the version, prevent the
next wave/publication, and retain active-wave resources on the updater. Recover
by recreating the failed updater/process; automatic partial-version retry is
not supported.

CPU integration tests exercise both production updater variants with 2/4
logical engine groups, including load failure on a peer and staging retention.
They are not a complete Ray/SGLang GPU training validation.

`tools/benchmark_weight_sync_callsite.py` invokes the production
`update_weights_in_engine_group_waves` function with real process groups and
synthetic payloads. It is intentionally a callsite/module probe: it does not
replace the scheduler, change defaults, or claim Ray/SGLang load performance.

Use a four-process launch to retain three real two-rank
`[trainer, engine]` groups. A/B are engine groups 0 and 1; the third group is
the competing operation that distinguishes the all-at-once and window-2
policies. Only use device counts permitted by the test environment.

Each command below creates exactly one independent process-run artifact:

```bash
torchrun --standalone --nproc-per-node=4 \
  tools/benchmark_weight_sync_callsite.py \
  --backend gloo --policy candidate_windowed \
  --evidence-role selection --run-id selection-windowed-00 \
  --order ab --output-json /tmp/selection-windowed-00.json
```

In a minimal PyTorch container without slime's Ray/Megatron control-plane
dependencies, set `SLIME_CALLSITE_SOURCE_LOAD=1`. That opt-in mode loads the
exact production source file while stubbing only unused actor/conversion
imports; the artifact records `source_with_control_plane_stubs` so it cannot be
silently mixed with an installed-runtime campaign.
When the checkout's `.git` metadata is not mounted into the container, pass the
tested revision as `SLIME_BENCHMARK_SOURCE_COMMIT`.

Run every policy in both `selection` and `confirmation` roles in separate
process launches (five launches per policy/role by default), alternating
`--order ab` and `--order ba`. Then validate and summarize the immutable raw
artifacts:

```bash
python tools/benchmark_weight_sync_callsite.py \
  --summarize /tmp/weight-sync-callsite/*.json \
  --min-runs-per-role 5 \
  --summary-json /tmp/weight-sync-callsite-summary.json
```

The summary fails closed on duplicate process/run identity, incomplete rank
coverage, payload mismatch, mixed runtime/message/topology cells, or too few
independent runs. It reports communication A/B, rank-local pair makespan,
receiver consumer waits, callsite return, device readiness, and whole-stage
sync readiness separately. NCCL intervals are labeled `event_bracket`, never
`kernel_observed`. The tool records PyTorch/CUDA/NCCL versions, launch-order
configuration, dtype, message geometry, graph state, hostname, device identity,
and exact process-group membership.

This evidence does not establish end-to-end training throughput, SGLang load
latency, multi-node behavior, or a production policy winner. The summarizer
does not automatically select or apply a policy.
