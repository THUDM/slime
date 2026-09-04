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
For five engines and a limit of two, the waves are therefore `(0, 1)`, `(2, 3)`,
and `(4)`; no particular world size or number of engines is assumed.

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
