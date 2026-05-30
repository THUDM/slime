# Fault Tolerance

Long-running RL jobs fail in different ways from short supervised runs. Rollout engines can hang, long-tail samples can keep a round open, and serving state must be refreshed after weight updates. slime's fault-tolerance support focuses on making the rollout side observable, restartable, and debuggable without changing the training / rollout / Data Buffer loop.

Enable fault tolerance with:

```bash
--use-fault-tolerance
```

## Current Scope

slime currently provides rollout-engine fault tolerance:

- health checks for SGLang rollout servers;
- timeout-based rollout server restart;
- correct parameter update after restart;
- debug rollout dumps for replaying training-side issues without rerunning rollout;
- trace/profiling hooks for inspecting long-tail rollout behavior.

Cluster-level preemption, trainer-rank failure, and full-job resume should still be handled through your cluster scheduler, Ray restart policy, and slime checkpointing. In practice, production jobs combine rollout fault tolerance with frequent checkpoints and debug dumps.

## Rollout Health Checks

During rollout, slime periodically sends heartbeat requests (`/health_generate`) to all SGLang servers. If a heartbeat times out, the unhealthy SGLang server is stopped. After the current rollout round completes, slime restarts the server and updates it with the correct parameters before it serves future rollout requests.

The main arguments are:

- `--rollout-health-check-first-wait`: wait before starting heartbeat checks for the first rollout. Large MoE models may compile kernels on first run. Default: `300` seconds.
- `--rollout-health-check-interval`: interval between heartbeat checks. Default: `10` seconds.
- `--rollout-health-check-timeout`: timeout for one heartbeat request. Default: `5` seconds.

Example:

```bash
--use-fault-tolerance \
--rollout-health-check-first-wait 600 \
--rollout-health-check-interval 10 \
--rollout-health-check-timeout 5
```

## Debug and Replay Path

Fault tolerance is more useful when failures are reproducible. slime provides separate rollout-only and train-only debugging paths:

- `--debug-rollout-only`: run rollout and save generated data without training;
- `--save-debug-rollout-data /path/to/rollout_{rollout_id}.pt`: save rollout samples for later inspection or replay;
- `--load-debug-rollout-data /path/to/rollout_{rollout_id}.pt`: replay saved rollout data and skip SGLang initialization;
- `--debug-train-only`: run training-side logic without rollout.

This lets you isolate whether a failure belongs to serving/rollout, data conversion, reward/verifier logic, or Megatron training.

## Recommended Production Pattern

For long-running jobs:

1. Enable `--use-fault-tolerance`.
2. Save checkpoints regularly with `--save-interval`.
3. Save rollout debug dumps for new agentic or verifier-heavy workloads.
4. Use [Trace Viewer](../developer_guide/trace.md) to inspect long-tail samples and reward/model-call spans.
5. Use [Profiling](../developer_guide/profiling.md) to separate rollout bottlenecks from training bottlenecks.
6. Keep SGLang deployment explicit with [SGLang Config](sglang-config.md) for complex multi-model or PD topologies.

## What to Watch

- If startup health checks fail on large MoE models, increase `--rollout-health-check-first-wait`.
- If transient load spikes cause false positives, increase `--rollout-health-check-timeout`.
- If a server repeatedly restarts after weight sync, inspect the SGLang logs and the latest rollout debug dump.
- If the trainer fails rather than rollout, resume from checkpoint and use debug replay to isolate whether the saved rollout batch is valid.

## Related Docs

- [Debugging](../developer_guide/debug.md)
- [Trace Viewer](../developer_guide/trace.md)
- [Profiling](../developer_guide/profiling.md)
- [CI](../developer_guide/ci.md)
