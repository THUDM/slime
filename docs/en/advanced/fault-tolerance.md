# Fault Tolerance

Long-running RL jobs fail in different ways from short supervised runs. Rollout engines can hang, long-tail samples can keep a round open, and serving state must be refreshed after weight updates. slime's fault-tolerance support focuses on making the rollout side observable, restartable, and debuggable without changing the training / rollout / Data Buffer loop.

Internal serving always has an independent, detached owner, health checks and unhealthy-engine recovery. The following flag is retained for compatibility and the existing external-serving policy:

```bash
--use-fault-tolerance
```

## Current Scope

slime provides rollout-engine fault tolerance and manual Megatron restart:

- health checks for SGLang rollout servers;
- timeout-based rollout server restart;
- correct parameter update after restart;
- debug rollout dumps for replaying training-side issues without rerunning rollout;
- trace/profiling hooks for inspecting long-tail rollout behavior.
- retain SGLang servers and replay training data after a Megatron failure, when Straw transport or debug rollout dumps are enabled.

Cluster-level preemption and loss of the Ray cluster still require your cluster scheduler and slime checkpointing. A retained serving session survives training driver and rollout manager failures within the same live Ray cluster. Internal serving follows the same lifetime and health policy with or without the compatibility flag. External-cluster behavior is unchanged.

## Manual Megatron Restart

The restart path has three responsibilities:

- `ServingCluster` is the named, detached owner of routers, engines, placements,
  the queue controller and the weight-update lock. A new driver finds it by the
  stable session name, rather than scanning for router processes.
- `RolloutManager` owns generation, data readers, conversion and trainer shards.
  It uses engine snapshots from the owner; losing this manager leaves serving alive.
- `TrainingRecovery` journals the source cursor and accepted batches together.
  Converted batches are saved before DP sharding, so a restarted trainer may
  change parallelism. Batches are retained until their model checkpoint commits.

At rollout completion, dead workers are unregistered before abort/drain, and the
owner checks once more before offload or weight-update controls. Recovery starts
replacement engines just before the trainer installs their weights. Periodic
health checks and this boundary check share the same owner and failure handling.

For replayable training recovery, use either mode below. Persistence is always enabled for these modes, independently of `--use-fault-tolerance`.

```bash
--rollout-data-transport straw --rollout-data-dir /shared/my-run/queue
```

or:

```bash
--save-debug-rollout-data '/shared/my-run/rollout_{rollout_id}.pt'
```

If Megatron runs out of memory, wait for the failed training job to terminate, fix its training configuration, and submit `train.py` again to the **same Ray cluster**. You can change TP/CP/EP, microbatch sizing, and token limits. Colocated trainers must fit within the retained GPU placement; separate trainers can allocate a new training placement without moving rollout GPUs. Keep the model, rollout configuration, global batch size, and session identity unchanged. Set `--rollout-session-id` to choose the serving identity explicitly. Otherwise Straw uses its storage directory and `--rollout-queue-run-id`, debug mode uses the dump path template, and other modes use the save directory or model/rollout configuration. Use a different identity for an independent training run.

The serving owner keeps SGLang processes, routers, rollout GPU placements and the Straw controller alive independently of the rollout manager. When the manager survives, it pauses producer admission. If it was killed, resubmission creates a new manager against the same owner and loads its atomic recovery journal. Built-in data sources restore their cursor and metadata; the live Straw controller fences old readers and recovers accepted prefetch results. Trainer ranks are registered with the serving owner, so it can release them even after manager death. The new trainer reconnects to those engines and reloads the last successfully saved model, optimizer and RNG state. Batches after that checkpoint are replayed, including completed training batches whose model updates were not checkpointed. Without a saved checkpoint, replay starts from the original model. Converted rewards and token data are retained before DP splitting, so changing parallelism rebuilds the partitions without regenerating samples or running reward postprocessing again.

Use `--save` and `--save-interval` to bound replay work and storage. Recovery uses `torch_dist` checkpoints and enables fully reshardable optimizer saves for parallelism changes. Saves must include optimizer and RNG state; `--no-save-optim` and `--no-save-rng` are rejected. Megatron restores RNG state when the topology is compatible and reinitializes it when TP/PP changes, so recovery across different parallel layouts is not bitwise replay. Debug dumps must include `{rollout_id}` in the path. Recovery files stay retained until a model checkpoint commits or training finishes successfully. The serving weight version continues to increase across trainer restarts.

Do not stop Ray, recreate the serving container, or run cleanup commands such as `pkill sglang` between attempts. `slime.utils.external_utils.command_utils.execute_train` preserves a running Ray head and SGLang for every launch; shell launch scripts with unconditional cleanup must skip that cleanup on restart. Successful training disposes the retained session. This is a manual restart workflow, not an automatic trainer retry, and it does not require changes to SGLang itself.

Without Straw or debug dumps, serving still survives failures, but uncheckpointed training batches cannot be replayed. Custom data sources keep their existing constructor and rollout hook signatures; manager reconstruction is supported for the built-in sources. Custom sources need compatible `state_dict` / `load_state_dict` methods and controllers whose lifetime is independent of the manager. Losing the serving owner or the Ray cluster requires a cold restart from checkpoints.

## Rollout Health Checks

During rollout, slime periodically sends heartbeat requests (`/health_generate`) to all internal SGLang servers. At the rollout boundary it also checks immediately, regardless of interval or warmup grace. Synchronous rollout prunes failed router workers before abort/drain requests; the serving owner then unregisters failed engines and removes their actor handles before offload or other control requests. These checks have bounded HTTP and Ray RPC waits. Failed engines restart before the next weight update.

The main arguments are:

- `--rollout-health-check-first-wait`: grace before background checks after resume. Large MoE models may compile kernels on first run. Boundary checks bypass this grace. Default: `0` seconds.
- `--rollout-health-check-interval`: interval between background checks. Default: `600` seconds.
- `--rollout-health-check-timeout`: timeout for a heartbeat request or queued health RPC. Default: `30` seconds.

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

1. Tune the always-on internal health checks with `--rollout-health-check-*` for model warmup and response latency.
2. Save checkpoints regularly with `--save-interval`.
3. Save rollout debug dumps for new agentic or verifier-heavy workloads.
4. Use [Trace Viewer](../developer_guide/trace.md) to inspect long-tail samples and reward/model-call spans.
5. Use [Profiling](../developer_guide/profiling.md) to separate rollout bottlenecks from training bottlenecks.
6. Keep SGLang deployment explicit with [SGLang Config](sglang-config.md) for complex multi-model or PD topologies.

## What to Watch

- If startup health checks fail on large MoE models, increase `--rollout-health-check-first-wait`.
- If transient load spikes cause false positives, increase `--rollout-health-check-timeout`.
- If a server repeatedly restarts after weight sync, inspect the SGLang logs and the latest rollout debug dump.
- If the trainer fails, correct its configuration and resubmit within the retained session. If the Ray cluster was lost, resume from a durable checkpoint and use debug replay to inspect the failed batch.

## Related Docs

- [Debugging](../developer_guide/debug.md)
- [Trace Viewer](../developer_guide/trace.md)
- [Profiling](../developer_guide/profiling.md)
- [CI](../developer_guide/ci.md)
