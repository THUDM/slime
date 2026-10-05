# Fault Tolerance

Long-running RL jobs fail in different ways from short supervised runs. Rollout engines can hang, long-tail samples can keep a round open, and serving state must be refreshed after weight updates. slime's fault-tolerance support focuses on making the rollout side observable, restartable, and debuggable without changing the training / rollout / Data Buffer loop.

Enable fault tolerance with:

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

Cluster-level preemption and loss of the Ray cluster still require your cluster scheduler and slime checkpointing. A retained serving session survives a training driver failure within the same live Ray cluster.

## Manual Megatron Restart

Combine `--use-fault-tolerance` with either:

```bash
--rollout-data-transport straw --rollout-data-dir /shared/my-run/queue
```

or:

```bash
--save-debug-rollout-data '/shared/my-run/rollout_{rollout_id}.pt'
```

If Megatron runs out of memory, wait for the failed training job to terminate, fix its training configuration, and submit `train.py` again to the **same Ray cluster**. You can change TP/CP/EP, microbatch sizing, and token limits. Colocated trainers must fit within the retained GPU placement; separate trainers can allocate a new training placement without moving rollout GPUs. Keep the model, rollout configuration, global batch size, and session identity unchanged. Straw identifies the session by its storage directory and `--rollout-queue-run-id`; debug mode uses the dump path template. Use a different identity for an independent training run.

The retained rollout manager stops producer admission and releases failed trainer ranks, while keeping SGLang processes, routers and rollout GPU placements alive. The new trainer reconnects to those engines and reloads the last successfully saved model, optimizer and RNG state. Batches after that checkpoint are replayed, including completed training batches whose model updates were not checkpointed. Without a saved checkpoint, replay starts from the original model. Converted rewards and token data are retained before DP splitting, so changing parallelism rebuilds the partitions without regenerating samples or running reward postprocessing again.

Use `--save` and `--save-interval` to bound replay work and storage. Recovery uses `torch_dist` checkpoints and enables fully reshardable optimizer saves for parallelism changes. Saves must include optimizer and RNG state; `--no-save-optim` and `--no-save-rng` are rejected. Megatron restores RNG state when the topology is compatible and reinitializes it when TP/PP changes, so recovery across different parallel layouts is not bitwise replay. Debug dumps must include `{rollout_id}` in the path. Recovery files stay retained until a model checkpoint commits or training finishes successfully. The serving weight version continues to increase across trainer restarts.

Do not stop Ray, recreate the serving container, or run cleanup commands such as `pkill sglang` between attempts. `slime.utils.external_utils.command_utils.execute_train` preserves Ray and SGLang for these recovery modes; shell launch scripts with unconditional cleanup must skip that cleanup on restart. Successful training disposes the retained session. This is a manual restart workflow, not an automatic trainer retry, and it does not require changes to SGLang itself.

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
- If the trainer fails, correct its configuration and resubmit within the retained session. If the Ray cluster was lost, resume from a durable checkpoint and use debug replay to inspect the failed batch.

## Related Docs

- [Debugging](../developer_guide/debug.md)
- [Trace Viewer](../developer_guide/trace.md)
- [Profiling](../developer_guide/profiling.md)
- [CI](../developer_guide/ci.md)
