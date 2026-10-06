# Fault Tolerance

slime checks SGLang engines, removes failed engines from service, and restarts them before the next weight update. When you use Straw transport or save rollout debug data, you can also retain the serving cluster after a Megatron failure, adjust the training configuration, and resume training.

These features are enabled by default for serving clusters started by slime. You do not need `--use-fault-tolerance`; the flag remains for compatibility with older commands. External serving clusters retain their existing policy.

## Resuming After a Megatron Failure

This recovery workflow requires the **Ray cluster to remain running**. Losing the training driver or `RolloutManager` does not destroy the SGLang engines or routers. When you resubmit training, slime finds and reuses the existing serving cluster. This requires no changes to SGLang and does not automatically retry the training job.

### Retain Recovery Data from the First Launch

Choose either option below. slime automatically persists recovery data in these modes, independently of `--use-fault-tolerance`.

Use Straw to store rollout and training data:

```bash
--rollout-data-transport straw --rollout-data-dir /shared/my-run/queue
```

Or save rollout debug data. The path must include `{rollout_id}`:

```bash
--save-debug-rollout-data '/shared/my-run/rollout_{rollout_id}.pt'
```

Use `--save` and `--save-interval` to save model checkpoints regularly. Longer intervals mean more batches to retrain after a failure and more recovery data to retain.

Recovery requires `torch_dist` checkpoints with RNG state and, for a stateful optimizer, optimizer state. slime automatically enables an optimizer checkpoint format that supports resharding across parallel configurations. `--no-save-rng` is rejected.

With `--use-stateless-adam --no-save-optim`, Adam keeps no moments between optimizer steps, so recovery can omit optimizer tensors. Slime saves the LR/WD scheduler in `opt_param_scheduler.pt` inside each model checkpoint and restores it at the same boundary; a missing scheduler file prevents resume. FP32 master parameters are rebuilt from the saved model precision, so this mode does not promise bitwise-identical updates after a restart. Ordinary Adam still requires its optimizer checkpoint.

If saving a large MoE optimizer exhausts GPU memory, use `--distrib-optim-fully-reshardable-mem-efficient`. Megatron then gathers checkpoint tensors through Gloo on CPU, preserving resharding support while trading GPU temporary memory for host memory and save time. `--overlap-grad-reduce --ddp-bucket-size 40000000` bounds the temporary buffers used for each checkpoint gather; reserve host memory for the full optimizer state as well.

### Resubmit After the Failed Job Exits

For example, after a Megatron OOM:

1. Wait for the failed training job to exit.
2. Adjust TP/CP/EP, microbatch size, or the token limit per GPU.
3. Submit `train.py` again to the **same Ray cluster**.

Keep the model, rollout configuration, global batch size, and session identity unchanged. With colocated training and rollout, the trainers must fit within the retained GPU allocation. With separate training and rollout, you can allocate training resources again without moving the rollout GPUs.

Retained workers keep their original configuration. Sampling, filters, reward hooks, and custom arguments are checked on reattachment; adding or removing an argument also counts as a change. Only explicitly supported trainer and operational settings may change.

Between attempts, do not stop Ray, recreate the serving container, or run cleanup commands such as `pkill sglang`. `slime.utils.external_utils.command_utils.execute_train` preserves a running Ray head and SGLang. If your shell launcher cleans up processes on every launch, skip that step when restarting. Once training finishes successfully, slime releases the retained session and its resources.

### Where Training Resumes

The new trainer loads the last successfully saved model, scheduler, RNG, and any stateful optimizer state, then retrains the batches after that checkpoint. This includes batches that finished training but whose model updates were not saved. If no checkpoint exists yet, replay starts from the original model.

Retained data includes generated samples, tokens, and reward postprocessing results. When you change training parallelism, slime repartitions this data without regenerating samples or repeating reward postprocessing. Recovery data is kept until the corresponding model checkpoint commits or training finishes successfully. Serving weight versions continue to increase across restarts.

Megatron restores RNG state when the parallel layout is compatible and reinitializes it when TP/PP changes. You can therefore resume with a different layout, but the results are not guaranteed to be bitwise identical.

## Finding the Existing Serving Cluster

slime manages the serving cluster through a named Ray actor. A new job looks up that actor by its session name and gets the existing router and engine information from it. It does not scan for router processes.

Set `--rollout-session-id` to choose a session identity explicitly. Otherwise, slime selects an identity in the following order and hashes it to produce a stable name:

1. With Straw, use the absolute storage directory path and `--rollout-queue-run-id`.
2. Otherwise, use the absolute path template from `--save-debug-rollout-data`.
3. If debug data is not saved, use the absolute `--save` directory path.
4. If none of these is configured, use the model and rollout configuration.

Use the same identity for both attempts of a training run. Independent runs should use different identities to avoid connecting to the same serving cluster.

Three components handle recovery:

- `ServingCluster` manages routers, SGLang engines, GPU resources, the Straw queue controller, and the weight-update lock. It is a named, detached Ray actor that survives the job that created it.
- `RolloutManager` handles generation, data reading, sample conversion, and training-data partitioning. It can be reused or recreated after it exits; recreating it does not destroy resources held by `ServingCluster`.
- `TrainingRecovery` owns the model checkpoint boundary and retains the batches needed after it. The Straw accepted log owns raw and converted batch receipts; recovery reconciles those receipts if a manager or RPC fails before the journal is updated. A converted batch is stored once, and each DP rank receives an index view that can be rebuilt for a new parallel configuration.

If `RolloutManager` survives, it pauses admission of new generation tasks. If it exits, its replacement reconnects to `ServingCluster`, restores data-source progress, and uses the queue controller to prevent old readers from taking more tasks. Completed prefetch results that have not reached training remain available. Trainer processes are also registered with `ServingCluster`, so it can clean up old trainers even after the manager exits.

## Engine Health Checks and Restarts

During rollout, slime periodically requests SGLang's `/health_generate` endpoint to check whether each engine responds. It also checks immediately when rollout finishes, **regardless of the background interval or initial wait**.

Synchronous rollout removes failed services from the router before sending abort and drain requests. Before returning to training, `ServingCluster` checks the engines again, unregisters failed services, and clears their Ray actor handles. This prevents later memory-offload or weight-update requests from reaching stopped engines. Both HTTP requests and Ray calls have timeouts.

Missing engines restart before the next weight update and then load the trainer's weights. Background checks and rollout-completion checks use the same failure handling. Background checks pause while weights or memory allocation change.

Reattachment also bounds the time spent resetting old trainer connections. An engine that cannot acknowledge is unregistered and terminated with its serving children; healthy peers remain running. This applies to prefill and decode groups as well as ordinary serving groups.

| Argument | Default | Description |
|---|---|---|
| `--rollout-health-check-first-wait` | `0` seconds | Wait after each resumption of background checks to allow model warmup and kernel compilation. Rollout-completion checks bypass this wait. |
| `--rollout-health-check-interval` | `600` seconds | Interval between background health checks. |
| `--rollout-health-check-timeout` | `30` seconds | Timeout for each health check, including the wait for the Ray health-check call. |

For example, a large MoE model that needs more warmup time can use:

```bash
--rollout-health-check-first-wait 600 \
--rollout-health-check-interval 10 \
--rollout-health-check-timeout 5
```

Increase the timeout if load spikes cause false failures. If engines repeatedly fail after weight updates, inspect the SGLang logs and recent rollout dumps.

## Debugging Rollout and Training Separately

Saving rollout data lets you hold training inputs fixed while investigating training failures:

- `--debug-rollout-only`: initialize only rollout, without training. Combine it with the save option to inspect generation and reward results.
- `--save-debug-rollout-data /path/to/rollout_{rollout_id}.pt`: save samples from each rollout.
- `--load-debug-rollout-data /path/to/rollout_{rollout_id}.pt`: train on saved samples and skip SGLang initialization.
- `--debug-train-only`: initialize only training, without starting SGLang.

For slow generation requests, use [Trace Viewer](../developer_guide/trace.md) to inspect time spent on generation, rewards, and model calls, then use [Profiling](../developer_guide/profiling.md) to locate bottlenecks. See [SGLang Config](sglang-config.md) for multi-model or PD-separated deployments.

## Recovery Scope and Limitations

Without Straw or rollout debug dumps, you can still retain the serving cluster, but you cannot replay training batches after the last checkpoint.

Recreating `RolloutManager` supports the built-in data sources. Custom sources need compatible `state_dict` / `load_state_dict` methods and queue controllers that survive independently of the manager. Data-source constructors and rollout hook signatures are unchanged.

If `ServingCluster` or the entire Ray cluster is lost, you must start serving again and recover from a checkpoint. Cluster preemption and node loss still require coordination between your scheduler and checkpoint recovery.

See [Debugging](../developer_guide/debug.md) for more debugging options and [CI](../developer_guide/ci.md) for recovery-test coverage.
