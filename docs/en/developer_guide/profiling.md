# Profiling

In slime, we can perform detailed performance analysis of the rollout process using the profiling interface provided by SGLang.

## 1. Sleeping the Rollout Process

For more flexible stress testing and profiling, it is often useful to make the slime rollout process enter a waiting state after initialization, instead of starting generation immediately.

You can achieve this by replacing the `rollout_function_path` in your startup arguments without modifying the source code:

```bash
python train.py \
    --rollout-function-path slime.rollout.sleep_rollout.sleep \
    ... (other arguments)
```

This function will make the rollout process enter an infinite wait loop, allowing you to manually send requests or run stress testing tools.

## 2. Obtaining SGLang Engine List

SGLang engines (workers) are registered with the router. You can retrieve the list of all active engines by accessing the `/workers` endpoint of the router.

The router address is typically printed in the startup logs:
```
Router launched at 127.0.0.1:3000
```

You can use `curl` to view the workers:
```bash
curl http://127.0.0.1:3000/workers
```

## 3. Using Automated Profiling Tool

To simplify profiling across multiple engines simultaneously, we provide an automated script: `tools/profile_rollout.py`.

### Starting Profiling

By default, this tool starts profiling on all workers and will automatically stop after 3 steps:

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action start --num-steps 3
```

**Key Parameters:**
* `--router-url`: The URL of the Router.
* `--num-steps`: Number of steps to record, defaults to 3.
* `--output-dir`: Directory where trace files will be saved.
* `--activities`: Activities to monitor, e.g., `GPU` `CPU`.
* `--profile-by-stage`: Whether to profile by stage (prefill/decode).

### Stopping Profiling Manually

If you did not set `num_steps` or wish to stop early:

```bash
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action stop
```

## 4. Running Stress Tests

While the Rollout process is in a waiting state via `sleep_rollout`, you can:
1. Start profiling using `tools/profile_rollout.py`.
2. Use stress testing tools (such as SGLang's built-in benchmark tools) to send requests to the router or directly to the engines.
3. Wait for profiling to complete (if `num_steps` was set) or stop it manually.
4. Collect the `.json` trace files from the `output_dir` and view them using `chrome://tracing` in Chrome or [Perfetto](https://ui.perfetto.dev/).
