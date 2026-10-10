# Debugging

## Aligning Precision

During the development of slime, it is often necessary to check if the model's precision is correct. This can be verified in the following ways:

1.  **First Training Step**
    1.  Check if the generated `rollout` is coherent. If not, there are two possible reasons:
        * Parameters were not loaded correctly. You need to check the logs for a confirmation that Megatron successfully loaded the checkpoint (ckpt).
        * There was an error in updating the parameters. You can check if all parameters were converted and mapped correctly, or if the parameter names were converted according to the parallelization strategy (e.g., when `pp_size > 1`, check if the layer IDs for the parameters provided by the second stage are correct). A thorough method is to save all parameters in the `load_weights` implementation of the corresponding model in SGLang and verify that they are consistent with the loaded checkpoint.
        * If all parameters are updated correctly and the problem persists, it's possible that some special buffers in SGLang were released during the release process.
        * If you are testing with a pretrained model, you can switch to an instruct version of a model with the same architecture to see if this garbled output is specific to the pretrained model.

    2.  Check the printed rollout stats to see if `log_probs` and `ref_log_probs` are exactly equal (meaning KL divergence is 0 in the first step) and their values are small.
        * If they are not exactly equal, it is usually caused by certain non-deterministic kernels in the Transformer Engine, for example:
            * In some versions of Transformer Engine (TE), Megatron requires `--attention-backend flash` to enforce the use of Flash Attention, thereby avoiding numerical instability from the fused attention under Context Parallelism (CP).
        * If the values are large (e.g., > 1), there are generally two possibilities:
            * If the value is extremely large, there is likely a problem with the training configuration.
            * If the value is only slightly larger than the SFT loss, for example, if the log probability of an instruct model reaches 0.8, it might be because the data does not conform to the trained chat template or does not match the cold-start distribution.

    3.  When running one inference step per training step (`num_steps_per_rollout == 1`), check if the KL divergence is 0 and if the `grad_norm` is small.
        * This is basically due to some Megatron / TE related bugs, for example:
            * Mixture of Experts (MoE) requires enabling `--moe-permute-fusion`.

2.  **Second Training Step**
    1.  For integrated training and inference, check if the second step can be loaded correctly and whether it results in an Out of Memory (OOM) error.

## Separate Debugging for Training and Inference

slime supports debugging the training and inference parts separately, which allows for the following:

* When tuning/debugging the inference part, you can start the task with only a few GPUs.
* When tuning/debugging the training part, you can ensure the model input is fixed, removing the randomness of rollouts.

Specifically, slime currently provides the following parameters for separate debugging:

1.  `--debug-rollout-only`

    When enabled, slime will not load Megatron and will only initialize SGLang. You can use this method to debug the inference part.

2.  `--debug-train-only`

    When enabled, slime will not load SGLang and will only initialize Megatron. You can use this method to debug the training part.

3.  `--save-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt`

    When enabled, the results of each rollout will be saved. This can be used in conjunction with `--debug-rollout-only`. Note that the data is saved using the format: `args.save_debug_rollout_data.format(rollout_id=rollout_id)`.

4.  `--load-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt`

    When enabled, data will be loaded from `args.load_debug_rollout_data.format(rollout_id=rollout_id)`, and SGLang will not be initialized (automatically setting `debug_train_only=True`). This method allows you to fix the input for the training part to tune it, for example, by switching between different parallelization strategies.

5.  `--save-debug-train-data /your/saved/debug/train_{rollout_id}.pt`

    Saves one train-side file per rollout. Only the last Pipeline Parallel stage and Tensor Parallel rank 0 participate. They restore response-token fields such as `log_probs`, `ref_log_probs`, `values`, `advantages`, `returns`, `kl`, and `entropy` across Context Parallel ranks. Context Parallel rank 0 moves each restored tensor to CPU immediately, so complete tensors do not accumulate on the GPU, and then gathers the distinct Data Parallel shards to one writer.

    The version-2 payload mirrors the rollout debug dump: a top-level `samples` list holds one dict per training sample (`sample_index`, `data_parallel_rank`, and its per-sample fields such as `tokens`, `log_probs`, `advantages`), sorted by `sample_index` so it lines up one-to-one with the rollout dump's `samples` (join on `sample_index` ↔ the rollout side's `index`). A parallel `dp_shards` key preserves the DP/micro-batch layout — each entry records `rank`, `data_parallel_rank`, that shard's `sample_indices`, and the DP-local schedule (`micro_batch_indices`, `num_microbatches`, `global_batch_sizes`) — without duplicating any per-sample tensor. Whole-batch fields such as `raw_reward` are stored once at the top level. If any sample lacks a `sample_index` (custom rollouts that build fresh `Sample` objects leave it `None`), the samples stay in DP-gather order and a warning is logged. With or without CP, response-token fields use the same full-response format. In configs that skip the separate actor log-prob recompute (`can_reuse_log_probs_in_loss` or `--use-rollout-logprobs`), the actor `log_probs` are snapshotted from the training forward itself (keyed by rollout position, at no extra forward), so the dump still carries them.

## Training Hangs at the First Step After a Rollout (Colocate)

A colocate run that finishes its rollout and then goes quiet at `Timer actor_train start` — SGLang health checks keep printing, no training output, the NCCL watchdog fires at 600 s or nothing fires at all — is usually not a communication deadlock. Look at what the ranks are doing before you look at NCCL:

```bash
py-spy dump --pid <pid of a MegatronTrainRayActor>   # one rank per node, per pipeline stage
```

Two stack shapes explain most of these hangs:

1. **One pipeline stage is compiling, the next is waiting for it.** The first stage sits inside `triton/.../compiler.py`, reached from a Triton kernel such as `fla/ops/gated_delta_rule` (GDN models such as Qwen3.5 / Qwen3.6); the next stage sits in `recv_forward` → `_communicate_shapes` → `torch.cuda.synchronize`. This is JIT compilation and autotuning, not a deadlock. Triton caches under `~/.triton/cache` inside the container, so every fresh node starts cold, and a single-node smoke run on the same node is what makes a later multi-node run "work". Measured on a slime fork with Qwen3.6-35B-A3B on 2×8 H20: the cold first `actor_train` took 639 s (the log-prob pass before it 394 s); the same passes took 60 s and 13 s once the cache was warm. Warm the cache before the multi-node launch, or size `--distributed-timeout-minutes` to cover the first step.

Where that cache lives is a genuine trade-off. Pinning `TRITON_CACHE_DIR` to node-local storage keeps compilation off a shared filesystem — the reason some large-scale recipes pin it to `/tmp`, having seen the NFS defaults race across nodes under many-process cold compiles — but it pays the compile again on every fresh node. A shared `TRITON_CACHE_DIR` pays a slower, contended first compile and then costs nothing. One measurement of the case the node-local pin guards against — 16 ranks across two nodes compiling the same GDN kernels concurrently into a single NFS directory — did not reproduce the races: no errors, and a second run on both nodes read the cache back in 0.7 s rather than 145 s, adding no new entries. That is one filesystem and one kernel set, so measure yours before sharing a cache.

2. **The timeout you set is not the timeout you got.** Colocate destroys every process group in `sleep()` and rebuilds it in `wake_up()` on each rollout→train switch. Before #2208 (shipped in v0.3.1) the rebuild called `new_group(ranks, backend="nccl")` and dropped `timeout`, `pg_options` and `group_desc`, so after the first offload cycle every group ran with the default 10-minute timeout regardless of `--distributed-timeout-minutes`. A hang that "ignores" the flag and dies at exactly 600 s on an older release is this combined with a slow first step, not a deadlock. Upgrade, or verify with a two-process probe that reads `pg._get_backend(device).options._timeout` before and after `destroy_process_groups()` / `reload_process_groups()`.

If the stacks show neither shape — every rank inside a collective, none compiling — you are looking at a real collective mismatch. Keep `NCCL_DEBUG=INFO` and the flight recorder on (`TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_TRACE_BUFFER_SIZE=2000`) and post the per-rank stacks in the issue.

## INT4 / Compressed-Tensors Quantization Checkpoint Issues

When using INT4-quantized models (e.g., `compressed-tensors` with `W4A16`), the checkpoint's `config.json` contains a `quantization_config.ignore` list that specifies which parameters should **not** be quantized. During online weight updates (Megatron → SGLang), slime also reads this ignore list to decide which parameters to INT4-quantize. An incorrect ignore list can cause silent errors:

1. **MoE router weights (`mlp.gate.weight`) become all zeros**

   The MoE router weight (`mlp.gate.weight`, shape `[num_experts, hidden_size]`) is a plain 2D weight tensor, but it is **not** a Linear layer weight. If it is not in the ignore list, the online quantizer will INT4-quantize it into `weight_packed`, `weight_scale`, `weight_zero_point`, etc. However, SGLang does not expect quantized names for the router, so these parameters are silently skipped during `load_weights`, resulting in all-zero gate weights.

   **Fix**: Ensure `config.json` contains `"re:.*mlp\\.gate\\..*"` in the ignore list.

2. **Other non-Linear 2D weights**

   Similar issues can occur with any 2D `.weight` tensor that is not a true Linear layer, such as `model.embed_tokens.weight`. Always verify the ignore list covers all non-Linear weights.

   **Recommended ignore patterns** (for GLM-style MoE models):
   ```json
   "ignore": [
     "lm_head",
     "model.embed_tokens.weight",
     "re:.*self_attn.*",
     "re:.*mlp\\.shared_experts.*",
     "re:.*mlp\\.gate_up_proj.*",
     "re:.*mlp\\.gate_proj.*",
     "re:.*mlp\\.up_proj.*",
     "re:.*mlp\\.down_proj.*",
     "re:.*eh_proj.*",
     "re:.*mlp\\.gate\\..*"
   ]
   ```

3. **Missing safetensors shards**

   Conversion tools may occasionally produce an incomplete checkpoint (e.g., a missing `model-00010-of-00093.safetensors`). After conversion, always verify:
   - The number of `.safetensors` files matches the expected count.
   - The `model.safetensors.index.json` contains entries for every layer.
   - Spot-check that critical layers (e.g., the first MoE layer) have the expected number of keys.

4. **How to diagnose**

   - Use `--check-weight-update-equal` to verify that weights after a Megatron → SGLang sync match the expected values. If a parameter shows all zeros on the SGLang side, it was likely incorrectly quantized or missing from the checkpoint.
   - Use `--debug-rollout-only` with a small number of GPUs to quickly test whether SGLang can generate coherent text from the quantized checkpoint alone.

## Debug sglang illegal memory access (IMA)

When running large scale RL, we will occationally meet the IMA in SGLang, there are some debug suggestions based on our experience:

1. Enable `CUDA_LAUNCH_BLOCKING=1`

2. Enable or disable speculative decoding and cuda graph to see if anything changed

   IMA always appears in the padding in cuda graph replay, or the difference between draft model and main model. We can minimize the scope by tuning them.

3. Turn off deepep

   If you are using deepep during training or inference, you can try turn it off.

4. Try CUDA Core Dump to find the error kernel

   We recommend reading the blog from the vLLM team: [CUDA Core Dump: An Effective Tool to Debug Memory Access Issues and Beyond](https://blog.vllm.ai/2025/08/11/cuda-debugging.html)

## Step-by-Step Debugging with Ray Distributed Debugger

Ray provides a [distributed debugger](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html) based on debugpy that lets you set breakpoints in the driver process and step through code interactively.

1. Install debugpy:

   ```bash
   pip install debugpy==1.8.0
   ```

2. Enable `RAY_DEBUG_POSTMORTEM` in your launch script:

   ```bash
   export RAY_DEBUG_POSTMORTEM=1

   RUNTIME_ENV_JSON="{
     \"env_vars\": {
       ...
       \"RAY_DEBUG_POSTMORTEM\": \"${RAY_DEBUG_POSTMORTEM:-0}\"
     }
   }"

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train.py [args...]
   ```

3. Add `ray.init()` before `breakpoint()` in `train.py`:

   ```python
   if __name__ == "__main__":
       ray.init()
       breakpoint()
       args = parse_args()
       train(args)
   ```

   `ray.init()` is required because the distributed debugger depends on `core_worker`, which is only available after Ray initialization. Without it, `breakpoint()` raises `AttributeError: 'Worker' object has no attribute 'core_worker'`.

4. Connect via VS Code:

   Install the [Ray Distributed Debugger](https://marketplace.visualstudio.com/items?itemName=ray-project.ray-distributed-debugger) extension in VS Code. Run your launch script to submit the job. Once the job hits `breakpoint()`, open the Ray Dashboard panel in VS Code and click the active breakpoint to attach the debugger. You can then step through code, inspect variables, and set additional breakpoints directly in the editor.

> **Note**: Remove `ray.init()` and `breakpoint()` after debugging. An explicit `ray.init()` without arguments may cause issues in multi-node training where Ray injects specific namespace and runtime environment configurations via `ray job submit`.
