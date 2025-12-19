# FAQ

1.  **Why do I see garbled text during training?**

    This situation generally occurs because Megatron is not loaded correctly. Please check if there is a corresponding checkpoint in the directory specified by `--load` or `--ref-load`. Note that Megatron can only load a directory that contains a `latest_checkpointed_iteration.txt` file.

    If you need to specify a particular iteration, you can refer to the current Megatron usage instructions. Generally, you can specify the step number using `--ckpt-step`.

2.  **Why is my task stuck on the Ray submission page?**

    Please check whether your task is set up for co-located training and inference or decoupled training and inference.

    If it's **co-located** (training and inference share the same GPUs), please check:

      * Whether the `--colocate` parameter is set to enable co-located mode.
      * Whether the total number of GPUs for the current task is greater than or equal to `actor_num_nodes * actor_num_gpus_per_node`.

    If it's **decoupled**, please check:

      * Whether the total number of GPUs for the current task is greater than or equal to `actor_num_nodes * actor_num_gpus_per_node + rollout_num_gpus`.

3.  **Why did I encounter an Out-of-Memory (OOM) error during training? What is `max_tokens_per_gpu` for?**

    OOM errors often happen because `max_tokens_per_gpu` is set too high. This parameter defines the maximum number of tokens that can be processed on each GPU during training. If you are concerned about OOM, you can initially set this value to `rollout_max_response_len / cp_size` and then increase it later to improve training efficiency. Note that `--max-tokens-per-gpu` is only active when `--use-dynamic-batch-size` is enabled.

    If you still experience OOM with a small `max_tokens_per_gpu`, check if the data generated in a single pass is too long. You may need to enable context parallelism (CP) with `--context-parallel-size`. If you are using custom data generation, check if the total length of multi-turn generations is much longer than expected.

4.  **During multi-node training, what should I do if the `transformers` library reports it cannot find a model?**

    This usually happens when multiple processes try to read local files simultaneously using methods like `AutoConfig.from_pretrained` or `AutoModelForCausalLM.from_pretrained`, causing file system write conflicts. You can mitigate this issue by setting the `--model-name` argument.

5.  **How do I resume training?**

    Simply set the `--load` directory to your `--save` directory.

6.  **How is the batch size calculated?**

    A single rollout uses `rollout_batch_size` prompts. For each prompt, `n_samples_per_prompt` samples are generated. Therefore, one rollout contains a total of `rollout_batch_size * n_samples_per_prompt` data entries.

    You can use `--num-steps-per-rollout` to determine how many steps to run per rollout. This is equivalent to setting the `global_batch_size` to `rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`.

7.  **Does slime perform data packing / variable-length (varlen) processing?**

    Yes. Data packing refers to the process of concatenating samples of varying lengths during training to improve GPU utilization. slime performs this operation by default.

8.  **What should I do if the sglang component shows a `Max retries exceeded with url: /get_model_info (Caused by NewConnectionError)` error?**

    This issue primarily stems from port conflicts caused by multiple sglang servers running on a single machine. We are currently working with the sglang team to resolve this. A temporary workaround is to minimize the number of sglang servers on a single machine, for example, by setting `tp=8`.

9.  **My gradient norm is very high and the training crashes. What should I do?**

    First, ensure that your data and model are compatible. For example, if your data already uses a chat template, check if this template matches the one used by the original model. If the data is correct, please refer to our [Debug Guide](../developer_guide/debug.md) for a more in-depth analysis.

10. **My sglang generation takes an extremely long time, GPU power is maxed out, and there's no output for a long while. Why?**

    Please verify that the model corresponding to `--hf-checkpoint` has its stop tokens configured correctly. If not, you can set them using the `--rollout-stop` or `--rollout-stop-token-ids` arguments.

11. **Sglang shows an `an illegal memory access was encountered` error.**

    According to the sglang documentation ([https://docs.sglang.ai/references/troubleshooting.html](https://docs.sglang.ai/references/troubleshooting.html)), this could be an OOM error. Consider reducing the value of `--sglang-mem-fraction-static`.

12. **A `JSONDecodeError` occurs related to torch compile/inductor.**

    This is generally an issue with the torch compiler's cache read/write operations. You can try adding `"TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"` to the `env_vars` in your Ray configuration.

13. **Gradient becomes NaN or Inf during training.**

    You can try setting the `--no-check-for-nan-in-loss-and-grad` flag to skip the corresponding training steps.