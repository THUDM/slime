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