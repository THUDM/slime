# Usage Guide


## Introduction to slime Parameters

When using slime, parameters are primarily passed for the following purposes:

1.  To allocate a portion of the GPUs in the cluster for training and another portion for inference.
2.  To load Megatron for the training portion.
3.  To load SGLang for the inference portion.
4.  To configure the hyperparameters required for RL training.

Following this order, we need to configure these parameters:

### Cluster Resource Allocation

There are four main parameters for cluster resource allocation:

  - `--actor-num-nodes`: The number of nodes required for RL actor training.
  - `--actor-num-gpus-per-node`: The number of GPUs per node for RL actor training.
  - `--rollout-num-gpus`: The total number of GPUs required for rollout (inference).
  - `--rollout-num-gpus-per-engine`: The number of GPUs per inference engine. This parameter is similar to SGLang's `tp_size`. When performing multi-node serving, this value should be the total number of GPUs. For example, if serving one model with 2 nodes and 16 GPUs, this value should be 16.
    The reason for not using a parameter like `--sglang-tp-size` is that we might consider supporting SGLang's `dp_size` parameter in the future, which means an engine could contain multiple SGLang servers (currently, only `--sglang-dp-size` under the `--sglang-enable-dp-attention` condition is supported).

With the default configuration, we use these parameters to allocate `actor_num_nodes * actor_num_gpus_per_node` GPUs for training and `rollout_num_gpus` GPUs for inference via Ray, thus achieving a separation of training and inference resources.

For co-located training and inference, you also need to configure:

  - `--colocate`: Enables co-located training and inference. When enabled, it ignores `--rollout-num-gpus` and makes the number of GPUs for training and inference equal.

### Loading Megatron

Unlike tools such as SGLang, vLLM, or Hugging Face Trainer, Megatron cannot directly read Hugging Face checkpoints. Instead, the user must configure the parameters for the model to be trained and load Megatron's own checkpoint format.

Generally, we need to perform three preparatory steps:

  - Configure model parameters.
  - Configure parallelism and other optimizations.
  - Configure the checkpoint to be loaded.

For details on some of Megatron's customizations and the principles behind how slime incorporates Megatron, please see the "How to Use Megatron" section.

#### Configuring Model Parameters

Taking qwen3 4B as an example, we need these parameters:

```bash
MODEL_ARGS=(
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --swiglu
   --vocab-size 151936
   --disable-bias-linear
   # attn head
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --kv-channels 128
   --qk-layernorm
   # norm
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   # rope
   --use-rotary-position-embeddings
   --rotary-base 1000000
)
```

We provide configurations for common models in [scripts/models](../../scripts/models), which you can reuse directly. If you are also using Megatron for pre-training/SFT, you can directly reuse the model configurations from your pre-training/SFT setup.

Note:

  - slime will load all parameters of Megatron found in the `PYTHONPATH`, so you can find parameters and their descriptions within the Megatron in your environment.
  - slime uses data packing (also known as varlen or thd) for training. There is no need to configure `--seq-length` or `--max-positional-embedding`, as these parameters do not affect the maximum context length of the trained model.

#### Setting Up Parallelism and Recomputation

Megatron is currently the most comprehensively optimized training framework. A major reason for using Megatron is to pursue its excellent performance. Here is a brief introduction to configuring Megatron's parallelism and recomputation.

  - Here we list Megatron's parallelism strategies. For a more detailed discussion on the trade-offs between these strategies, please refer to more specialized discussions:
      - `--tensor-model-parallel-size`: TP
      - `--sequence-parallel`: Megatron's SP is an optimization for TP. It is recommended to always enable SP when using TP.
      - `--pipeline-model-parallel-size`: PP
      - `--context-parallel-size`: Megatron's CP, also known as sequence parallelism, generally corresponds to ring attention.
      - `--expert-model-parallel-size`: EP for MoE, where each GPU has `num_experts / ep_size` experts.
      - `--expert-tensor-parallel-size`: Megatron supports using a different `tp_size` for the MoE experts than for other parts of the model, which we generally call ETP.
  - For recomputation, the following flags are commonly configured in Megatron:
      - `--recompute-granularity`: This can be set to `full` or `selective`. `full` means complete recomputation, while `selective` recomputes less. If not configured, no recomputation is done.
      - `--recompute-method`: `uniform` is generally sufficient.
      - `--recompute-num-layers`: The number of layers per group for recomputation. A value of 1 is usually fine.

#### Loading Megatron Checkpoints

Megatron supports several of its custom checkpoint formats. Here are two of the more common ones:

  - The once mainstream `torch` format (corresponding to `--ckpt-format torch`).
  - The currently recommended `torch_dist` format (corresponding to `--ckpt-format torch_dist`).

The `torch` format is Megatron's older storage format. Its structure consists of directories like `mp_rank_xxx`, where each directory corresponds to the checkpoint stored by each rank under a specific parallel partitioning. Because of this, when loading a `torch` format checkpoint, you must ensure that the checkpoint's parallelism strategy matches that of the training task.

We recommend using the `torch_dist` format because it supports automatic parallel sharding, meaning that training tasks with different parallelism settings can share the same checkpoint, which is much more convenient. `torch_dist` is also the default format in the open-source Megatron. A `torch_dist` format checkpoint typically contains a set of `.distcp` files. When using `torch_dist`, you can convert from Hugging Face to `torch_dist` and vice versa using the checkpoint conversion method described in the [README](../../README.md).

In terms of storage structure, a Megatron checkpoint typically looks like this, assuming the storage path is `/ckpt/`:

```bash
--/ckpt/
    |-- latest_checkpointed_iteration.txt
    |-- iter_0000100/
         |-- _0_0.distcp
         |-- _0_1.distcp
         |-- ...
    |-- iter_0000200/
    |-- iter_0000300/
    |-- ...
```

The `latest_checkpointed_iteration.txt` file records the latest training step. When loading a model, you should not directly pass `/ckpt/iter_xxxxxxx`, but rather pass `/ckpt/` and use `--ckpt-step` to select the corresponding training step (if `--ckpt-step` is not used, the step will be read from `latest_checkpointed_iteration.txt`).

When using slime, there are three parameters for loading and saving checkpoints:

  - `--ref-load`: The Megatron checkpoint for the reference model.
  - `--load`: The Megatron checkpoint for the actor. If `--load` is not set, or if the specified directory does not exist or does not contain `latest_checkpointed_iteration.txt`, the actor will be initialized from the `--ref-load` checkpoint.
  - `--save`: The path where the actor's checkpoints are saved.

Note:

  - Regardless of the checkpoint storage method (i.e., however `--ckpt-format` is set), Megatron can load both `torch` and `torch_dist` formats.

### Loading SGLang

Loading SGLang is very simple. You only need:

  - `--hf-checkpoint`: The Hugging Face checkpoint used to initialize SGLang.

Note:

  - Before the first training step, slime will synchronize the parameters from Megatron to SGLang. Therefore, the `--hf-checkpoint` does not need to contain the latest training parameters, and you do not need to change the HF checkpoint when resuming training.
  - By default, SGLang reads the maximum context length from the `config.json` in the Hugging Face checkpoint. You can use the `--sglang-context-length` parameter to override this value to support longer inference.
  - During co-located training and inference, although Megatron and SGLang will offload sequentially, they still need to leave some memory for each other. You need to adjust SGLang's total VRAM usage by reducing `--sglang-mem-fraction-static`.

For details on some of SGLang's customizations and the principles behind how slime incorporates SGLang, please see the "How to Use SGLang" section.

### Data Format

Currently, slime only supports loading files in `.jsonl` format, where each line of the file is a JSON object. An example of a single data entry (expanded) is as follows:

```json
{
  "prompt": [
    {
      "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\".",
      "role": "user",
      "step_loss_mask": 1,
    }
  ],
  "label": "34"
}
```

This corresponds to the following configuration:

```bash
  --input-key prompt
  --label-key label
  --apply-chat-template
```

Please note that the `step_loss_mask` (default=1) here is for SFT phase. If it is set to 0, the turn will not contibute to the final loss; if it is set to 1, slime will use the normal `loss_mask`.
Additionally, we provide a `metadata_key`, which defaults to `"metadata"`. When read, slime will load the metadata from the data, which can be helpful for custom data generation or creating custom reward models.

### Hyperparameters for RL Training

- `--advantage-estimator`: Specifies the RL algorithm for the training process. Currently supported algorithms include:
    - `grpo` ([https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300))
    - `gspo` ([https://arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071))
    - `reinforce_plus_plus` and `reinforce_plus_plus_baseline` ([https://arxiv.org/abs/2501.03262](https://arxiv.org/abs/2501.03262))
    - `ppo` ([https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347))
- `--calculate-per-token-loss`: By default, Slime calculates loss on a per-sample basis, i.e., `mean(sum(sample_i) / len(sample_i))`. Enable this flag to calculate loss on a per-token basis, i.e., `sum(sum(sample_i)) / sum(len(sample_i))`.
- `--use-tis`: Enable this setting to use TIS (Truncated Importance Sampling) (https://fengyao.notion.site/off-policy-rl).

## Custom Rollout Function

slime supports customizing data generation (rollout) to various degrees.

  - By default, it uses the `generate_rollout` function from [slime/rollout/sglang\_example.py](../../slime/rollout/sglang_rollout.py) for data generation. This file implements an asynchronous (asyncio) data generation flow based on SGLang and supports features like dynamic sampling and partial rollout.

  - You can completely replace the `generate_rollout` in sglang\_example.py by using the `--rollout-function-path` parameter. You just need to ensure that the function signature passed via `--rollout-function-path` is as follows:

    ```python
    def generate_rollout(args, rollout_id, data_buffer, evaluation=False) -> list[list[Sample]]:
        """
        Args:
            args: the whole args
            rollout_id: int, the id of the rollout, used for deterministic data generation
            data_buffer: the data buffer to store the generated samples
            evaluation: bool, whether the rollout is for evaluation or not
        
        Returns:
            list[list[Sample]]: a list of samples generated by the rollout
        """
            ...
            return samples
    ```

    Where:

      - `args`: The complete arguments used for the slime run.

      - `rollout_id`: The ID of the current data generation round, used to ensure data order when resuming training.

      - `data_buffer`: A globally unique data buffer in slime, which can be used to get initial prompts, data IDs, and store partially generated samples for later use.

      - `evaluation`: A boolean indicating if the rollout is for evaluation. You can configure a separate evaluation function using `--eval-function-path`.

      - The returned `Sample` type is defined in [slime/utils/types.py](../../slime/utils/types.py). When implementing, you need to ensure the following fields are correctly set:

          - `tokens`: The tokens for the prompt + response.
          - `response_length`: The total length of the response. For multi-turn tasks, this is the length of the tokens remaining after the first-turn prompt.
          - `reward`: The reward for this data sample.
          - `truncated`: Whether this data sample was truncated, similar to `finish_reason == length` in SGLang.

        And if there are scenarios like tool calls or multi-turn usage, ensure the `loss_mask` is correct:

          - `loss_mask` should be the same length as `response_length`, with `1` for tokens that should be included in the loss calculation and `0` for those that should be masked out.

  - In some cases, you may only need to replace the data generation logic. You can do this using `--custom-generate-function-path`. A simplified implementation of this function is as follows:

    ```python
    async def generate(args, sample: Sample, sampling_params) -> Sample:
        global TOKENIZER
        if TOKENIZER is None:
            TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        # send request to router
        output = await post(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
            {
                "text": sample.prompt,
                "sampling_params": sampling_params,
            }
        )

        prompt_tokens_ids = TOKENIZER(sample.prompt, add_special_tokens=False)["input_ids"]
        response_token_ids = TOKENIZER(output["text"], add_special_tokens=False)["input_ids"]

        # set sample
        sample.tokens = prompt_tokens_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        sample.truncated = output["meta_info"]["finish_reason"]["type"] == "length"
        sample.response = output["text"]
        sample.aborted = output["meta_info"]["finish_reason"]["type"] == "abort"

        return sample
    ```

    For a more complete version, please refer to [slime/rollout/sglang\_example.py](../../slime/rollout/sglang_rollout.py).

  - Sometimes, you may also need to support a custom reward model. This can be configured by setting `--custom-rm-path`.

## How to Use SGLang

slime implements a server-based engine using SGLang via the `HttpServerEngineAdapter` as an intermediary.

### Parameter Configuration

slime incorporates almost all SGLang parameters by using SGLang's `ServerArgs.add_cli_args`. When setting an SGLang parameter, you need to add the `--sglang-` prefix. For example:

  - In co-located training and inference, you often need to limit `--mem-fraction-static`. This parameter should be changed to `--sglang-mem-fraction-static`.
  - During training, if you want SGLang to infer beyond the maximum context length specified in the Hugging Face checkpoint's `config.json`, you need to use `--context-length`, which becomes `--sglang-context-length` in slime.
  - For multi-node large EP inference, you might need `--enable-ep-moe`, `--enable-dp-attention`, `--dp-size`, `--enable-deepep-moe`, etc. These can be passed as `--sglang-enable-ep-moe`, `--sglang-enable-dp-attention`, `--sglang-dp-size`, and `--sglang-enable-deepep-moe` respectively.

Some parameters related to slime's resource scheduling are configured by slime itself, for example:

  - `--tp-size` in slime is set using `--rollout-num-gpus-per-engine`.
  - `--model-path` in slime is set using `--hf-checkpoint`.

The way SGLang parameters are integrated into slime can be found in [slime/backends/sglang\_utils/arguments.py](../../slime/backends/sglang_utils/arguments.py).

### How to Use the Router

slime uses [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-router) to manage the SGLang servers during the training process. You can configure the address of the [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-router) using `--sglang-router-ip` and `--sglang-router-port`. If not configured, a router will be started by default within the cluster.

After starting, all SGLang servers will register with the router via the `/add_worker` endpoint. When actually generating data, you only need to send HTTP requests to the router, which will perform load balancing and forward the requests to the servers.

When you configure an external router using `--sglang-router-ip` and `--sglang-router-port`, slime will not start an internal router. Instead, it will register all its servers with this external router. You can then use this external router's address to implement more complex data generation workflows. Note that the router supports OpenAI-compatible APIs.

## How to Use Megatron

slime supports different and lightly modified versions of Megatron by reusing common functions from the `megatron.training` directory, such as `parse_args`, `save_checkpoint`, and `load_checkpoint`. Therefore, when using it, you must ensure that Megatron is accessible in the `PYTHONPATH`, for example, by adding `export PYTHONPATH=/root/Megatron-LM` at runtime.

### Parameter Configuration

slime directly imports all parameters of the Megatron in the current environment by using `from megatron.training.arguments import parse_args`. If the version of Megatron you are using has parameters defined outside of `parse_args`, you can configure them by passing them in, similar to how it's done in [train.py](../../train.py), for example:

```python
if __name__ == "__main__":
    try:
        from pretrain_gpt import extra_args_provider
    except:
        extra_args_provider = None
    args = parse_args(extra_args_provider)
    train(args)
```

### Custom Parameters

In some customized Megatron implementations, special operations need to be performed during initialization or before/after a training step. We have added the following plugins for this purpose:

  - `--custom-megatron-init-path`: Adds some initialization calls.
  - `--custom-megatron-before-log-prob-hook-path`: Is called before calculating the log probability.
  - `--custom-megatron-before-train-step-hook-path`: Is called before each training step. You could use this to mix in special training losses, for example.