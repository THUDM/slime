# Quick Start


This document will guide you through setting up the environment and getting started with slime within one hour, covering environment configuration, data preparation, training startup, and key code analysis and modifications.

## Basic Environment Setup

Since slime may contain temporary patches for sglang/megatron, to avoid potential environment configuration issues, we strongly recommend **users to use our latest Docker image**, which comes pre-configured with all dependencies.

### Hardware Support

**slime** supports multiple NVIDIA GPU hardware platforms:

- **B200 Series**: Fully supported with identical setup steps as H-series GPUs
- **H-Series (H100/H200)**: Official support with comprehensive CI testing and stable performance

**Important Notes**:
- Latest Docker images are compatible with both B-series and H-series GPUs without additional configuration
- Megatron backend on H-series GPUs has CI protection, thoroughly validated, recommended for production environments
- B-series basic functionality is stable and suitable for development/testing, but currently lacks CI protection
- Both hardware platforms use identical installation and startup procedures

- For scenarios where Docker is not convenient, please refer to [build_conda.sh](https://github.com/THUDM/slime/blob/main/build_conda.sh);
- For AMD support, please refer to [AMD Usage Tutorial](../platform_support/amd_tutorial.md).

### Pull and Start Docker Container

Please execute the following commands to pull the latest image and start an interactive container:

```shell
# Pull the latest image
docker pull slimerl/slime:latest

# Start the container
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```

### Install slime

After entering the Docker container, please follow these steps to clone the slime repository and install it:

```bash
# Path can be adjusted according to actual situation
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
```

## Model and Dataset Download

You can download required models and datasets from platforms like Hugging Face, ModelScope, etc. Here are the commands to download example resources using `huggingface_hub`:

```bash
pip install -U huggingface_hub

# Download model weights (GLM-Z1-9B)
hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414

# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

## Model Weight Conversion

### Convert from Hugging Face Format to Megatron Format

When using Megatron as the training backend, you need to first convert Hugging Face format model weights to Megatron `torch_dist` format.

First, load the configuration file of the target model. The `slime/scripts/models` directory contains configuration files for supported models. You need to `source` the corresponding model script to load the configuration parameters into the current environment. Here we use GLM4-9B model as an example, and it's similar for Qwen3-4B, Qwen3-30B-A3B, etc.

```bash
cd /root/slime
source scripts/models/glm4-9B.sh
```

Next, run the conversion script. Please note the following parameters:
- `--hf-checkpoint`: Specify the path of the downloaded Hugging Face model weights.
- `--save`: Specify the save path for the converted `torch_dist` format weights.

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist
```

For larger models, you can use `torchrun` to start the covnersion script to convert with multi-gpus or even multi-nodes.
Note: When converting the kimi-k2 model weights, you need to open config.json in the model path and change "model_type": "kimi_k2" to "model_type": "deepseek_v3".

### Convert from Megatron Format to Hugging Face Format

You can use the following script to convert the saved Megatron chekcpoints back to Hugging Face format:

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/torch_dist_ckpt/iter_xxx/ \
  --output-dir /root/GLM-Z1-9B-0414-iter_xxx \
  --origin-hf-dir /root/GLM-Z1-9B-0414
```

Note that as Megatron will do padding to embedding for better performance, it may happen that the converted embedding is not correct. In that case, please manually set `--vocab-size` during convertion.


## Training Script and Parameter Overview

After completing the above preparation work, you can run the training script.

```bash
cd /root/slime
bash scripts/run-glm4-9B.sh
```

We still use the run-glm4-9B.sh script as an example to briefly analyze the main parameters.

### MODEL_ARGS: Model Configuration Parameters

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"
```

This part loads model configuration from the `scripts/models/glm4-9B.sh` file through the `source` command. These configurations are all hyperparameters required by Megatron. Since Megatron cannot directly read model configuration from checkpoints, it needs to be manually specified. We provide configuration examples for some commonly used models in the `scripts/models/` directory.

> ⚠️ **Note**:
> Please make sure to check whether the parameters in the model configuration file (such as `--rotary-base`) completely match the model you are currently using. Different versions of the same model structure may use different configuration values. If you need to modify, you can directly override after `source`, for example:
> ```bash
> source "${SCRIPT_DIR}/models/glm4-9B.sh"
> MODEL_ARGS+=(--rotary-base 10000)
> ```

### CKPT_ARGS: Checkpoint and Path Parameters

```bash
CKPT_ARGS=(
   # To load tokenizer and other information, won't actually use model weight parameters from hf path
   --hf-checkpoint /root/GLM-Z1-9B-0414
   # Reference Model's Megatron format checkpoint
   --ref-load /root/GLM-Z1-9B-0414_torch_dist
   # Actor model loading path. Should typically match --save for checkpoint resumption
   # If empty or doesn't contain a valid checkpoint, loads from --ref-load instead
   --load /root/GLM-Z1-9B-0414_slime/
   # Model save path during training
   --save /root/GLM-Z1-9B-0414_slime/
   # Model save interval (steps)
   --save-interval 20
)
```

### ROLLOUT_ARGS: Data Generation (Rollout) Parameters

The entire training process can be viewed as a closed loop of **"Data Sampling → Weight Update"**.

**Phase One: Data Sampling (Rollout)**
- `--rollout-batch-size`: Defines the **number of Prompts** for each round of sampling
- `--n-samples-per-prompt`: Defines the **number of responses** generated for each Prompt (used for GRPO-like algorithms)

> The product of the two determines the **total number of samples generated in a single round of sampling**.

**Phase Two: Model Training (Training)**
- `--global-batch-size`: Defines the **sample size required to execute one parameter update (optimizer.step)**
- `--num-steps-per-rollout`: Defines **how many parameter updates to execute** using the current sampled data (we default to 1, using on-policy training)

> The product of the two determines the **total number of samples consumed in a single round of training**.

> ⚠️ The **parameter update** here refers to the optimizer.step() in the training phase, which is different from the weight synchronization (Weight Sync) initiated by the training engine to the inference engine.

In this process, the "output" and "consumption" of each round must be equal, following this constraint:
**`(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)`**

- In slime, if `--num-steps-per-rollout` is set, `--global-batch-size` will be automatically set if not set, and if set, it will be validated using the above formula.

**Training Process Count Control**
- `--num-rollout`: Controls the **total number of execution rounds** of the entire **"sampling→training"** loop.

```bash
ROLLOUT_ARGS=(
   # Prompt dataset, JSONL format
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   # If the `input_key` of Prompt is in OpenAI message format, apply Chat Template
   --apply-chat-template
   # Whether to shuffle data in Rollout phase
   --rollout-shuffle

   # Reward Model type. slime has built-in multiple types, also supports custom through --custom-rm-path
   --rm-type deepscaler

   # These five parameters control the relationship between rollout and train
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --num-steps-per-rollout 1
   --global-batch-size 128

   # Rollout sampling parameters
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   # Load balancing for data collected in rollout phase. It ensures that the computational workload allocated to each training process (DP rank) is roughly equal, which may be beneficial for training speed
   --balance-data
)
```

### EVAL_ARGS: Evaluation Parameters

The evaluation process inherits most of the Rollout parameters, but you can override them with the following parameters to implement evaluation strategies different from training.

```bash
EVAL_ARGS=(
   # Evaluation interval (number of Rollouts)
   --eval-interval 5
   # Prompt dataset for evaluation
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   # Number of samples per evaluation Prompt
   --n-samples-per-eval-prompt 16
   # Maximum response length during evaluation
   --eval-max-response-len 16384
   # Sampling parameters during evaluation
   --eval-top-p 0.7
)
```

### PERF_ARGS: Performance and Parallelism Parameters

This part mainly contains Megatron's parallel configuration. `--use-dynamic-batch-size` and `--max-tokens-per-gpu` are slime-specific optimizations.

- `--max-tokens-per-gpu`: Maximum number of tokens processed per GPU. After enabling dynamic batching (`use_dynamic_batch_size`), the system will intelligently pack samples of varying lengths so that the total token count of each micro-batch approaches this limit, thereby improving training efficiency. If a single sample length exceeds this value, it will form an independent batch. In context parallel (CP) mode, `N` CP cards share the total length of `N * max_tokens_per_gpu`.
- `--use-dynamic-batch-size`: Enable dynamic batching. At this time, `--micro-batch-size` will be ignored.

> 💡 **Tip**:
> slime always trains models through data packing methods and strictly ensures that per sample loss or per token loss is correct. Therefore, enabling dynamic batch size will not affect loss calculation, and it is strongly recommended to enable it.

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1 # This item is ignored when dynamic batching is enabled
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)
```

### GRPO_ARGS: GRPO Algorithm Parameters

- `--use-kl-loss`: Enabling this option will load a reference model and calculate the KL divergence between the current model and the reference model as a monitoring metric. Whether KL divergence is included in the final training loss depends on the `--kl-loss-coef` parameter. If this parameter is set to 0, KL divergence will only be displayed as an observation metric and will not participate in loss calculation.

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

- `--advantage-estimator`: In addition to [GRPO](https://arxiv.org/abs/2402.03300), slime also supports several other training algorithms, such as [GSPO](https://arxiv.org/abs/2507.18071), [Reinforce++](https://arxiv.org/abs/2501.03262) and [Reinforce++ Baseline](https://arxiv.org/abs/2501.03262), and [PPO](https://arxiv.org/abs/1707.06347).
- `--calculate-per-token-loss`: By default, slime calculates the loss on a per-sample basis, i.e., `mean(sum(sample_i) / len(sample_i))`. To calculate the loss on a per-token basis, i.e., `sum(sum(sample_i)) / sum(len(sample_i))`, you can enable this flag.
- `--use-tis`: Enable this setting to use TIS (Truncated Importance Sampling), which is introduced by this [blog](https://fengyao.notion.site/off-policy-rl).

### OPTIMIZER_ARGS: Optimizer Parameters

```bash
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
```

### SGLANG_ARGS: SGLang Service Parameters

This part of parameters is used to configure SGLang inference service.
- `--rollout-num-gpus-per-engine`: Basically equivalent to SGLang's `tp_size`.
- Other SGLang parameters can be passed to slime by adding the `--sglang-` prefix, and slime will automatically forward them to SGLang. For example, to set SGLang's `--log-level INFO` parameter, just use `--sglang-log-level INFO`.

> ⚠️ **Note**:
> slime uses `sgl-router` to schedule multiple SGLang Servers. Without enabling DP Attention, `dp_size` will be calculated through `rollout-num-gpus/rollout-num-gpus-per-engine`.

```bash
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)
```

## Feature Introduction

### Colocated Actor and Rollout

Under the default configuration, training (Actor) and inference (Rollout) resources are specified separately. Ray allocates `actor_num_nodes * actor_num_gpus_per_node` GPUs to the training part and `rollout_num_gpus` GPUs to inference, that is, training and inference are separated.

**Standard (Disaggregated) Configuration**:
```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ...
```
In the above configuration, Actor uses 4 cards, and Rollout also uses 4 cards, running in parallel.

**Training-Inference Integration (Colocated) Configuration**:
To deploy training and inference on the same group of GPUs, please add the `--colocate` parameter. After enabling, `--rollout-num-gpus` will be ignored to make the number of cards for training and inference equal.

```bash
ray job submit ... \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ...
```
At this time, training and inference will share all 8 GPUs.

> ⚠️ **Note**:
> In training-inference integration mode, Megatron will occupy a certain amount of GPU memory before it can be offloaded after initialization. You need to adjust the `--sglang-mem-fraction-static` parameter to reduce SGLang's GPU memory usage ratio to avoid insufficient GPU memory. We usually recommend 0.8.

### Dynamic Sampling

slime supports more complex sampling strategies, such as dynamic sampling used in [DAPO](https://dapo-sia.github.io/). To enable this feature, you need to configure the following parameters:

```bash
   --over-sampling-batch-size 64 \
   --dynamic-sampling-filter-path \
     slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

Here `over_sampling_batch_size` needs to be greater than `rollout_batch_size`, for example, configured as:

```bash
   --rollout-batch-size 32 \
   --n-samples-per-prompt 8 \
   --over-sampling-batch-size 64 \
```

Then each sampling will directly sample 64 prompts, and each prompt will be sampled 8 times. Because slime performs asynchronous sampling internally, we will successively obtain 8 responses for each prompt. When receiving responses, the function corresponding to `dynamic_sampling_filter_path` will be used for filtering. If it passes, these 8 pieces of data will be kept; otherwise, they will be discarded.

The filtering function `check_reward_nonzero_std` in the example will check whether the standard deviation of rewards for a group of samples is greater than zero, ensuring that the reward scores of each group of samples left have differences, thereby avoiding overly homogeneous data and improving data diversity.

```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.reward for sample in samples]
    return torch.tensor(rewards, dtype=torch.float).std() > 0.0
```

If the filtering function is very strict, causing a large number of prompt groups to be discarded, the system will monitor the number of pending tasks in `remaining_batch_size`. Once the number of pending tasks drops below the target number (32) due to too many being discarded, the system will automatically trigger a new round of oversampling, requesting `over_sampling_batch_size` (64) new prompts again to repeat the above process.

### Partial Rollout

During dynamic sampling, a large number of requests may be aborted early, causing waste of computational resources. By enabling the `--partial-rollout` parameter, these half-generated samples can be cached and continued to be generated in the next Rollout phase, thereby improving performance.

You can also customize the strategy for extracting data from the cache through `--buffer-filter-path`. The default strategy is `pop_first`, which extracts the required number of samples in first-in-first-out order.

```python
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

That is, take out the first `num_samples` prompts corresponding to `num_samples * n_samples_per_prompt` pieces of data each time.

> 💡 **Tip**:
> The `sample.metadata` of each partial rollout sample stores the rollout id of the first generation, which can be used for data filtering.

### bf16 Training fp8 Inference

slime directly supports bf16 training and fp8 inference. For Qwen3-4B model, you only need to download the following model:

```bash
hf download Qwen/Qwen3-4B-FP8 --local-dir /root/Qwen3-4B-FP8
```

And replace `--hf-checkpoint` with:

```bash
   # Used to load tokenizer and other information, actually won't use model weight parameters from hf path
   --hf-checkpoint /root/Qwen3-4B-FP8

   # The megatron checkpoint still needs to be the dist weights converted from bf16 huggingface at the beginning, not modified because of FP8 rollout.
   --ref-load /root/Qwen3-4B_torch_dist
```

This will trigger fp8 inference. Currently, we will directly cast bf16 weights to fp8, and we will gradually add quantization schemes with less impact on accuracy in the future.

⚠️ The training megatron checkpoint still needs to be the one converted from bf16 huggingface at the beginning.

## Multiturn Adaptation

The slime framework is highly extensible and supports complex Agent scenarios (such as multi-turn interaction and tool calling). Its core mechanism is to rewrite the default data generation (Rollout) and reward calculation (Reward) logic through custom functions.

This section uses an implementation based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) as an example to illustrate how to adapt slime to support multi-turn interaction.

### Adaptation Strategy Summary

Adapting slime to support multi-turn interaction mainly includes three steps:

1. **Data Preparation**: Adapt the multi-turn interaction dataset to slime's `Sample` objects. Map conversation history, real labels, etc. to `prompt` and `label` fields, and store additional information such as tool definitions and intermediate states in the `metadata` field for subsequent function calls.

2. **Implement Custom Generation Function**: Write functions to simulate the interaction loop of "model generates action → executes tool → concatenates observation results", and correctly handle Loss Masking.

3. **Implement Custom Reward Function**: Write functions to evaluate complete interaction trajectories and return final reward scores.

### Data Preparation and Mapping

To pass complex contextual information to custom functions, you need to aggregate all relevant additional fields during the **data preprocessing stage**.

**Core Idea**: Merge all additional information in the dataset except `prompt` and `label` (such as `session_id`, `user_profile`, `tool_code`, etc.) to construct a **single, structured field** (for example, a column named `metadata` with JSON string content).

### Step One: Construct `metadata` Field in Dataset

Before training starts, you need to process the original dataset. For example, your original data might be as follows:

| question | final_answer | session_id | tool_code |
| :--- | :--- | :--- | :--- |
| "..." | "..." | "sess_123" | "code_A" |

You need to convert it to:

| question | final_answer | metadata |
| :--- | :--- | :--- |
| "..." | "..." | `{"session_id": "sess_123", "tool_code": "code_A"}` |

### Step Two: Specify Mapping in Training Script

After completing data preparation, in the training script, map this preprocessed `metadata` column to slime's `Sample.metadata` field through `ROLLOUT_ARGS`.

```bash
ROLLOUT_ARGS=(
   # 1. Specify the preprocessed dataset file
   --prompt-data /root/nq_search/train_processed.json

   # 2. Map "question" column to input prompt
   --prompt-key question

   # 3. Map "final_answer" column to evaluation label
   --label-key final_answer

   # 4. Load the pre-constructed "metadata" column into Sample.metadata
   #    slime will automatically parse it as a Python dictionary
   --metadata-key metadata
)
```

Through this approach, you can easily access all pre-prepared structured information through methods like `sample.metadata['session_id']` in custom `generate` or `reward` functions.

### Writing Custom Generation Function

First, specify a custom asynchronous Python function through the `--custom-generate-function-path` parameter.

**Function Signature**: `async def generate(args, sample: Sample, sampling_params) -> Sample:`

**Core Implementation Points**:

1. **Build Interaction Loop**: Create a loop to control maximum interaction rounds (such as `for _ in range(max_turns):`).
2. **Call Model to Generate Action**: In each round of the loop, call SGLang service to let the model generate the next action (such as `<search>query</search>`) based on the current conversation history.
3. **Parse and Execute Action**: Parse model output, identify actions and parameters, and call external tools or APIs (such as Google search).
4. **Build Observation Results**: Format the results returned by tools and append them to the conversation history as input for the next round.
5. **Handle Loss Masking**: This is the key to Agent training.
   - Note: `loss_mask` should be the same length as `response`, where tokens that need to calculate loss are 1, and masked ones are 0
   - **Model-generated** tokens (such as thinking, action instructions) → set `loss_mask` to `1`, participate in loss calculation.
   - **Tool or environment returned** tokens (such as API results) → set `loss_mask` to `0`, do not participate in loss calculation.
6. **Termination Conditions**: End the loop when the model generates termination tags (such as `<answer>...`) or reaches maximum rounds.
7. **Encapsulate Return**: Fill the complete interaction history, token IDs, and `loss_masks` into the `Sample` object and return.

**Code Example (Pseudocode)**:
```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    # ... initialization ...
    prompt, full_response, loss_masks = sample.prompt, "", []

    for _ in range(max_turns):
        # 1. Model generates action
        model_output = await call_sglang(prompt + full_response, ...)
        # ... tokenization and appending ...
        loss_masks += [1] * len(model_tokens) # loss_mask = 1
        full_response += model_output

        # 2. Parse and execute action
        action, content = parse_action(model_output)
        if action == "search":
            # 3 & 4. Get and append observation results
            tool_output = await google_search(content)
            # ... tokenization and appending ...
            loss_masks += [0] * len(tool_tokens) # loss_mask = 0
            full_response += tool_output

        elif action == "answer":
            break # end loop

    # 7. Fill and return Sample object
    sample.response = full_response
    sample.tokens = ...
    sample.loss_mask = loss_masks
    return sample
```

### Writing Custom Reward Function

Similarly, specify a custom reward function through `--custom-rm-path`.

**Function Signature**: `async def reward_func(args, sample: Sample, **kwargs) -> float:`

This function receives a complete `Sample` object and calculates scores based on the final interaction results. You can implement custom scoring logic here or call external Reward Model services.

### Configure in Training Script

Finally, in the training script, enable the above custom functions through the following parameters:

```bash
CUSTOM_ARGS=(
   # Specify the path of custom generation function (format: path.to.your.file:function_name)
   --custom-generate-function-path your_module.multiturn_logic.generate

   # Specify the path of custom reward function
   --custom-rm-path your_module.multiturn_logic.reward_func
)
```

## Multi-Node Training for Large-Scale MOE Models

To start a multi-node task, you need to first start a Ray cluster. On node 0, run:

```bash
# Node0 (HEAD)
ray start --head --node-ip-address ${MASTER_ADDR} \
  --num-gpus 8 --disable-usage-stats

# Other Nodes
ray start --address=${MASTER_ADDR}:6379 --num-gpus 8
```

After the Ray cluster has started, you can submit a job from node 0, for example:

```bash
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        ... # e.g., no_proxy, API variables, etc.
     }
   }' \
   -- python3 train.py \
   --... # Other Megatron/SGLang/slime arguments
```

slime has been deeply optimized for distributed training of large-scale Mixture of Experts (MoE) models. We provide some end-to-end training cases for reference:

- [Example: 64xH100 Training GLM-4.5](models/glm4.5-355B-A32B.md)
- [Example: 128xH100 Training DeepSeek-R1](models/deepseek-r1.md)
