# Curriculum Learning Example

This example showcases a basic curriculum learning setup for LLM RL training, offering a flexible framework for multi-task training across multiple data sources with configurable curriculum strategies. Users can define their custom RL training curriculum, ranging from simple multi-task weighted training to advanced dynamic weighting and online filtering.

## Key Features

- **[Dynamic Data Mixture Weights](docs/dynamic_weights.md)**: At each rollout step, prompt batches are sampled from multiple data sources using user-specified weights. The weights can vary across steps and be defined as constants, lambda expressions, or user-defined functions.
- **[Reward-Cap–Based Dynamic Sampling](docs/reward_cap_filter.md)**: To flexibly control the difficulty of prompts, reward caps are applied during dynamic sampling to filter out prompt groups with all-zero rewards OR average rewards exceeding a specified threshold. Caps are defined per data source and can also vary over time.

## Quick Start Example
We provide a quick-start example for merged training of reasoning (DAPO-Math-17k) and instruction following (VerInstruct). Following common practice, the curriculum prioritizes reasoning early in training and gradually shifts to instruction following by adjusting sampling weights in opposite directions across training steps (reasoning: 0.9→0.1, instruction following: 0.1→0.9).
### 1. Download Data

```bash
cd examples/curriculum_learning/data
huggingface-cli download zhangzx369/curriculum-learning-minimal-example \
  --include "data/*.jsonl" \
  --local-dir ../
cd ../../.. # back to the slime repo dir
```

This downloads 4 JSONL files:
- **Training**: dapo-math-17k.jsonl (17,398 samples), verinstruct.jsonl (19,756 samples)
- **Evaluation**: aime-2024.jsonl (32 samples), IFBench_eval.jsonl (300 samples)

See [data/download_data.md](data/download_data.md) for details.

### 2. Configure Checkpoint Paths

Edit [run_math_if_qwen3_8b.sh](run_math_if_qwen3_8b.sh) and update checkpoint paths:

```bash
CKPT_ARGS=(
   --hf-checkpoint /path/to/Qwen3-8B
   --ref-load /path/to/Qwen3-8B_torch_dist
   --load /path/to/Qwen3-8B_slime/
   --save /path/to/Qwen3-8B_slime/
)
```
Also, make sure you enter your own `WANDB_API_KEY` and enable wandb logging in slime training args.
### 3. Launch Training

```bash
bash examples/curriculum_learning/run_math_if_qwen3_8b.sh
```
## Monitoring

WandB metrics are tracked and organized independently for each data source:

**Per-Source Sampling**
- `rollout/{source}/weight` - Current weight curve
- `rollout/{source}/sample_count` - Sample counts per source

**Per-Source RL metrics**
- `rollout/{source}/raw_reward` - Average rewards
- `rollout/{source}/response_len` - Response lengths

**Reward-Cap-Based Dynamic Filtering**
- `rollout/{source}/filter_ratio_all_zero` - Zero-reward drop rate
- `rollout/{source}/filter_ratio_reward_cap` - Reward cap drop rate
- `rollout/{source}/reward_cap` - Current cap value (for dynamic caps)
## Configuration

The training config [configs/math_if_train_config.yaml](configs/math_if_train_config.yaml) demonstrates the following curriculum strategy:

```yaml
prompt_data:
  - examples/curriculum_learning/data/dapo-math-17k.jsonl
  - examples/curriculum_learning/data/verinstruct.jsonl

prompt_data_source_names:
  - dapo_math_17k
  - verinstruct

num_rollout: 200
# Dynamic weights: first prioritize on reasoning, and then gradually shifting to instruction following
data_source_weights:
  - "lambda step: 0.9 - 0.7 * min(step / 200, 1.0)"  # 0.9 → 0.2
  - "lambda step: 0.1 + 0.7 * min(step / 200, 1.0)"  # 0.1 → 0.8

# Dynamic caps: gradually increasing the filtering strictness based on average rewards
data_source_reward_caps:
  - "lambda step: 0.9 - 0.05 * min(step / 200, 1.0)"  # dapo_math_17k: 0.9 -> 0.85
  - "lambda step: 0.9 - 0.1 * min(step / 200, 1.0)"  # verinstruct: 0.9 -> 0.8

# Enable dynamic filtering
over_sampling_batch_size: 256
dynamic_sampling_filter_path: examples.curriculum_learning.reward_cap_filter.check_reward_cap_per_source
```

### Weight Configuration Options

Each data source weight can be specified as:

**1. Constant**
```yaml
data_source_weights:
  - 0.5
  - 0.5
```

**2. Lambda function**
```yaml
data_source_weights:
  - "lambda step: 0.9 - 0.8 * min(step / 200, 1.0)"
```

**3. Function from module**
```yaml
data_source_weights:
  - "examples.curriculum_learning.weight_scheduler.decreasing_weight"
  - "examples.curriculum_learning.weight_scheduler.increasing_weight"
```

See [docs/dynamic_weights.md](docs/dynamic_weights.md) for details.

### Reward Cap Configuration

Configure per-source reward caps similarly:

```yaml
data_source_reward_caps:
  - 0.3  # Static cap
  - "lambda step: 0.1 + 0.5 * min(step / 200, 1.0)"  # Dynamic cap
  - "examples.curriculum_learning.reward_cap_scheduler.increasing_cap"  # Function
```

See [docs/reward_cap_filter.md](docs/reward_cap_filter.md) for details.

## Example Curriculum Learning Strategies

### Easy-to-Hard Dataset Shift
Gradually shift emphasis from easy to hard data, reflecting the intuition that the model should learn simpler patterns before tackling more difficult ones.
```yaml
prompt_data:
  - {easy_dataset_path}
  - {hard_dataset_path}
data_source_weights:
  - "examples.curriculum_learning.weight_scheduler.decreasing_weight"  # 0.9 → 0.1
  - "examples.curriculum_learning.weight_scheduler.increasing_weight"  # 0.1 → 0.9
```

### Strengthening Quality Filter
Begin with relaxed prompt filtering and gradually apply stricter criteria to admit more challenging data:
```yaml
data_source_reward_caps:
  - "examples.curriculum_learning.reward_cap_scheduler.decreasing_cap"  # 0.9 → 0.1
```

### Your Custom Curriculum
You can define your own curriculum with any number of data sources and any types of weighting and reward-filtering schedules.
```yaml
prompt_data:
  - "course 1"  
  - "course 2"  
  - "course 3"  
  - "..."  

data_source_weights: # Design any curriculum learning weight schedules you like
  - "lambda step: 0.9 - 0.7 * min(step / 200, 1.0)"  # 0.9 → 0.2
  - "lambda step: 0.1 + 0.7 * min(step / 200, 1.0)"  # 0.1 → 0.8
  - 0.05 # constant weight
  - ...

data_source_reward_caps:
  - "lambda step: 0.1 + 0.4 * min(step / 200, 1.0)"  # Easy: 0.1 → 0.5
  - 0.8  # constant
  - null # do not apply dynamic sampling
  - ...
```

## Documentation

- [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) - Code structure and component documentation
- [docs/dynamic_weights.md](docs/dynamic_weights.md) - Dynamic weight scheduling
- [docs/reward_cap_filter.md](docs/reward_cap_filter.md) - Reward filtering system

## Tips & Best Practices

1. **Balance Over-Sampling**: Set `over_sampling_batch_size` 2-3x larger than `rollout_batch_size`
2. **Start Conservative**: Begin with higher reward caps (0.6-0.8) to ensure sufficient data
3. **Debug Incrementally**: Start with constant weights, add curriculum components one at a time

## References

- DAPO Paper: https://arxiv.org/abs/2503.14476
- VerInstruct Paper: https://arxiv.org/pdf/2506.09942
