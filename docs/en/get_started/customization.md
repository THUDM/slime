# Customization Guide

slime provides extensive customization capabilities through function path arguments. These allow you to inject custom logic at various stages of the training and rollout pipeline without modifying the core codebase.

## Overview of Customization Interfaces

Below is a summary of all available customization interfaces and their purposes.

| Interface Argument | Purpose |
| :--- | :--- |
| [`--rollout-function-path`](#1-rollout-function---rollout-function-path) | Override the entire rollout generation logic. |
| [`--custom-generate-function-path`](#2-custom-generate-function---custom-generate-function-path) | Override only the generation step (e.g., for RAG or tool use). |
| [`--custom-rm-path`](#3-reward-model---custom-rm-path) | Implement custom reward computation logic. |
| [`--dynamic-sampling-filter-path`](#4-dynamic-sampling-filter---dynamic-sampling-filter-path) | Filter samples during dynamic sampling (e.g., DAPO). |
| [`--buffer-filter-path`](#5-buffer-filter---buffer-filter-path) | Filter samples in the rollout buffer before training. |
| [`--rollout-sample-filter-path`](#6-rollout-sample-filter---rollout-sample-filter-path) | Determine if individual samples participate in loss calculation. |
| [`--rollout-all-samples-process-path`](#7-rollout-all-samples-process---rollout-all-samples-process-path) | Process all samples (including filtered ones) after rollout. |
| [`--rollout-data-postprocess-path`](#8-rollout-data-postprocess---rollout-data-postprocess-path) | Post-process rollout data after log probs are computed. |
| [`--custom-loss-function-path`](#9-custom-loss-function---custom-loss-function-path) | Implement custom training loss computation. |
| [`--custom-tis-function-path`](#10-custom-tisrs-function---custom-tis-function-path) | Implement custom importance sampling for off-policy correction. |
| [`--custom-pg-loss-reducer-function-path`](#11-custom-pg-loss-reducer---custom-pg-loss-reducer-function-path) | Customize pg_loss reduction (e.g., for Dr.GRPO). |
| [`--custom-reward-post-process-path`](#12-reward-post-processing---custom-reward-post-process-path) | Custom post-processing of rewards before advantage computation. |
| [`--custom-convert-samples-to-train-data-path`](#13-samples-to-train-data-conversion---custom-convert-samples-to-train-data-path) | Override the conversion of samples to training data format. |
| [`--custom-rollout-log-function-path`](#14-logging-functions) | Custom logging for training rollouts. |
| [`--custom-eval-rollout-log-function-path`](#14-logging-functions) | Custom logging for evaluation rollouts. |
| [`--data-source-path`](#15-data-source---data-source-path) | Override the data source for rollout prompts. |
| [`--eval-function-path`](#16-evaluation-function---eval-function-path) | Override the rollout function specifically for evaluation. |
| [`--custom-megatron-init-path`](#17-megatron-hooks) | Custom initialization after Megatron setup. |
| [`--custom-megatron-before-log-prob-hook-path`](#17-megatron-hooks) | Custom logic before log probability computation. |
| [`--custom-megatron-before-train-step-hook-path`](#17-megatron-hooks) | Custom logic before each training step. |
| [`--slime-router-middleware-paths`](#18-slime-router-middleware---slime-router-middleware-paths) | Add custom middleware to slime router. |

## Detailed Interface Reference

### 1. Rollout Function (`--rollout-function-path`)

**Default**: `slime.rollout.sglang_rollout.generate_rollout`

**Purpose**: Override the entire rollout generation logic.

**Signature**:
```python
async def generate_rollout(args, rollout_id, *, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput
```

**Use Cases**:
- Implementing complex multi-turn conversations
- Adding custom sampling strategies
- Integrating external tools or APIs during generation

**Example**: See [examples/multi_agent/rollout_with_multi_agents.py](../../../examples/multi_agent/rollout_with_multi_agents.py)

---

### 2. Custom Generate Function (`--custom-generate-function-path`)

**Default**: `None` (uses built-in generate function)

**Purpose**: Override only the generation step within the default rollout function.

**Signature**:
```python
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample
```

**Use Cases**:
- Implementing tool-calling or function-calling capabilities
- Adding retrieval-augmented generation (RAG)
- Multi-turn conversation handling

**Example**: See [examples/search-r1/generate_with_search.py](../../../examples/search-r1/generate_with_search.py)

---

### 3. Reward Model (`--custom-rm-path`)

**Default**: `None` (uses built-in reward models based on `--rm-type`)

**Purpose**: Implement custom reward computation logic.

**Signature** (single sample mode):
```python
async def custom_rm(args, sample: Sample) -> float
```

**Signature** (batch mode, when `--group-rm` is enabled):
```python
async def batched_custom_rm(args, samples: list[Sample]) -> list[float]
```

**Use Cases**:
- Custom rule-based rewards
- Integration with external reward model services
- Multi-dimensional reward signals

**Built-in Options** (`--rm-type`):
- `math`: Mathematical answer verification
- `dapo`: DAPO-style scoring
- `deepscaler`: DeepScaler rule-based reward
- `f1`: F1 score computation
- `gpqa`: GPQA reward computation
- `ifbench`: IFBench reward computation
- `remote_rm`: Remote reward model service (requires `--rm-url`)

---

### 4. Dynamic Sampling Filter (`--dynamic-sampling-filter-path`)

**Default**: `None`

**Purpose**: Filter samples during dynamic sampling (e.g., DAPO-style filtering).

**Signature**:
```python
def filter_function(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput
```

**Return Type**:
```python
@dataclass
class DynamicFilterOutput:
    keep: bool  # Whether to keep this sample group
    reason: str | None  # Reason for filtering (for logging)
```

**Use Cases**:
- Filtering out samples where all responses have the same reward
- Implementing curriculum learning strategies
- Quality-based sample selection

**Example**: `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`

---

### 5. Buffer Filter (`--buffer-filter-path`)

**Default**: `None`

**Purpose**: Filter samples in the rollout buffer before training.

**Signature**:
```python
def buffer_filter(samples: list[list[Sample]]) -> list[list[Sample]]
```

**Use Cases**:
- Removing low-quality samples before training
- Implementing priority-based sample selection
- Balancing sample distributions

---

### 6. Rollout Sample Filter (`--rollout-sample-filter-path`)

**Default**: `None`

**Purpose**: Determine whether individual samples participate in loss calculation.

**Signature**:
```python
def filter_function(args, samples: list[Sample]) -> None
```

**Note**: This function should directly modify the `remove_sample` attribute of each `Sample` object.

**Use Cases**:
- Filtering samples based on response quality
- Implementing selective training strategies

---

### 7. Rollout All Samples Process (`--rollout-all-samples-process-path`)

**Default**: `None`

**Purpose**: Process all samples (including filtered ones) after rollout.

**Signature**:
```python
def process_function(args, samples: list[list[Sample]]) -> None
```

**Use Cases**:
- Logging and analysis of all generated samples
- Computing statistics across filtered and kept samples

---

### 8. Rollout Data Postprocess (`--rollout-data-postprocess-path`)

**Default**: `None`

**Purpose**: Post-process rollout data after log probabilities are computed.

**Signature**:
```python
def postprocess_function(args, samples: list[list[Sample]]) -> None
```

**Use Cases**:
- Updating loss masks based on computed values
- Adding additional metadata to samples

---

### 9. Custom Loss Function (`--custom-loss-function-path`)

**Default**: `None` (requires `--loss-type custom_loss`)

**Purpose**: Implement custom training loss computation.

**Use Cases**:
- Novel RL objectives
- Multi-objective optimization
- Custom regularization terms

---

### 10. Custom TIS/RS Function (`--custom-tis-function-path`)

**Default**: `None`

**Purpose**: Implement custom importance sampling for off-policy correction.

**Use Cases**:
- Custom importance sampling ratio computation
- Advanced off-policy correction methods

**Example**: `examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp`

---

### 11. Custom pg_loss Reducer (`--custom-pg-loss-reducer-function-path`)

**Default**: `None`

**Purpose**: Customize the reduction of pg_loss while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean.

**Signature**:
```python
def get_pg_loss_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]
```

**Use Cases**:
- Dr.GRPO: Divide by a constant instead of effective token count
- Custom loss normalization strategies

**Example**: `examples/DrGRPO/custom_reducer.py:get_pg_loss_reducer`

---

### 12. Reward Post-Processing (`--custom-reward-post-process-path`)

**Default**: `None` (uses default GRPO normalization)

**Purpose**: Custom post-processing of rewards before advantage computation.

**Use Cases**:
- Custom reward normalization strategies
- Reward shaping

---

### 13. Samples to Train Data Conversion (`--custom-convert-samples-to-train-data-path`)

**Default**: `None` (uses built-in conversion logic)

**Purpose**: Override the conversion of samples to training data format.

**Signature**:
```python
def convert_samples_to_train_data(
    args,
    samples: list[Sample] | list[list[Sample]],
) -> dict
```

**Return Type**:
```python
dict: {
    "tokens": list[list[int]],           # Token IDs for each sample
    "response_lengths": list[int],        # Response lengths
    "rewards": list[float],               # Normalized rewards
    "raw_reward": list[float],            # Raw rewards
    "truncated": list[int],               # Truncation flags (0 or 1)
    "sample_indices": list[int],          # Sample indices
    "loss_masks": list[list[int]],        # Loss masks for each sample
    # Optional fields:
    "round_number": list[int],            # Round numbers (for rollout buffer)
    "rollout_log_probs": list,            # Log probs (for off-policy correction)
    "rollout_routed_experts": list,       # Routed experts (for MoE)
    "metadata": list,                     # Train metadata
    "multimodal_train_inputs": list,      # Multimodal tensors (for VLM)
    "teacher_log_probs": list,            # Teacher log probs (for distillation)
}
```

**Use Cases**:
- Handling `list[list[Sample]]` inputs
- Custom data format requirements for training

---

### 14. Logging Functions

#### Training Rollout Logging (`--custom-rollout-log-function-path`)

**Signature**:
```python
def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool
```

**Return**: `True` to skip default logging, `False` to continue with default logging.

#### Evaluation Rollout Logging (`--custom-eval-rollout-log-function-path`)

**Signature**:
```python
def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool
```

**Return**: `True` to skip default logging, `False` to continue with default logging.

---

### 15. Data Source (`--data-source-path`)

**Default**: `slime.rollout.data_source.RolloutDataSourceWithBuffer`

**Purpose**: Override the data source for rollout prompts.

**Base Class**: `slime.rollout.data_source.DataSource`

**Required Methods**:
```python
class CustomDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Return num_samples samples"""
        
    def add_samples(self, samples: list[list[Sample]]):
        """Add samples back to the data source"""
        
    def save(self, rollout_id):
        """Save state for checkpointing"""
        
    def load(self, rollout_id=None):
        """Load state from checkpoint"""
```

---

### 16. Evaluation Function (`--eval-function-path`)

**Default**: Same as `--rollout-function-path`

**Purpose**: Override the rollout function specifically for evaluation.

**Use Cases**:
- Different sampling parameters for evaluation
- Evaluation-specific logic

---

### 17. Megatron Hooks

#### Megatron Initialization (`--custom-megatron-init-path`)

**Signature**:
```python
def custom_init(args) -> None
```

**Purpose**: Custom initialization after Megatron setup.

#### Before Log Prob Hook (`--custom-megatron-before-log-prob-hook-path`)

**Signature**:
```python
def custom_hook(args, model, store_prefix) -> None
```

**Purpose**: Custom logic before log probability computation.

#### Before Train Step Hook (`--custom-megatron-before-train-step-hook-path`)

**Signature**:
```python
def custom_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None
```

**Purpose**: Custom logic before each training step.

---

### 18. slime Router Middleware (`--slime-router-middleware-paths`)

**Purpose**: Add custom middleware to the slime router for request processing.

**Use Cases**:
- Request/response transformation
- Custom routing logic
- Caching and optimization


