# Agent Rollout Usage Documentation

### Starting Agent RL Training

First, you need to configure the Slime runtime environment according to the [Readme](../../README.md) documentation and cd to the Slime project directory.

#### Download Dataset and Model

```bash

huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Ensure you are in the Slime root directory

python ./slime_plugins/rollout_buffer/tools/assign_instance_id.py --input_path /root/dapo-math-17k/dapo-math-17k.jsonl

# Download model
mkdir /root/hf_models/
mkdir /root/megatron_model/

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir /root/hf_models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B --local-dir-use-symlinks False

# Convert model
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /root/hf_models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B  \
    --save /root/megatron_model/DeepSeek-R1-Distill-Qwen-7B-25.02
```

#### Start slime
```bash
chmod +x ./scripts/run_agent.sh
./scripts/run_agent.sh
```

## Overview

Agent Rollout is a specialized module in the Slime framework for agent task data generation, implementing a **fully asynchronous** data generation workflow based on external APIs. Unlike traditional reinforcement learning Rollout, Agent Rollout generates training data by interacting with external agent services (Rollout Buffer component in Slime Plugins).

**The main workflow is as follows:**

1. Slime sends a request to start Rollout to the Rollout Buffer via HTTP request
2. After receiving the request, Rollout Buffer starts executing Rollout tasks by accessing the SGLang Server launched in Slime
3. After Rollout Buffer begins executing Rollout tasks, Slime periodically calls the `/get_rollout_data` interface to obtain corresponding Rollout data and log information
4. When the amount of obtained data reaches the preset threshold, this data is stored in Slime's Data Buffer
5. Slime uses this data for model training

## Important Notes

- **Buffer Terminology Distinction**: "Rollout Buffer" in the documentation refers to the Buffer component in Slime Plugins, while "Slime Buffer" or "Data Buffer" refers to the data buffer within the Slime framework.
- **Parameter Passing Mechanism**: Start Rollout passes all parameter configurations from Slime to Rollout Buffer, and the specific Start Rollout implementation logic needs to be completed in Rollout Buffer.
- **Data Retrieval Retry Mechanism**: Get Rollout Data retrieves data from the Rollout Buffer port at fixed time intervals, and if errors occur, the retry count is increased. The retry count limit can be set through the `--fetch-trajectory-retry-times` parameter, with a default value of -1, indicating unlimited retries.
- **Return Data Format**: The default implementation requires that the return value of Get Rollout Data must contain multiple specific fields. Although this can be customized, it is strongly recommended to retain these key fields.
- **Unique Identifier**: Instance ID serves as a unique identifier for different data (similar to a primary key), typically of string type.
- **Data Format Conversion**: In the `generate_agent_rollout` function, when converting Rollout Data to Slime's internal storage universal format (Sample), the `prompt` field is not used during Agent Training. Since subsequent data sorting needs to be performed based on Index, it is essential to ensure that the Instance ID field is assigned to the Index field in Sample.
- **Statistical Information Recording**: When obtaining Rollout Data, in addition to basic Message, Reward and other results, statistical information can also be returned (such as data count before filtering, average reward before filtering, etc.). This additional information can be customized in the Log Raw Info function.
- **Log Prefix Convention**: When recording logs in this file, please use `rollout/` as a prefix. If you need to add new prefixes, you need to modify the initialization part accordingly, otherwise it may cause log information anomalies.
- **Asynchronous Training Characteristics**: Since Agent Training adopts a pure asynchronous mode, it may cause initially high data Reward (because the earliest generated data will be prioritized for training), which is normal.

## Core Components

### Main Functions

- **`generate_agent_rollout`**: Core data generation function, responsible for coordinating the entire Rollout process. It sequentially calls Start Rollout, Get Rollout Data, and Log Raw Info functions, and returns results after adding data to Slime Buffer.
- **`get_rollout_data`**: Retrieves generated data from Rollout Buffer, polling at fixed time intervals until sufficient data is obtained or the retry limit is reached.
- **`start_rollout`**: Starts Rollout tasks for external agent services. Slime is responsible for sending parameter configurations (including sampling parameters, Server URL, etc.) and notifying Rollout Buffer to start Rollout tasks.
- **`log_raw_info`**: Records and statistics meta-information during the Rollout process. Since different tasks and users have different logging requirements, this function can record required information in WandB, with specific information sourced from the return results of Get Rollout Data.

## Overall Workflow

### 1. Initialization Phase

When this round is the first training (if it's resume training, it will read previously trained data IDs, and then Rollout Buffer will skip these data), Agent Rollout will:
- Call `start_rollout()` to send a startup request to Rollout Buffer
- Pass necessary configuration parameters, including model URL, task type, input files, etc. (ensure file paths are accessible to Rollout Buffer)
- Rollout Buffer starts executing data generation tasks

### 2. Data Collection Phase

```python
while len(results) < args.rollout_batch_size * args.n_samples_per_prompt:
    time.sleep(5)  # polling interval
    data, meta_info = await get_rollout_data(api_base_url=base_url)
    results.extend(data)
    if meta_info:
        all_meta_info.append(meta_info)
```

- Periodically poll Rollout Buffer to retrieve generated data
- Accumulate data until reaching the expected batch size
- Collect meta-information for monitoring and logging

### 3. Data Processing Phase

```python
for record in results:
    oai_messages = record["messages"]
    
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
    token_ids, loss_mask = mask_generator.get_loss_mask(oai_messages)
    response_length = mask_generator.get_response_lengths([loss_mask])[0]
    
    sample = Sample(
        index=record["instance_id"],
        prompt=record["uid"],
        tokens=token_ids,
        response_length=response_length,
        reward=record["reward"],
        truncated=False,
        loss_mask=loss_mask[-response_length:],
        metadata={**record["extra_info"], "raw_reward": record["raw_reward"]}
    )
```

- Convert OpenAI format messages returned by external services to Token sequences
- Generate appropriate Loss Mask, supporting multi-turn conversations
- Create `Sample` objects that conform to Slime specifications

## Adding Custom Agent Tasks

If you need to customize Agent Rollout for specific agent tasks, you need to understand and possibly modify four core functions. The following details the purpose and interface specifications of each function:

### Core Function Details

#### 1. `generate_agent_rollout` - Main Control Function

**Function Description**: Serves as the main entry function for Agent Rollout, coordinating the entire data generation process, managing interactions with external services and converting data formats.

**Function Signature**:
```python
async def generate_agent_rollout(
    args, 
    rollout_id: int, 
    data_buffer: Buffer, 
    evaluation: bool = False
) -> List[Sample]
```

**Input Parameters**:
- `args`: Object containing all configuration parameters
- `rollout_id`: Unique identifier for the current Rollout, indicating how many times Rollout has been performed, also used as identifier for saving Checkpoint and resume training
- `data_buffer`: Slime's global data buffer for storing and managing sampled data samples
- `evaluation`: Whether in evaluation mode (not supported in current version), recommend evaluating Checkpoint separately

**Output**:
- `List[Sample]`: List of samples conforming to Slime specifications, each sample contains:
  - `tokens`: Token sequence
  - `response_length`: Response length (length after excluding Prompt Length), in multi-turn conversations it's the length from the first non-zero position of Loss Mask to the end
  - `reward`: Reward value. All Samples obtained in Slime should ensure they are valid Samples, specific filtering operations need to be handled in Rollout Buffer
  - `loss_mask`: Loss mask, currently only calculates loss for all Assistant type messages
  - `metadata`: Metadata information, containing various custom information

**Core Logic**:

1. Determine whether to start a new Rollout task by checking the attributes of Start Rollout (note that you cannot use whether the data length in Slime Buffer is 0 to judge, because according to different Slime Buffer Filter rules, Slime Buffer may also be 0 during training)
2. Poll to get data generated by external services, implemented through Get Rollout Data function

#### 2. `get_rollout_data` - Data Retrieval Function

**Function Description**: Asynchronously retrieves generated data from Rollout Buffer, supporting batch retrieval and meta-information collection.

**Function Signature**:
```python
async def get_rollout_data(
    api_base_url: str, 
    num: Optional[int] = None, 
    timeout: float = 100.0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]
```

**Input Parameters**:
- `api_base_url`: URL of Rollout Buffer
- `num`: Optional, specifies the number of data items to retrieve, default is None, meaning retrieve all data
- `timeout`: Request timeout time (seconds)

**Output**:
- `Tuple[List[Dict], Dict]`: Tuple containing data list and meta-information
  - Each dictionary in the data list must contain:
    ```python
    {
        "uid": "unique id",
        "instance_id": "instance id", 
        "messages": [openai format messages],
        "reward": 0.85,
        "extra_info": {...}
    }
    ```
  - Meta-information dictionary supports customization, just needs to be consistent with subsequent logging operations

**Error Handling**:
- Throws `aiohttp.ClientError` on network errors
- Throws `ValueError` on data format errors
- Throws `asyncio.TimeoutError` on timeout

#### 3. `start_rollout` - Task Startup Function

**Function Description**: Sends startup request to Rollout Buffer, configures task parameters and starts data generation process.

**Function Signature**:
```python
def start_rollout(api_base_url: str, args, metadata) -> Dict[str, Any]
```

**Input Parameters**:
- `api_base_url`: URL of Rollout Buffer
- `args`: Parameter object containing all configurations
- `metadata`: Metadata of data buffer, containing list of completed instance IDs for resume training

**Output**:
- `Dict[str, Any]`: Startup confirmation information returned by external service

**Sent Payload Format Example**:
```python
{
    "num_process": 1024,  # number of parallel processes
    "num_epoch": 3,       # generation epochs
    "remote_engine_url": "http://sglang-router:port",
    "task_type": "math",  # task type
    "input_file": "/path/to/input.jsonl",  # input file path
    "num_repeat_per_sample": "8",  # number of repetitions per sample
    "max_tokens": 8192,   # maximum number of tokens
    "sampling_params": {    # sampling parameters
        "max_tokens": 8192,
        "temperature": 0.8,
        "top_p": 0.9
    },
    "tokenizer_path": "/path/to/tokenizer",  # tokenizer path
    "skip_instance_ids": ["id1", "id2"]  # instance IDs to skip, skip previously trained instance IDs during resume training
}
```

#### 4. `log_raw_info` - Logging Function

**Function Description**: Collects, statistics and records key metrics during the Rollout process, supports WandB integration, content is fully customizable.

**Function Signature**:
```python
def log_raw_info(args, all_meta_info: List[Dict], rollout_id: int) -> None
```

**Input Parameters**:
- `args`: Configuration parameter object, including log-related settings
- `all_meta_info`: List of meta-information collected from all data retrieval requests
- `rollout_id`: Current Rollout ID

**Log Format Example**:
```python
{
    "rollout/no_filter/total_samples": 100,
    "rollout/no_filter/avg_reward": 0.75,
}
```

## Custom Agent Task Implementation Steps

### 1. Prepare Input Data Format

Your input file (`--prompt-data`) should contain task-related data, for example:

```jsonl
{"instance_id": "math_001", "prompt": [{"role":"user","content":"Solve the equation x^2 + 5x + 6 = 0"}]}
{"instance_id": "math_002", "prompt": [{"role":"user","content":"Prove Fermat's Last Theorem"}]}
```

Where `instance_id` is a required field, you can use `slime_plugins/rollout_buffer/tools/assign_instance_id.py` to automatically generate Instance ID for each data item.

### 2. Custom Loss Mask Generator

If your task has special conversation formats, you need to add a custom Mask generator in `slime/utils/mask_utils.py`:

```python
class CustomTaskLossMaskGenerator(MultiTurnLossMaskGenerator):
    def get_custom_multi_turn_loss_mask(self, messages):
        # return (token_ids, loss_mask)
        pass
        
    def get_loss_mask(self, messages: List[Dict]) -> List[int]:
        # ... existing code ...
        elif self.tokenizer_type == "custom_task":
            return xxx
        # ... existing code ...
```

Then use:
```bash
--loss-mask-type custom_task
```

Currently, Loss Mask generation for multi-turn conversations in Qwen and Distill Qwen conversation formats has been implemented.

### 3. Implement External Service Interface

In Rollout Buffer, you need to implement the key function `run_rollout()` for your task. For specific implementation, please refer to [Rollout Buffer Usage Documentation](./rollout_buffer_usage.md).

## Parameter Configuration

### Agent Rollout Core Parameters

- **`--rollout-function-path`**: Specify using Agent Rollout function
  ```bash
  --rollout-function-path slime.rollout.agent_rollout.generate_rollout
  ```

- **`--agent-rollout-buffer-url`**: API address of Rollout Buffer
  ```bash
  --agent-rollout-buffer-url http://0.0.0.0:8889
  ```
  For specific deployment methods, please check Rollout Buffer documentation

- **`--prompt-data`**: Input data file path, containing task data to be processed
  ```bash
  --prompt-data ./deepscaler_rl_buffer_instance_id.jsonl
  ```

- **`--loss-mask-type`**: Loss Mask type, used for Token masking in multi-turn conversations, supports customization, custom path is in `slime/utils/mask_utils.py`
  ```bash
  --loss-mask-type distill_qwen
  ```

### Rollout Generation Parameters

- **`--num-rollout`**: Total number of Rollouts (maximum Rollout ID count)
  ```bash
  --num-rollout 3000
  ```

- **`--rollout-batch-size`**: Batch size for each Rollout
  ```bash
  --rollout-batch-size 16
  ```

- **`--rollout-max-response-len`**: Maximum response length (single turn)
  ```bash
  --rollout-max-response-len 8192
  ```

- **`--rollout-temperature`**: Sampling temperature parameter
  ```bash
  --rollout-temperature 0.8
  ```

- **`--n-samples-per-prompt`**: Number of samples generated per Prompt
  ```bash
  --n-samples-per-prompt 8
  ```

### Filtering and Reward Configuration

- **`--buffer-filter-path`**: Data buffer filter path, supports customization, defaults to using the latest batch of data in Buffer for updates.

- **`--disable-rewards-normalization`**: Disable reward normalization, if rewards are already normalized in Rollout Buffer, please enable this parameter
  ```bash
  --disable-rewards-normalization
  ``` 