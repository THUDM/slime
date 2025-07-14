## Agent Rollout 使用文档


### 启动 Agent RL Training 

首先需要根据[Readme](../../README_zh.md)文档中配置好 Slime的运行环境并且 cd 到 Slime项目的目录下。

#### 下载数据集和模型

```bash

huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

#确保在 Slime 根目录下

python ./slime_plugins/rollout_buffer/tools/assign_instance_id.py --input_path /root/dapo-math-17k/dapo-math-17k.jsonl

# 下载模型
mkdir /root/hf_models/
mkdir /root/megatron_model/

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir /root/hf_models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B --local-dir-use-symlinks False

# 转换模型
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /root/hf_models/deepseek-ai--DeepSeek-R1-Distill-Qwen-7B  \
    --save /root/megatron_model/DeepSeek-R1-Distill-Qwen-7B-25.02
```

#### 启动slime
```bash
chmod +x ./scripts/run_agent.sh
./scripts/run_agent.sh
```
### 流程概述

Agent Rollout 是 Slime 框架中用于智能体任务数据生成的专用模块，实现了基于外部 API 的**完全异步**数据生成流程。与传统的强化学习 Rollout 不同，Agent Rollout 通过与外部智能体服务（Slime Plugins 中的 Rollout Buffer 组件）进行交互来生成训练数据。

**主要工作流程如下：**

1. Slime 通过 HTTP 请求向 Rollout Buffer 发送开始 Rollout 的指令
2. Rollout Buffer 接收到请求后，通过访问 Slime 中启动的 SGLang Server 开始执行 Rollout 任务
3. 在 Rollout Buffer 开始执行 Rollout 任务后，Slime 会定期调用 `/get_rollout_data` 接口获取对应的 Rollout 数据和日志信息
4. 当获取的数据量达到预设阈值时，这些数据会被存储到 Slime 的 Data Buffer 中
5. Slime 使用这些数据进行模型训练

### 重要说明

- **Buffer 术语区分**：文档中的 "Rollout Buffer" 指 Slime Plugins 中的 Buffer 组件，而 "Slime Buffer" 或 "Data Buffer" 指 Slime 框架内部的数据缓冲区。
- **参数传递机制**：Start Rollout 会将 Slime 的所有参数配置传递给 Rollout Buffer，具体的 Start Rollout 实现逻辑需要在 Rollout Buffer 中完成。
- **数据获取重试机制**：Get Rollout Data 按固定时间间隔从 Rollout Buffer 端口获取数据，如果发生错误则会增加重试次数。重试次数上限可通过 `--fetch-trajectory-retry-times` 参数设置，默认值为 -1，表示无限重试。
- **返回数据格式**：默认实现要求 Get Rollout Data 的返回值必须包含多个特定的字段，虽然可以自定义修改，但强烈建议保留这几个关键字段。
- **唯一标识符**：Instance ID 作为不同数据的唯一标识符（类似主键），通常为字符串类型。
- **数据格式转换**：在 `generate_agent_rollout` 函数中，将 Rollout Data 转换为 Slime 内部存储的通用格式（Sample）时，`prompt` 字段在 Agent Training 过程中不会被使用。由于后续需要根据 Index 对数据进行排序，因此必须确保将 Instance ID 字段赋值给 Sample 中的 Index 字段。
- **统计信息记录**：获取 Rollout Data 时，除了基本的 Message、Reward 等结果外，还可以返回统计信息（如过滤前的数据数量、过滤前的奖励均值等），这些额外信息可以在 Log Raw Info 函数中自定义记录方式。
- **日志前缀规范**：在该文件中记录日志时，请使用 `rollout/` 作为前缀。如需添加新的前缀，需要相应修改初始化部分，否则可能导致日志信息异常。
- **异步训练特性**：由于 Agent Training 采用纯异步模式，可能导致最初的数据 Reward 较高（因为最早生成完成的数据会优先用于训练），这是正常现象。

### 核心组件

#### 主要函数

- **`generate_agent_rollout`**：核心数据生成函数，负责协调整个 Rollout 流程。依次调用 Start Rollout、Get Rollout Data 和 Log Raw Info 函数，并将数据加入 Slime Buffer 后返回结果。
- **`get_rollout_data`**：从 Rollout Buffer 获取生成的数据，按固定时间间隔轮询，直到获取足够数据或达到重试次数上限。
- **`start_rollout`**：启动外部智能体服务的 Rollout 任务。Slime 负责发送参数配置（包括采样参数、Server URL 等），通知 Rollout Buffer 开始 Rollout 任务。
- **`log_raw_info`**：记录和统计 Rollout 过程中的元信息。由于不同任务和用户的日志需求不同，可通过该函数在 WandB 中记录所需信息，具体信息来源于 Get Rollout Data 的返回结果。

### 整体流程

#### 1. 初始化阶段

当该轮次为第一次训练时（如果是续训将会读取之前训练过的数据 ID，随后 Rollout Buffer 会跳过这些数据），Agent Rollout 会：
- 调用 `start_rollout()` 向 Rollout Buffer 发送启动请求
- 传递必要的配置参数，包括模型 URL、任务类型、输入文件等（确保文件路径是 Rollout Buffer 可以访问的）
- Rollout Buffer 开始执行数据生成任务

#### 2. 数据收集阶段

```python
while len(results) < args.rollout_batch_size * args.n_samples_per_prompt:
    time.sleep(5)  # 轮询间隔
    data, meta_info = await get_rollout_data(api_base_url=base_url)
    results.extend(data)
    if meta_info:
        all_meta_info.append(meta_info)
```

- 定期轮询 Rollout Buffer 获取生成的数据
- 累积数据直到达到预期的批次大小
- 收集元信息用于监控和日志记录

#### 3. 数据处理阶段

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

- 将外部服务返回的 OpenAI 格式消息转换为 Token 序列
- 生成适当的 Loss Mask，支持多轮对话
- 创建符合 Slime 规范的 `Sample` 对象

### 添加自定义 Agent Task

如果您需要为特定的智能体任务定制 Agent Rollout，需要理解并可能修改四个核心函数。以下详细说明每个函数的目的和接口规范：

#### 核心函数详解

##### 1. `generate_agent_rollout` - 主控制函数

**功能描述**：作为 Agent Rollout 的主入口函数，协调整个数据生成流程，管理与外部服务的交互并转换数据格式。

**函数签名**：
```python
async def generate_agent_rollout(
    args, 
    rollout_id: int, 
    data_buffer: Buffer, 
    evaluation: bool = False
) -> List[Sample]
```

**输入参数**：
- `args`: 包含所有配置参数的对象
- `rollout_id`: 当前 Rollout 的唯一标识符，表示当前是第几次 Rollout，同时也会被用作保存 Checkpoint 和续训的标识符
- `data_buffer`: Slime 的全局数据缓冲区，用于存储和管理采样的数据样本
- `evaluation`: 是否为评估模式（当前版本不支持），建议单独对 Checkpoint 进行评估

**输出**：
- `List[Sample]`: 符合 Slime 规范的样本列表，每个样本包含：
  - `tokens`: Token 序列
  - `response_length`: 响应长度（长度是剔除 Prompt Length 后的总长度），在多轮对话中为 Loss Mask 第一个非零位置到最后的长度
  - `reward`: 奖励值。Slime 中所有获取的 Sample 请确保都是有效的 Sample，具体过滤操作需要在 Rollout Buffer 中进行处理
  - `loss_mask`: 损失掩码，目前只对所有 Assistant 类型的消息进行损失计算
  - `metadata`: 元数据信息，包含各种自定义信息

**核心逻辑**：

1. 判断是否需要启动新的 Rollout 任务，通过查看 Start Rollout 的属性来判断（注意此处不可以使用 Slime Buffer 中的数据长度是否为 0 来判断，因为根据不同的 Slime Buffer Filter 规则，可能会导致 Slime Buffer 在训练过程中也可能为 0）
2. 轮询获取外部服务生成的数据，通过 Get Rollout Data 函数实现

##### 2. `get_rollout_data` - 数据获取函数

**功能描述**：异步从 Rollout Buffer 中获取已生成的数据，支持批量获取和元信息收集。

**函数签名**：
```python
async def get_rollout_data(
    api_base_url: str, 
    num: Optional[int] = None, 
    timeout: float = 100.0
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]
```

**输入参数**：
- `api_base_url`: Rollout Buffer 的 URL
- `num`: 可选，指定获取的数据条数，默认为 None，意味着获取全部数据
- `timeout`: 请求超时时间（秒）

**输出**：
- `Tuple[List[Dict], Dict]`: 包含数据列表和元信息的元组
  - 数据列表中每个字典必须包含：
    ```python
    {
        "uid": "唯一标识符",
        "instance_id": "实例ID", 
        "messages": [OpenAI格式的对话],
        "reward": 0.85,  # 奖励值
        "extra_info": {...}  # 额外信息
    }
    ```
  - 元信息字典支持自定义，只需要和后续日志记录等操作保持一致即可

**错误处理**：
- 网络错误时抛出 `aiohttp.ClientError`
- 数据格式错误时抛出 `ValueError`
- 超时时抛出 `asyncio.TimeoutError`

##### 3. `start_rollout` - 任务启动函数

**功能描述**：向 Rollout Buffer 发送启动请求，配置任务参数并开始数据生成流程。

**函数签名**：
```python
def start_rollout(api_base_url: str, args, metadata) -> Dict[str, Any]
```

**输入参数**：
- `api_base_url`: Rollout Buffer 的 URL
- `args`: 包含所有配置的参数对象
- `metadata`: 数据缓冲区的元数据，包含已完成的实例 ID 列表，用于续训

**输出**：
- `Dict[str, Any]`: 外部服务返回的启动确认信息

**发送的载荷格式示例**：
```python
{
    "num_process": 1024,  # 并行进程数
    "num_epoch": 3,       # 生成轮次
    "remote_engine_url": "http://sglang-router:port",
    "task_type": "math",  # 任务类型
    "input_file": "/path/to/input.jsonl",  # 输入文件路径
    "num_repeat_per_sample": "8",  # 每个样本重复次数
    "max_tokens": 8192,   # 最大 Token 数
    "sampling_params": {    # 采样参数
        "max_tokens": 8192,
        "temperature": 0.8,
        "top_p": 0.9
    },
    "tokenizer_path": "/path/to/tokenizer",  # Tokenizer 路径
    "skip_instance_ids": ["id1", "id2"]  # 跳过的实例 ID，续训时跳过之前已经训练过的 Instance ID
}
```

##### 4. `log_raw_info` - 日志记录函数

**功能描述**：收集、统计和记录 Rollout 过程中的关键指标，支持 WandB 集成，内容均可自定义。

**函数签名**：
```python
def log_raw_info(args, all_meta_info: List[Dict], rollout_id: int) -> None
```

**输入参数**：
- `args`: 配置参数对象，包含日志相关设置
- `all_meta_info`: 从所有数据获取请求中收集的元信息列表
- `rollout_id`: 当前 Rollout 的 ID

**日志格式示例**：
```python
{
    "rollout/no_filter/total_samples": 100,
    "rollout/no_filter/avg_reward": 0.75,
}
```

### 自定义 Agent Task 实现步骤

#### 1. 准备输入数据格式

您的输入文件（`--prompt-data`）应包含任务相关的数据，例如：

```jsonl
{"instance_id": "math_001", "prompt": [{"role":"user","content":"求解方程 x^2 + 5x + 6 = 0"}]}
{"instance_id": "math_002", "prompt": [{"role":"user","content":"证明费马大定理"}]}
```

其中 `instance_id` 为必需字段，可以使用 `slime_plugins/rollout_buffer/tools/assign_instance_id.py` 自动生成每条数据的 Instance ID。

#### 2. 自定义 Loss Mask 生成器

如果您的任务有特殊的对话格式，需要在 `slime/utils/mask_utils.py` 中添加自定义的 Mask 生成器：

```python
class CustomTaskLossMaskGenerator(MultiTurnLossMaskGenerator):
    def get_custom_multi_turn_loss_mask(self, messages):
        # 自定义实现
        # 返回 (token_ids, loss_mask)
        pass
        
    def get_loss_mask(self, messages: List[Dict]) -> List[int]:
        # ... existing code ...
        elif self.tokenizer_type == "custom_task":
            return xxx
        # ... existing code ...
```

然后使用：
```bash
--loss-mask-type custom_task
```

目前已经实现了 Qwen 和 Distill Qwen 两种对话格式的多轮对话 Loss Mask 生成。

#### 3. 实现外部服务接口

在 Rollout Buffer 中需要针对您的任务实现关键函数 `run_rollout()`，具体实现请参见 [Rollout Buffer 使用文档](./rollout_buffer_usage.md)。

### 参数配置

#### Agent Rollout 核心参数

- **`--rollout-function-path`**：指定使用 Agent Rollout 函数
  ```bash
  --rollout-function-path slime.rollout.agent_rollout.generate_rollout
  ```

- **`--agent-rollout-buffer-url`**：Rollout Buffer 的 API 地址
  ```bash
  --agent-rollout-buffer-url http://0.0.0.0:8889
  ```
  具体部署方式请查看 Rollout Buffer 文档

- **`--prompt-data`**：输入数据文件路径，包含待处理的任务数据
  ```bash
  --prompt-data ./deepscaler_rl_buffer_instance_id.jsonl
  ```

- **`--loss-mask-type`**：Loss Mask 类型，用于多轮对话的 Token 掩码，支持自定义，自定义路径在 `slime/utils/mask_utils.py`
  ```bash
  --loss-mask-type distill_qwen
  ```

#### Rollout 生成参数

- **`--num-rollout`**：总的 Rollout 次数（最大 Rollout ID 数）
  ```bash
  --num-rollout 3000
  ```

- **`--rollout-batch-size`**：每次 Rollout 的批次大小
  ```bash
  --rollout-batch-size 16
  ```

- **`--rollout-max-response-len`**：最大响应长度（单轮）
  ```bash
  --rollout-max-response-len 8192
  ```

- **`--rollout-temperature`**：采样温度参数
  ```bash
  --rollout-temperature 0.8
  ```

- **`--n-samples-per-prompt`**：每个 Prompt 生成的样本数
  ```bash
  --n-samples-per-prompt 8
  ```

#### 过滤与奖励配置

- **`--buffer-filter-path`**：数据缓冲区过滤器路径，支持自定义，默认采用 Buffer 中最新的一批数据进行更新。

- **`--disable-rewards-normalization`**：禁用奖励归一化，如果 Rollout Buffer 中已经归一化奖励，请启用此参数
  ```bash
  --disable-rewards-normalization
  ```

