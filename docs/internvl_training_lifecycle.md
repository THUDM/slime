# InternVL3.5-4B 训练数据完整生命周期

本文档详细介绍一条 KIE (Key Information Extraction) 训练数据在 SLIME 框架中的完整生命周期。

## 1. 数据加载阶段

### 1.1 原始数据格式

训练数据存储在 `data/kie_train.parquet` 中，每条数据包含：

```python
{
    "problem": "<image>\n请从图片中提取关键信息，以JSON格式输出...",
    "answer": '{"company": "XX公司", "date": "2024-01-01", "total": "1000.00"}',
    "images": ["/path/to/document.jpg"],
    "id": "sample_001",
    "source_file": "wildreceipt_synthesized_v3_json.jsonl"
}
```

### 1.2 数据加载流程

```
train_internvl_kie.py
    ↓
slime/train.py: main()
    ↓
slime/utils/data.py: Dataset.__init__()
    ↓
pandas.read_parquet() → DataFrame → List[dict]
```

关键代码位置：`slime/utils/data.py:45-80`

```python
class Dataset:
    def __init__(self, args, tokenizer, processor=None):
        # 加载 parquet 数据
        df = pd.read_parquet(args.prompt_data)
        self.data = df.to_dict("records")

        # 过滤过长的 prompt
        self.data = filter_long_prompt(
            self.data,
            tokenizer,
            processor,
            args.rollout_max_response_len
        )
```

## 2. 采样分组阶段

### 2.1 Rollout 批次构建

每个 rollout 批次从数据集中采样 `rollout_batch_size=16` 条数据。

```
slime/rollout/rollout.py: RolloutManager.generate()
    ↓
random.sample(dataset, rollout_batch_size)
    ↓
16 条原始样本
```

### 2.2 多采样扩展

每条 prompt 会被采样 `n_samples_per_prompt=4` 次，生成多个响应用于 GRPO 组内比较：

```python
# 原始: 16 条 prompt
# 扩展后: 16 × 4 = 64 条待生成样本

expanded_prompts = []
for prompt in batch:
    for _ in range(n_samples_per_prompt):
        expanded_prompts.append(prompt.copy())
```

## 3. SGLang 推理生成阶段

### 3.1 InternVL 模型加载

SGLang 引擎加载 InternVL3.5-4B 模型：

```
slime/rollout/sglang_engine.py: SGLangEngine.__init__()
    ↓
sglang.Engine(
    model_path="/path/to/InternVL3_5-4B",
    mem_fraction_static=0.6,
    cuda_graph_max_bs=32
)
```

### 3.2 图像预处理

InternVL 的图像处理流程：

```python
# slime/utils/processing_utils.py

def process_vision_info_internvl(prompt, processor):
    """处理 InternVL 的视觉输入"""
    images = []
    for message in prompt:
        content = message.get("content", [])
        for item in content:
            if item.get("type") == "image":
                image_path = item.get("image")
                # 加载图像 (支持路径、URL、base64)
                img = load_image(image_path)
                images.append(img)
    return {"images": images, "videos": []}
```

### 3.3 Chat Template 应用

将数据转换为 InternVL 的对话格式：

```python
# 输入 prompt
prompt = "<image>\n请从图片中提取关键信息..."

# 应用 chat template 后
formatted = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
请从图片中提取关键信息...<|im_end|>
<|im_start|>assistant
"""
```

### 3.4 生成响应

SGLang 使用 InternVL 生成响应：

```python
# slime/rollout/sglang_engine.py: generate()

responses = engine.generate(
    prompts=formatted_prompts,
    images=batch_images,
    sampling_params={
        "temperature": 0.7,
        "max_new_tokens": 2048,
        "top_p": 0.95
    }
)

# 生成结果示例
response = '{"company": "XX科技有限公司", "date": "2024-01-15", "total": "1280.50"}'
```

## 4. 奖励计算阶段

### 4.1 KIE 奖励函数

使用自定义的 KIE 奖励函数评估生成质量：

```python
# slime/rollout/rm_hub/kie_reward.py

async def kie_reward(args, sample: Sample, **kwargs) -> float:
    """
    KIE 任务奖励函数

    评估维度:
    1. JSON 格式正确性 (10%)
    2. Key 完整性
    3. Value 准确性 (90%)
    """
    response = sample.response  # 模型生成
    label = sample.label        # 真实答案

    # 解析 JSON
    pred_dict = extract_json_from_response(response)
    gt_dict = json.loads(label)

    if pred_dict is None:
        return 0.1 if any_value_in_response else 0.0

    # 计算内容分数
    content_score = compute_dict_score(pred_dict, gt_dict)

    # 最终奖励 = 内容分数 × 0.9 + 格式正确奖励 0.1
    return min(1.0, content_score * 0.9 + 0.1)
```

### 4.2 分数计算细节

```python
def compute_dict_score(pred_dict, gt_dict):
    """计算字典匹配分数"""
    total_score = 0.0

    for key, gt_value in gt_dict.items():
        if key not in pred_dict:
            continue  # 缺失 key，不得分

        pred_value = pred_dict[key]

        # 精确匹配: 1.0 分
        if normalize(pred_value) == normalize(gt_value):
            score = 1.0
        # 包含关系: 0.8 分
        elif gt_value in pred_value or pred_value in gt_value:
            score = 0.8
        # 相似度匹配
        else:
            score = string_similarity(pred_value, gt_value)

        total_score += score

    return total_score / len(gt_dict)
```

### 4.3 奖励示例

```python
# 真实答案
gt = {"company": "XX公司", "date": "2024-01-01", "total": "1000.00"}

# 生成响应 1 (高质量)
pred1 = {"company": "XX公司", "date": "2024-01-01", "total": "1000.00"}
reward1 = 1.0  # 完全匹配

# 生成响应 2 (部分正确)
pred2 = {"company": "XX公司", "date": "2024-01-02", "total": "1000"}
reward2 = 0.73  # company 正确, date 错误, total 部分匹配

# 生成响应 3 (格式错误)
pred3 = "公司名称是XX公司，日期是2024年1月1日"
reward3 = 0.1  # 无法解析 JSON，但包含部分正确内容

# 生成响应 4 (完全错误)
pred4 = '{"error": "无法识别"}'
reward4 = 0.1  # JSON 格式正确但内容错误
```

## 5. GRPO 优势估计阶段

### 5.1 组内奖励标准化

GRPO 的核心思想是在同一 prompt 的多个响应组内进行相对比较：

```python
# slime/algorithms/grpo.py

def compute_grpo_advantages(rewards, group_size=4):
    """
    计算 GRPO 优势值

    对于每组 n_samples_per_prompt 个响应:
    advantage = (reward - group_mean) / group_std
    """
    advantages = []

    for i in range(0, len(rewards), group_size):
        group_rewards = rewards[i:i+group_size]

        mean = np.mean(group_rewards)
        std = np.std(group_rewards) + 1e-8

        group_advantages = [(r - mean) / std for r in group_rewards]
        advantages.extend(group_advantages)

    return advantages
```

### 5.2 优势计算示例

```python
# 同一 prompt 的 4 个响应奖励
group_rewards = [1.0, 0.73, 0.1, 0.1]

mean = 0.4825
std = 0.396

# 计算优势
advantages = [
    (1.0 - 0.4825) / 0.396,   # = 1.31  (最好的响应，正优势)
    (0.73 - 0.4825) / 0.396,  # = 0.62  (较好，正优势)
    (0.1 - 0.4825) / 0.396,   # = -0.97 (较差，负优势)
    (0.1 - 0.4825) / 0.396,   # = -0.97 (较差，负优势)
]
```

## 6. 数据打包与分发阶段

### 6.1 序列打包

将多个样本打包成一个长序列，提高 GPU 利用率：

```python
# slime/utils/packing.py

def pack_sequences(samples, max_length=8192):
    """
    将多个样本打包成一个序列

    [sample1][sample2][sample3]...
    """
    packed = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "position_ids": [],
        "advantages": [],
        "multimodal_train_inputs": {}
    }

    current_length = 0
    for sample in samples:
        seq_len = len(sample["input_ids"])
        if current_length + seq_len > max_length:
            yield packed
            packed = reset_packed()
            current_length = 0

        # 添加样本到打包序列
        packed["input_ids"].extend(sample["input_ids"])
        packed["advantages"].extend([sample["advantage"]] * seq_len)
        current_length += seq_len
```

### 6.2 FSDP 分发

使用 Ray 将打包数据分发到各个 GPU：

```python
# slime/backends/fsdp_utils/actor.py

@ray.remote(num_gpus=1)
class FSDPActor:
    def train_step(self, packed_batch):
        """单个 GPU 的训练步骤"""
        # 数据已经通过 Ray 分发到本地
        self.model.train()

        # 准备输入
        inputs = self._get_model_inputs_args(packed_batch)

        # 前向传播
        outputs = self.model(**inputs)

        # 计算损失
        loss = self.compute_ppo_loss(outputs, packed_batch)

        # 反向传播
        loss.backward()
        self.optimizer.step()
```

## 7. InternVL 前向传播阶段

### 7.1 模型架构

InternVL3.5-4B 的架构：

```
InternVLChatModel
├── vision_model (InternViT-300M)
│   ├── embeddings
│   ├── encoder (24 layers)
│   └── output: [batch, num_patches, 1024]
│
├── mlp1 (Vision-Language Projector)
│   ├── Linear(1024 → 2048)
│   ├── GELU
│   └── Linear(2048 → 3584)
│   └── output: [batch, num_patches, 3584]
│
└── language_model (Qwen3-4B)
    ├── embed_tokens
    ├── layers (36 layers)
    └── lm_head
```

### 7.2 前向传播流程

```python
# InternVL forward pass

def forward(self, input_ids, pixel_values, image_flags, attention_mask, labels):
    # 1. 视觉编码
    # pixel_values: [num_images, 3, 448, 448]
    vision_outputs = self.vision_model(pixel_values)
    # vision_outputs: [num_images, 1024, 1024]

    # 2. 视觉特征投影
    # 将视觉特征投影到语言模型的维度
    vit_embeds = self.mlp1(vision_outputs)
    # vit_embeds: [num_images, 1024, 3584]

    # 3. 特征融合
    # 根据 image_flags 将视觉特征插入到文本序列中
    # <image> token 位置被替换为视觉特征
    inputs_embeds = self.language_model.embed_tokens(input_ids)

    # 找到 <image> token 位置并替换
    image_token_id = self.config.image_token_id  # 通常是特殊 token
    for batch_idx in range(batch_size):
        image_positions = (input_ids[batch_idx] == image_token_id).nonzero()
        for pos, img_embed in zip(image_positions, vit_embeds[batch_idx]):
            inputs_embeds[batch_idx, pos] = img_embed

    # 4. 语言模型前向传播
    outputs = self.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels
    )

    return outputs
```

### 7.3 image_flags 的作用

```python
# slime/backends/fsdp_utils/actor.py: _get_model_inputs_args()

def _get_model_inputs_args(self, packed_sequence):
    model_args = {
        "input_ids": packed_sequence["input_ids"],
        "attention_mask": packed_sequence["attention_mask"],
        "labels": packed_sequence["labels"],
    }

    if packed_sequence.get("multimodal_train_inputs"):
        mm_inputs = packed_sequence["multimodal_train_inputs"]
        model_args.update(mm_inputs)

        # InternVL 需要 image_flags 标记哪些图像是真实的
        if "pixel_values" in mm_inputs and "image_flags" not in mm_inputs:
            pixel_values = mm_inputs["pixel_values"]
            num_images = pixel_values.shape[0]
            # 全 1 表示所有图像都是真实的（非 padding）
            image_flags = torch.ones(num_images, 1, dtype=torch.long)
            model_args["image_flags"] = image_flags

    return model_args
```

## 8. PPO 损失计算阶段

### 8.1 损失函数组成

```python
# slime/algorithms/ppo_loss.py

def compute_ppo_loss(logits, labels, advantages, old_log_probs, args):
    """
    PPO 损失 = Policy Loss + KL Loss + Entropy Loss
    """
    # 1. 计算当前策略的 log probabilities
    log_probs = compute_log_probs(logits, labels)

    # 2. 计算 ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # 3. Policy Loss (Clipped PPO)
    # 限制策略更新幅度，防止过大变化
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - args.eps_clip, 1 + args.eps_clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 4. KL Loss (可选)
    # 防止策略偏离参考策略太远
    if args.kl_loss_coef > 0:
        kl_loss = compute_kl_divergence(log_probs, ref_log_probs)
        kl_loss = args.kl_loss_coef * kl_loss.mean()
    else:
        kl_loss = 0

    # 5. Entropy Loss (可选)
    # 鼓励探索，防止策略过早收敛
    if args.entropy_coef > 0:
        entropy = compute_entropy(logits)
        entropy_loss = -args.entropy_coef * entropy.mean()
    else:
        entropy_loss = 0

    total_loss = policy_loss + kl_loss + entropy_loss
    return total_loss
```

### 8.2 损失计算示例

```python
# 假设一个 token 的计算

# 当前策略输出
logits = model(input_ids)  # [batch, seq_len, vocab_size]
log_prob = -2.3  # 当前策略下生成该 token 的 log probability

# 旧策略（生成时）的 log probability
old_log_prob = -2.5

# 优势值（来自 GRPO）
advantage = 1.31  # 正优势，应该增加这个 token 的概率

# 计算 ratio
ratio = exp(-2.3 - (-2.5)) = exp(0.2) = 1.22

# Clipped ratio (eps_clip=0.2)
clipped_ratio = clip(1.22, 0.8, 1.2) = 1.2

# Policy loss
surr1 = 1.22 * 1.31 = 1.60
surr2 = 1.2 * 1.31 = 1.57
policy_loss = -min(1.60, 1.57) = -1.57

# 负损失意味着梯度会增加这个 token 的概率
```

## 9. 反向传播与参数更新阶段

### 9.1 FSDP 梯度同步

```python
# FSDP 自动处理梯度分片和同步

class FSDPActor:
    def __init__(self):
        # 使用 FSDP 包装模型
        self.model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32
            ),
            use_orig_params=True
        )

    def train_step(self, batch):
        self.optimizer.zero_grad()

        # 前向传播
        loss = self.compute_loss(batch)

        # 反向传播 - FSDP 自动处理梯度分片
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 参数更新 - FSDP 自动同步
        self.optimizer.step()
```

### 9.2 梯度检查点

为了节省显存，使用梯度检查点：

```python
# 配置: --gradient-checkpointing

# 前向传播时不保存中间激活值
# 反向传播时重新计算

# InternVL 的梯度检查点配置
model.gradient_checkpointing_enable()

# 这样可以训练更大的 batch size
# 代价是增加约 30% 的计算时间
```

## 10. 权重同步阶段

### 10.1 训练权重 → SGLang 引擎

每个 rollout 结束后，需要将更新的权重同步到 SGLang 推理引擎：

```python
# slime/rollout/rollout.py

class RolloutManager:
    def sync_weights(self):
        """同步训练权重到推理引擎"""
        # 1. 从 FSDP Actor 收集完整权重
        full_state_dict = self.trainer.get_full_state_dict()

        # 2. 更新 SGLang 引擎的权重
        self.sglang_engine.update_weights(full_state_dict)

        # 3. 清理旧的 KV cache
        self.sglang_engine.clear_cache()
```

### 10.2 Colocate 模式

在 `--colocate` 模式下，训练和推理共用 GPU：

```python
# 训练时: 模型在 GPU 上进行梯度计算
# 推理时: 同一模型用于生成

# 权重同步是原地进行的，无需跨设备传输
# 但需要在训练和推理之间切换模型状态

def switch_to_inference(self):
    self.model.eval()
    torch.cuda.empty_cache()

def switch_to_training(self):
    self.model.train()
```

## 11. 完整训练循环

```python
# 伪代码展示完整训练循环

for rollout_idx in range(num_rollout):  # 100 轮
    # 1. 采样数据
    batch = dataset.sample(rollout_batch_size)  # 16 条

    # 2. 扩展为多采样
    expanded = expand_samples(batch, n_samples_per_prompt)  # 64 条

    # 3. SGLang 生成响应
    responses = sglang_engine.generate(expanded)

    # 4. 计算奖励
    rewards = [kie_reward(sample) for sample in responses]

    # 5. GRPO 优势估计
    advantages = compute_grpo_advantages(rewards, group_size=4)

    # 6. 打包数据
    packed_batches = pack_sequences(responses, advantages)

    # 7. 分发到各 GPU 训练
    for micro_batch in packed_batches:
        # 前向传播
        loss = model(micro_batch)

        # 反向传播
        loss.backward()

        # 梯度累积
        if step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # 8. 同步权重到 SGLang
    sync_weights_to_sglang()

    # 9. 保存检查点
    if rollout_idx % save_interval == 0:
        save_checkpoint(f"checkpoint_{rollout_idx}")

    # 10. 日志记录
    wandb.log({
        "rollout": rollout_idx,
        "mean_reward": np.mean(rewards),
        "loss": loss.item()
    })
```

## 12. 关键配置参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--num-rollout` | 100 | 总训练轮数 |
| `--rollout-batch-size` | 16 | 每轮采样的 prompt 数 |
| `--n-samples-per-prompt` | 4 | 每个 prompt 的采样次数 |
| `--global-batch-size` | 64 | 全局 batch size (16×4) |
| `--rollout-temperature` | 0.7 | 生成温度 |
| `--rollout-max-response-len` | 2048 | 最大响应长度 |
| `--lr` | 5e-7 | 学习率 (VLM 建议较小) |
| `--eps-clip` | 0.2 | PPO clip 范围 |
| `--kl-loss-coef` | 0.01 | KL 散度损失系数 |
| `--entropy-coef` | 0.01 | 熵正则化系数 |

## 13. 监控指标

训练过程中需要关注的关键指标：

```python
# WandB 日志示例
{
    "rollout": 50,
    "train/loss": 0.023,
    "train/policy_loss": 0.018,
    "train/kl_loss": 0.003,
    "train/entropy": 2.1,
    "reward/mean": 0.72,
    "reward/std": 0.25,
    "reward/max": 1.0,
    "reward/min": 0.1,
    "advantage/mean": 0.0,
    "advantage/std": 1.0,
    "lr": 4.5e-7,
    "gpu_memory_used": 45.2  # GB
}
```

## 总结

一条 KIE 训练数据的完整生命周期：

1. **加载**: Parquet → Python dict
2. **采样**: 随机选入 batch，复制 4 份
3. **生成**: SGLang + InternVL 生成 JSON 响应
4. **评估**: KIE 奖励函数计算分数
5. **标准化**: GRPO 组内优势计算
6. **打包**: 多样本打包成长序列
7. **前向**: InternViT → MLP → Qwen3 LLM
8. **损失**: PPO clipped loss + KL + Entropy
9. **反向**: FSDP 分布式梯度计算
10. **更新**: Adam 优化器更新参数
11. **同步**: 新权重同步到 SGLang
12. **循环**: 进入下一轮 rollout
