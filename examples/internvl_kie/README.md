# InternVL3.5-4B KIE Training

使用 SLIME 框架训练 InternVL3.5-4B 模型进行关键信息提取（Key Information Extraction, KIE）任务。

## 模型格式

本示例使用**非 HF 格式**的 InternVL 模型，训练和推理使用相同的权重格式，无需转换。

### 非 HF 格式 vs HF 格式

| 特性 | 非 HF 格式 | HF 格式 |
|------|-----------|---------|
| `architectures` | `InternVLChatModel` | `InternVLForConditionalGeneration` |
| `model_type` | `internvl_chat` | `internvl` |
| 配置结构 | `llm_config` + `vision_config` | `text_config` + `vision_config` |
| `auto_map` | 有（指向本地 .py 文件） | 无 |
| SGLang 支持 | 原生支持 | 需要转换 |
| 图像占位符 | `<IMG_CONTEXT>` | `<image>` |

## 数据格式

训练数据应为 Parquet 格式，包含以下字段：

```python
{
    "problem": "<image>\nFind out the vendor's name...",  # 包含 <image> 占位符的问题
    "answer": '{"vendor": "...", "total": "..."}',        # JSON 格式的答案
    "images": ["/path/to/image.jpg"],                     # 图像路径列表
}
```

## 数据流

### 1. 数据加载（Dataset）

```
原始数据 (Parquet)
  ↓
Dataset.__init__
  ├─ 读取 problem 字段（保持 <image> 占位符）
  ├─ 从 images 字段加载 PIL Image 对象
  ├─ 应用 chat template: <image>\n问题 → <|im_start|>user\n<image>\n问题<|im_end|>...
  └─ 创建 Sample(prompt=formatted_text, multimodal_inputs={"images": [PIL.Image, ...]})
```

### 2. Rollout 生成（SGLang）

```
Sample
  ↓
sglang_rollout.generate()
  ├─ 替换占位符: <image> → <IMG_CONTEXT>（processor 期望的格式）
  ├─ 调用 processor(text=prompt, images=[PIL.Image, ...])
  │   ├─ 处理图像 → pixel_values
  │   └─ 扩展 <IMG_CONTEXT> → 256 个 image tokens
  ├─ 保存 multimodal_train_inputs (pixel_values, image_flags)
  └─ 发送到 SGLang: text + image_data (base64)
```

### 3. 训练（FSDP）

```
Sample (with multimodal_train_inputs)
  ↓
Actor.forward()
  ├─ input_ids: tokenized prompt + response
  ├─ pixel_values: 图像特征
  ├─ image_flags: 标记哪些样本有图像
  └─ img_context_token_id: 用于识别图像占位符
```

## 快速开始

### 1. 准备数据

```bash
python scripts/convert_kie_data.py \
    -i /path/to/data1.jsonl /path/to/data2.jsonl \
    -o data/kie_train.parquet \
    -f parquet
```

### 2. 配置模型路径

编辑 `examples/internvl_kie/train_internvl_kie.py`：

```python
MODEL_PATH = "/path/to/non-hf-internvl-model"  # 非 HF 格式模型
TRAIN_DATA = "data/kie_train.parquet"
```

### 3. 启动训练

```bash
bash scripts/train_internvl_kie.sh
```

## 关键配置

### 多模态配置

```python
multimodal_args = '--multimodal-keys \'{"image": "images"}\' '
```

- `image`: 占位符类型
- `images`: 数据中的字段名

### 图像占位符处理

SLIME 自动处理占位符差异：
- 数据中使用 `<image>`（通用格式）
- Rollout 时自动转换为 `<IMG_CONTEXT>`（模型期望格式）

### 奖励函数

```python
reward_args = "--custom-rm-path slime.rollout.rm_hub.kie_reward.kie_reward"
```

KIE 专用奖励函数，基于 JSON 字段匹配计算奖励。

## 训练输出

```
outputs/internvl_kie/
├── iter_0000010/
│   └── model/              # 非 HF 格式 checkpoint
│       ├── config.json
│       ├── modeling_internvl_chat.py
│       ├── model-*.safetensors
│       └── ...
├── iter_0000020/
└── ...
```

保存的 checkpoint 可直接用于 SGLang 推理，无需转换。

## 推理

使用 SGLang 进行推理：

```bash
python -m sglang.launch_server \
    --model-path outputs/internvl_kie/iter_0000100/model \
    --trust-remote-code \
    --port 30000
```

## 常见问题

### Q: 为什么使用非 HF 格式？

A: 非 HF 格式是 SGLang 原生支持的格式，训练和推理使用相同权重，避免格式转换带来的问题。

### Q: 如何从 HF 格式转换？

A: 使用官方转换脚本或直接下载非 HF 格式模型。

### Q: 图像占位符不匹配怎么办？

A: SLIME 会自动处理。数据中使用 `<image>`，系统会根据模型自动转换。

## 技术细节

### img_context_token_id

非 HF InternVL 模型需要设置 `img_context_token_id` 来识别图像占位符：

```python
# actor.py
img_context_token = "<IMG_CONTEXT>"
img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)
model.img_context_token_id = img_context_token_id
```

### Processor 选择

- 非 HF InternVL: 使用 HF 的 `InternVLProcessor`
- 其他模型: 使用 `InternVLProcessorWrapper`

系统会自动检测模型类型并选择合适的 processor。
