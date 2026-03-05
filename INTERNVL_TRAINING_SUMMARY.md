# InternVL 训练问题解决总结

## 问题背景

使用 SLIME 框架训练 InternVL3.5-4B 模型进行 KIE 任务时，遇到模型生成乱码、图像未被处理的问题。

## 根本原因

### 1. 模型格式不匹配

**问题**: HF 格式的 InternVL 模型（`InternVLForConditionalGeneration`）不被 SGLang 原生支持。

**表现**:
- SGLang 回退到 Transformers 后端
- Transformers 的 `generate()` 方法不正确处理图像
- 导致生成乱码，reward 为 0.0

**解决方案**: 使用非 HF 格式的 InternVL 模型（`InternVLChatModel`），SGLang 原生支持。

### 2. 图像占位符不匹配

**问题**: 不同格式的 InternVL 使用不同的图像占位符。

| 格式 | 占位符 | Processor |
|------|--------|-----------|
| HF 格式 | `<image>` | `InternVLProcessor` |
| 非 HF 格式 | `<IMG_CONTEXT>` | `InternVLProcessor` |

**表现**:
```
ValueError: Number of image placeholders in the prompt does not match the number of images.
```

**解决方案**: 在 rollout 时自动检测并替换占位符。

### 3. 数据格式处理问题

**问题**: SLIME 的数据处理将字符串 prompt 转换为结构化格式，导致 `<image>` 占位符丢失。

**原始数据**:
```python
{"problem": "<image>\n问题文本", "images": ["/path/to/image.jpg"]}
```

**错误处理后**:
```python
# <image> 被移除，变成结构化格式
[{"role": "user", "content": [{"type": "image", "image": "..."}, {"type": "text", "text": "问题文本"}]}]
```

**解决方案**: 为 InternVL 添加特殊处理，保持字符串格式的 prompt，单独提取图像。

### 4. 数据类型不匹配

**问题**: Parquet 读取的 `images` 字段是 `numpy.ndarray`，但 `InternVLProcessor` 需要 Python `list`。

**解决方案**: 在 `build_processor_kwargs` 中自动转换 numpy array 为 list。

### 5. img_context_token_id 缺失

**问题**: 非 HF InternVL 模型的 `forward()` 方法需要 `img_context_token_id` 来识别图像占位符。

**解决方案**: 在 Actor 初始化时设置：
```python
img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
model.img_context_token_id = img_context_token_id
```

## 解决方案架构

### 方案对比

#### 方案 1: HF 格式 + 转换（已废弃）
```
训练: HF 格式 → 保存 HF checkpoint → 转换 → SGLang 格式
推理: SGLang 格式
```

**缺点**:
- 需要维护转换脚本
- 权重命名映射复杂
- QKV 权重需要拼接
- 容易出错

#### 方案 2: 非 HF 格式（当前方案）✅
```
训练: 非 HF 格式 → 保存非 HF checkpoint
推理: 非 HF 格式（同一份权重）
```

**优点**:
- 训练和推理使用相同格式
- 无需转换
- SGLang 原生支持
- 简单可靠

### 代码修改

#### 1. Actor 初始化 (`slime/backends/fsdp_utils/actor.py`)

```python
# 设置 img_context_token_id
if hasattr(self.hf_config, "llm_config"):  # 非 HF InternVL
    img_context_token = "<IMG_CONTEXT>"
    img_context_token_id = self.tokenizer.convert_tokens_to_ids(img_context_token)
    # 处理 FSDP wrapping
    if hasattr(self.model, "_fsdp_wrapped_module"):
        self.model._fsdp_wrapped_module.img_context_token_id = img_context_token_id
    elif hasattr(self.model, "module"):
        self.model.module.img_context_token_id = img_context_token_id
    else:
        self.model.img_context_token_id = img_context_token_id
```

#### 2. Processor 加载 (`slime/utils/processing_utils.py`)

```python
def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        proc = None

    # 如果加载失败，检查是否是非 HF InternVL
    if proc is None:
        config = AutoConfig.from_pretrained(name_or_path, **kwargs)
        if hasattr(config, "llm_config") or getattr(config, "model_type", "") == "internvl_chat":
            tokenizer = load_tokenizer(name_or_path, **kwargs)
            proc = InternVLProcessorWrapper(tokenizer)

    return proc
```

#### 3. 数据加载 (`slime/utils/data.py`)

```python
# InternVL 特殊处理：保持字符串 prompt，单独提取图像
if multimodal_keys and processor and is_internvl_model(processor):
    prompt = data.get(prompt_key)  # 保持 <image> 占位符

    # 从 images 字段加载 PIL Image
    image_paths = data.get("images")
    if isinstance(image_paths, np.ndarray):
        image_paths = image_paths.tolist()
    multimodal_inputs = {"images": [load_image(p) for p in image_paths]}

    # 应用 chat template
    if apply_chat_template:
        messages = [{"role": "user", "content": prompt}]
        output_prompt = tokenizer.apply_chat_template(messages, ...)
```

#### 4. Rollout 处理 (`slime/rollout/sglang_rollout.py`)

```python
# 替换图像占位符
prompt_text = sample.prompt
if hasattr(state.processor, 'image_token'):
    if '<image>' in prompt_text and state.processor.image_token != '<image>':
        prompt_text = prompt_text.replace('<image>', state.processor.image_token)

processor_output = state.processor(text=prompt_text, **processor_kwargs)
```

#### 5. 数据类型转换 (`slime/utils/processing_utils.py`)

```python
def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:
    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # 转换 numpy array 为 list
    import numpy as np
    for key in ("images", "videos"):
        if key in result and isinstance(result[key], np.ndarray):
            result[key] = result[key].tolist()

    return result
```

## 数据流详解

### 完整流程

```
1. 数据准备
   原始 JSONL → convert_kie_data.py → Parquet
   {
     "problem": "<image>\n问题",
     "answer": "答案",
     "images": ["/path/to/image.jpg"]
   }

2. 数据加载 (Dataset)
   Parquet → Dataset.__init__
   ├─ 读取 problem (保持 <image>)
   ├─ 加载 PIL Image
   ├─ 应用 chat template
   └─ Sample(
        prompt="<|im_start|>user\n<image>\n问题<|im_end|>...",
        multimodal_inputs={"images": [PIL.Image]}
      )

3. Rollout (SGLang)
   Sample → sglang_rollout.generate()
   ├─ 替换: <image> → <IMG_CONTEXT>
   ├─ processor(text, images=[PIL.Image])
   │   ├─ 图像处理 → pixel_values
   │   └─ 扩展 <IMG_CONTEXT> → 256 tokens
   ├─ 保存 multimodal_train_inputs
   └─ 发送到 SGLang (text + base64 images)

4. 训练 (FSDP)
   Sample → Actor.forward()
   ├─ input_ids: tokenized text
   ├─ pixel_values: 图像特征
   ├─ image_flags: 标记有图像的样本
   └─ img_context_token_id: 识别占位符

5. 保存
   Checkpoint → 非 HF 格式
   ├─ config.json (model_type: internvl_chat)
   ├─ modeling_internvl_chat.py
   └─ model-*.safetensors

6. 推理
   Checkpoint → SGLang (直接加载，无需转换)
```

## 关键技术点

### 1. 占位符自动转换

```python
# 数据中统一使用 <image>
data = {"problem": "<image>\n问题"}

# Rollout 时根据模型自动转换
if processor.image_token == "<IMG_CONTEXT>":
    prompt = prompt.replace("<image>", "<IMG_CONTEXT>")
```

### 2. FSDP Wrapping 处理

```python
# 需要处理 FSDP 的多层包装
if hasattr(model, "_fsdp_wrapped_module"):
    model._fsdp_wrapped_module.attr = value
elif hasattr(model, "module"):
    model.module.attr = value
else:
    model.attr = value
```

### 3. Processor 类型检测

```python
def is_internvl_model(processor):
    return (
        hasattr(processor, "image_processor") and
        "InternVL" in type(processor.image_processor).__name__
    )
```

## 测试验证

### 1. 数据加载测试

```python
from slime.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

dataset = Dataset(
    path="data/kie_train.parquet",
    tokenizer=tokenizer,
    processor=processor,
    max_length=2048,
    prompt_key="problem",
    multimodal_keys={"image": "images"},
    apply_chat_template=True,
)

sample = dataset[0]
print("Prompt:", sample.prompt[:200])
print("Images:", len(sample.multimodal_inputs["images"]))
```

### 2. Processor 测试

```python
from transformers import AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
img = Image.open("/path/to/image.jpg")

text = "<|im_start|>user\n<IMG_CONTEXT>\n问题<|im_end|>\n<|im_start|>assistant\n"
output = processor(text=text, images=[img], return_tensors="pt")

print("input_ids shape:", output["input_ids"].shape)
print("pixel_values shape:", output["pixel_values"].shape)
```

### 3. 训练测试

```bash
# 启动训练，观察日志
bash scripts/train_internvl_kie.sh

# 检查是否有图像处理日志
# [multimodal] Replaced <image> with <IMG_CONTEXT>
# [multimodal] Sending 1 images to SGLang
```

## 性能优化

### 1. 图像预处理缓存

当前每次 rollout 都重新处理图像，可以考虑缓存：

```python
# 在 Dataset 中预处理并缓存
processor_output = processor(text=prompt, images=images, return_tensors="pt")
sample.multimodal_train_inputs = {
    "pixel_values": processor_output["pixel_values"],
    "image_flags": torch.ones(1, dtype=torch.long)
}
```

### 2. 批量处理

当前是单样本处理，可以改为批量：

```python
# 批量处理多个样本
texts = [s.prompt for s in samples]
images = [s.multimodal_inputs["images"] for s in samples]
outputs = processor(text=texts, images=images, return_tensors="pt")
```

## 未来改进

### 1. 支持更多模型

- Qwen-VL
- LLaVA
- CogVLM

### 2. 支持视频

```python
multimodal_keys = {
    "image": "images",
    "video": "videos"
}
```

### 3. 动态图像数量

当前假设每个样本 1 张图，可以支持多图：

```python
# 数据格式
{
    "problem": "<image>\n图1描述\n<image>\n图2描述",
    "images": ["/path/to/img1.jpg", "/path/to/img2.jpg"]
}
```

## 参考资料

- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [SGLang Documentation](https://sgl-project.github.io/)
- [SLIME Framework](https://github.com/samaritan1998/slime)

## 总结

通过使用非 HF 格式的 InternVL 模型，我们实现了：

1. ✅ 训练和推理使用相同权重格式
2. ✅ 无需格式转换
3. ✅ 自动处理图像占位符差异
4. ✅ 正确处理多模态数据
5. ✅ 支持 FSDP 分布式训练

关键是理解不同格式模型的差异，并在数据流的各个环节做好适配。
