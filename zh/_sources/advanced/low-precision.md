# 低精度训练

- [FP8 推理与 BF16 训练](#FP8-推理与-BF16-训练)
- [FP8 推理与 FP8 训练](#FP8-推理与-FP8-训练)
- [INT4 QAT 训练](#INT4-QAT-训练)

## FP8 推理与 BF16 训练

你可以通过在 `--hf-checkpoint` 中设置块缩放（blockwise）量化的 HuggingFace 权重来运行 FP8 推演。转换命令如下：

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir $BF16_MODEL \
    --save-dir $FP8_model \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

请确保转换后的权重目录中的 `config.json` 包含正确的 `quantization_config`，以便 slime 在权重更新期间自动使用 FP8 量化。

## FP8 推理与 FP8 训练

我们观察到，在训练和推理阶段同时使用 FP8，可以获得更高效的推理吞吐量，并降低训推不一致，从而使训练更稳定。更多细节请参考 [此博客](https://lmsys.org/blog/2025-11-25-fp8-rl/)。

### 快速开始

1. 使用上述 `tools/convert_hf_to_fp8.py` 将 HuggingFace 模型权重转换为 FP8 格式。
2. 对于训练任务，需要添加以下参数：
```bash
--fp8-format e4m3
--fp8-recipe blockwise
# --fp8-param-gather # [可选] 目前与 CPU Adam 优化器不兼容

```

同时，确保启用了环境变量 `NVTE_FP8_BLOCK_SCALING_FP32_SCALES`，目前我们会默认将这个参数设置为 `1`。

注意：目前只有 TransformerEngine 中的 `Linear` 和 `GroupLinear` 层使用 FP8 格式。`embedding` 和 `lm_head` 仍保持原始精度。如果未开启 `--fp8-param-gather`，TransformerEngine 中的权重将以 BF16 格式存储，仅在 `GEMM` 或 `GroupGEMM` 运算期间临时转换为 FP8。

3. 启动训练：

```bash
# Qwen3-4B Int4 training
bash scripts/low_precision/run-qwen3-4b-fp8.sh

# Qwen3-30B-A3B (2 nodes)
bash scripts/low_precision/run-qwen3-30b-a3b-fp8.sh
```

4. 使用保存的 ckpt：TransformerEngine 不会专门保存 FP8 量化后的权重；保存的 `torch_dist` 检查点仍为原始精度（通常是 BF16）。如果你想在 FP8 下进行评估，需要先将 `torch_dist` 转换为 HuggingFace 格式，然后再转换为 FP8 HuggingFace 格式。

### 原理简述

以下是 slime 中 FP8 训练目前的实现方式：

1. **初始化**：如果启用了 FP8 方案，相关层将在 FP8 上下文中构建。
2. **训练过程**：在训练期间，权重和激活值会在线量化为 `nvfp8` 格式，并在前向和反向传播中调用 `cuBLAS FP8 GEMM` 进行计算。
3. **权重更新**：在强化学习（RL）权重更新期间，Megatron 首先将 FP8 权重反量化为 BF16 格式，然后 slime 再将这些 BF16 权重重新量化为 FP8 并发送给 sglang。（这种“反量化+再量化”的操作虽然不够优雅，但为了框架兼容性，目前尚未修改接口。）
4. **保存检查点**：与权重更新类似，从训练引擎保存检查点时，也会反量化回 BF16 并以 `torch_dist` 格式保存。

### 待办事项 (TODO)

目前 FP8 功能尚不完全成熟，仍存在以下已知问题：

* FP8 权重存储（`--fp8-param-gather`）虽然能节省显存，但目前必须配合 TransformerEngine 的 `FusedAdam` 使用，这与 Megatron-LM 中的 CPU Adam 技术冲突。

## INT4 QAT 训练

本指南提供了 INT4 STE（直通估计器，Straight-Through Estimator）训练和 INT4 推理的示例。使用 INT4 推理可显著提升吞吐量，从而加速整个训练流水线（特别是在 rollout 生成阶段）。

### 快速开始

1. **将 HuggingFace 权重转换为 INT4**
首先，从 HuggingFace 下载 PTQ（训练后量化）校准数据集：
[wikitext-2-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1)
接着，使用 `tools/convert_hf_to_int4.py` 脚本进行转换。确保 `--hf-checkpoint` 指向的目录中 `config.json` 包含正确的 `quantization_config`。
```bash
python tools/convert_hf_to_int4.py \
  --input-dir /path/to/your/original/models \
  --output-dir /path/to/your/save/models \
  --data-dir /path/to/your/wikitext

```

**提示**：如果你只想运行 INT4 推演（Rollout），只需将 `--hf-checkpoint` 设置为转换后的 INT4 路径即可。
2. **启动 INT4 QAT 训练**
你需要配置特定的环境变量来设定量化参数。
**环境变量说明：**
* **`OPEN_TRAINING_INT4_FAKE_QAT_FLAG`**: 启用 INT4 训练的伪量化（Fake Quantization）操作。
* **`OPEN_TRAINING_INT4_GROUP_SIZE`**: 指定模型量化的块大小（Group Size）。
* `moonlight-16B-A3B`、`qwen3-30B-A3B` 和 `qwen3-235B-A22B-int4` 设置为 **128**。
* `kimi-k2-Thinking-int4` 设置为 **32**。

**配置示例：**
```json
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    ...
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"128\"
  }
}"
```

**启动命令：**
```bash
# Moonlight-16B-A3B Int4 training
bash scripts/low_precision/run-moonlight-16B-A3B-int4.sh

# Qwen3‑30B‑A3B Int4 training
bash scripts/low_precision/run-qwen3‑30B‑A3B-int4.sh

# Qwen3-235B-A22B Int4 training (8 nodes)
bash scripts/low_precision/run-qwen3-235B-A22B-int4.sh

# Kimi-k2-Thinking Int4 training (32 nodes)
bash scripts/low_precision/run-kimi-k2-Thinking-int4.sh
```

*对于多节点环境，请根据您的集群配置启动 Ray 服务。*
