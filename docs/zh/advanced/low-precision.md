# 低精度训练与 Rollout

slime 中的低精度能力主要用于让 rollout 更快、更省显存，同时保持训练侧数值稳定。对于大规模 MoE RL，推荐的生产路径是：

> **Megatron BF16 训练 + SGLang FP8 rollout/inference**

Megatron 侧保持 BF16/torch_dist 的可训练 checkpoint；SGLang 侧使用 FP8 Hugging Face checkpoint 做 rollout。权重更新时，slime 会根据 `--hf-checkpoint` 中的量化配置，把更新后的 BF16 权重量化后再发送给 SGLang。

## Feature Maturity

| 功能 | 状态 | 推荐用法 |
|---|---|---|
| BF16 training + FP8 rollout/inference | Stable | 大规模 MoE RL 的默认推荐路径。训练保持稳定，rollout 降低显存和带宽开销。 |
| SGLang rollout FP8 KV cache | Stable，取决于当前 SGLang 版本和 GPU stack 支持 | 通过 `--sglang-kv-cache-dtype fp8_e4m3` 提升 long-context 或 agentic rollout 的 KV cache 容量。 |
| INT4 rollout / INT4 QAT | Beta | 当 rollout 显存或吞吐压力很高，并且目标模型路径已经验证时使用。 |
| FP8 training + FP8 rollout | Experimental | 适合研究训推不一致和吞吐优化，但仍有 optimizer/checkpoint 相关限制。 |

## BF16 训练 + FP8 Rollout

这是 slime 当前最主要的生产路径。

你可以通过将 `--hf-checkpoint` 指向 blockwise quantized Hugging Face checkpoint 来开启 FP8 rollout。可以用如下命令从 BF16 checkpoint 转换：

```bash
python tools/convert_hf_to_fp8.py \
    --model-dir $BF16_MODEL \
    --save-dir $FP8_MODEL \
    --strategy block --block-size 128 128 \
    --max-workers 4
```

请确保转换后的 checkpoint 中 `config.json` 包含正确的 `quantization_config`。slime 会在权重更新时使用这个配置，因此训练侧可以保持 BF16，而 rollout 侧收到 FP8 权重。

示例：

```bash
# Megatron 训练 checkpoint 仍然是 BF16 / torch_dist。
--ref-load /path/to/model_torch_dist

# SGLang rollout checkpoint 使用 FP8 Hugging Face。
--hf-checkpoint /path/to/model-fp8-hf
```

## Rollout 使用 FP8 KV Cache

对于 long-context、multi-turn 或 agentic workload，KV cache 容量经常是瓶颈。由于 slime 通过 `--sglang-` 前缀透传 SGLang 参数，可以直接开启 FP8 KV cache：

```bash
--sglang-kv-cache-dtype fp8_e4m3
```

这是 rollout 侧配置，不会改变 Megatron 的训练精度。它可以提升 SGLang 的有效 KV cache 容量，从而支持更长 context 或更高并发；实际精度和性能表现取决于当前 SGLang 版本与 GPU stack。

## FP8 训练 + FP8 Rollout

slime 也支持 experimental 的 FP8 training 路径。我们观察到，在一些设置下同时使用 FP8 training 和 FP8 inference，可以提升推理吞吐并降低训推不一致。更多细节请参考 [这篇博客](https://lmsys.org/blog/2025-11-25-fp8-rl/)。

### 快速开始

1. 使用 `tools/convert_hf_to_fp8.py` 将 Hugging Face 模型权重转换为 FP8 格式。

2. 添加 FP8 训练参数：

```bash
--fp8-format e4m3
--fp8-recipe blockwise
# --fp8-param-gather # 可选；目前与 CPU Adam 不兼容
```

3. 确保设置 `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`。slime 默认会为 Ray actors 设置为 `1`。

4. 启动 FP8 training example：

```bash
# Qwen3-4B FP8 training
bash scripts/low_precision/run-qwen3-4b-fp8.sh

# Qwen3-30B-A3B FP8 training (2 nodes)
bash scripts/low_precision/run-qwen3-30b-a3b-fp8.sh
```

### 实现说明

1. 如果启用 FP8 recipe，TransformerEngine layers 会在 FP8 context 中构建。
2. 训练时，权重和 activation 会在线量化为 NVFP8 格式，并在 forward/backward GEMM 中使用 cuBLAS FP8 GEMM。
3. RL 权重更新时，Megatron 会先把 FP8 权重反量化为 BF16，然后 slime 再把 BF16 权重量化为 FP8 并发送给 SGLang。
4. 从 training engine 保存 checkpoint 时，会反量化回 BF16 并保存为 `torch_dist`。

目前只有 TransformerEngine 中的 `Linear` 和 `GroupLinear` 层使用 FP8。`embedding` 和 `lm_head` 保持原始精度。如果未开启 `--fp8-param-gather`，TransformerEngine 中的权重以 BF16 存储，仅在 `GEMM` 或 `GroupGEMM` 时转换为 FP8。

### 已知限制

`--fp8-param-gather` 可以节省显存，但目前需要 TransformerEngine `FusedAdam`，这与大规模 Megatron-LM RL 中常用的 CPU Adam offload 路径冲突。

## INT4 QAT 训练

INT4 STE（Straight-Through Estimator）训练和 INT4 inference 可以进一步降低 rollout 显存并提升吞吐。在目标模型和 reward setup 验证前，请把这条路径视作 beta。

### 快速开始

1. 将 Hugging Face 权重转换为 INT4：

```bash
python tools/convert_hf_to_int4_direct.py \
  --model-dir /path/to/your/original/models \
  --save-dir /path/to/your/save/models
```

如果只需要 INT4 rollout，把 `--hf-checkpoint` 指向转换后的 INT4 checkpoint 即可。

2. 开启 INT4 fake QAT：

```json
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"128\"
  }
}"
```

`OPEN_TRAINING_INT4_GROUP_SIZE` 通常设置为：

- `128`：`moonlight-16B-A3B`、`qwen3-30B-A3B`、`qwen3-235B-A22B-int4`；
- `32`：`kimi-k2-Thinking-int4`。

3. 启动 example：

```bash
# Moonlight-16B-A3B INT4 training
bash scripts/low_precision/run-moonlight-16B-A3B-int4.sh

# Qwen3-30B-A3B INT4 training
bash scripts/low_precision/run-qwen3-30B-A3B-int4.sh

# Qwen3-235B-A22B INT4 training (8 nodes)
bash scripts/low_precision/run-qwen3-235B-A22B-int4.sh

# Kimi-k2-Thinking INT4 training (32 nodes)
bash scripts/low_precision/run-kimi-k2-Thinking-int4.sh
```

多机环境请根据集群配置启动 Ray 服务。
