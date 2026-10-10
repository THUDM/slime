# 在 Megatron-LM 中快速支持新模型架构

slime 可以通过替换部分层的规格（`ModuleSpec`）来扩展 Megatron 模型。Qwen3-Next 80B-A3B 的集成使用这一方式实现 Gated DeltaNet（GDN）层，同时保留 Megatron 的 full-attention 层、MoE 模块和流水线调度。

## 实现原理与核心组件

Qwen3-Next 的集成包含三个部分：

1. **选择需要替换的层。** `get_qwen3_next_spec` 从 Megatron 的 decoder block 规格开始，只替换 Hugging Face 配置中标记为 `linear_attention` 的层，并根据流水线阶段的层偏移选择对应的配置。参见 [qwen3_next.py](https://github.com/THUDM/slime/blob/main/slime_plugins/models/qwen3_next.py)。
2. **在 Megatron 中执行 GDN。** GDN 实现沿用 Hugging Face 模型的参数布局，使用 FLA 或可选的 FlashQLA 后端执行递归注意力内核。`HuggingfaceAttention` 先收集序列并行和上下文并行的输入分片，在各 rank 上执行同一计算，再返回本地输出分片。参见 [hf_attention.py](https://github.com/THUDM/slime/blob/main/slime_plugins/models/hf_attention.py)。
3. **加载模型权重。** checkpoint loader 将 Hugging Face 的参数名称和 tensor 布局映射到自定义模块及保留的 Megatron 层。参见 [Qwen3-Next 权重加载器](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/hf_to_megatron/qwen3_next.py)。

模型选择和启动配置见[实验 tutorial](../get_started/experiment-guide.md)。现有架构 launcher 保留在 `scripts/`。

## 当前限制

自定义 GDN 模块可以运行在使用 TP 和 CP 的任务中，但参数和收集输入后的计算会在这些 rank 上复制。其投影层尚未按 TP 切分，因此增加 TP 不会像原生张量并行层那样降低该模块的参数显存和计算量。

当前规格支持流水线阶段的层偏移，但不支持自定义 `pipeline_model_parallel_layout`。若要消除复制计算的成本，需要实现原生的并行 GDN 模块。
