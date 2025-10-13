# 在 Megatron-LM 中快速支持新模型架构

Megatron-LM 框架虽然并行效率高，但在支持日新月异的新模型架构（如 Qwen3Next）时，其灵活性有所欠缺。若要原生支持这些模型的特殊结构（例如 Gated-Delta-Net），往往需要对 Megatron 的核心代码进行侵入性较大、开发周期较长的改造。

为了能快速跟进这些前沿模型，`slime` 提出了一种更敏捷的方案：**与其深度改造 Megatron，不如直接引入并封装模型官方的 HuggingFace 实现**，将其作为一个“黑盒模块”无缝嵌入到 Megatron 的并行训练流程中。

本文以 Qwen3Next 80B-A3B 为例，介绍这一实现思路。

## 实现原理与核心组件

Megatron 的模型实例化分为两步：首先根据配置生成“层规格”（`ModuleSpec`），再依据该规格实例化具体的 PyTorch 模块。

`slime` 正是利用这一机制，在**生成 Spec 的阶段“劫持”并替换掉 Megatron 的原生模块**，从而将外部实现（此处为 HuggingFace 模块）无缝嵌入。这一过程主要涉及三个核心组件的协同：

1.  **替换 Megatron 模块规格 (Spec)**
    这是整个方案的入口。我们通过一个自定义函数（例如 `get_qwen3_next_spec`）来修改标准的 `ModuleSpec`，用我们自己的封装层换掉 Megatron 的原生 Attention 层。
    * **具体操作**：获取标准的 Decoder Block Spec，将其 `self_attention` 字段指向我们的自定义模块，并按需开启 `qk_layernorm` 等模型特有配置。
    * **对应文件**: `slime_plugins/models/qwen3_next.py`

2.  **封装 HuggingFace 实现**
    上一步的 Spec 会指向一个封装层，例如 `HuggingfaceAttention`。它继承了 Megatron 的 `MegatronModule`，核心职责是作为桥梁，处理好并行策略所需的数据对齐（如序列并行），然后在内部直接调用从 HuggingFace 加载的原生 `Qwen3NextAttention` 模块。
    * **对应文件**: `slime_plugins/models/hf_attention.py`

3.  **对齐模型权重**
    模型结构跑通后，还需要确保权重能正确加载。我们借助 [mbridge](https://github.com/ISEEKYAN/mbridge) 库，通过 `Qwen3NextBridge` 建立了 HuggingFace Checkpoint 与 Megatron 参数之间的命名映射关系，实现双向互通。
    * **对应文件**: `slime_plugins/mbridge/qwen3_next.py`

通过这三层协同，我们成功地将一个 Megatron 原本不支持的复杂模型结构（以其 HuggingFace 实现为载体），运行在了 Megatron 的并行框架之上，并完整保留了模型并行、MoE 加速、流水线调度等全部关键能力。

## 当前限制

* 本方案暂不支持被替换模块（如此处的 Attention 层）自身的张量并行（TP）。
* **影响**：在大多数大规模 MoE 模型中，Attention 层的参数量占比较小，因此该限制对显存占用和训练吞吐的影响通常有限。
* **替代方案**：如果该模块的 TP 至关重要，则需要回归到侵入式修改 Megatron 的原生实现方案。
