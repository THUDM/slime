# slime：为 RL Scaling 设计的 SGLang-Native 后训练框架

> 本文由英文版翻译而来，首发于 [lmsys.org](https://lmsys.org/blog/2025-07-09-slime/)。

## 愿景

我们相信强化学习。我们相信强化学习是通往 AGI 的最后一块拼图。

所以我们相信：

  - 应该在每个领域上都尝试端到端强化学习，把所有任务都转变为 agent 环境。
  - 强化学习训练运行都应持续更久，每个模型都应扩展得更大。
  - 强化学习系统应与现有基础设施无缝集成，让我们能够专注于新的想法，而不是琐碎的杂活。

所以我们实现了 [slime](https://github.com/THUDM/slime)，它是一个专为后训练设计的框架，旨在实现以下特性：

  - **多功能性**——具有完全可定制的推理接口和灵活的训练设置（同地或解耦，同步或异步，强化学习或 SFT 冷启动）。
  - **高性能**——原生集成了 SGLang 进行推理，以及 Megatron-LM 进行训练。
  - **可维护性**——代码库轻量，并能从 Megatron 预训练平滑过渡到 SGLang 部署。

简而言之，它是一个为强化学习扩展而生的后训练框架。

以下是我们如何实现这一切的。

## 自定义才能自由

> 我们应该停止尝试用简单的方式来思考心智的内容，例如简单地思考空间、物体、多智能体或对称性。
>
> — *The Bitter Lesson*

强化学习社区中一个普遍的误解是，不同任务需要不同的框架：一个用于纯粹的数学，一个用于多轮工具调用，一个用于异步训练，一个用于智能体任务，等等。维护和分叉多个框架令人沮丧，这导致了浪费时间的 bug 修复挑选，甚至更糟的是，因为遗漏补丁而导致的训练崩溃。

事情并非总是如此：没有人会为了一个新的数据加载器而分叉 PyTorch。我们认为目前的混乱源于一种陷阱，即规定人们应该如何构建他们的应用。如果我们坚持为每种推理场景定义一个通用模板，我们最终只会创建一个只满足一小部分实际需求的强化学习框架。

slime 以不同的方式看待强化学习中的数据采样。我们在 slime 内部通过 [sgl-router](https://github.com/sgl-project/sglang/tree/main/sgl-router) 管理所有 SGLang 服务器，并为数据生成组件提供一个接口，**允许用户注入自定义逻辑并自由地与 SGLang 服务器交互**。这能释放他们的创造力。

使用 sgl-router，用户只需向一个单一端点发送 HTTP 请求。通过暴露这个端点，复杂的智能体环境可以直接通过一个与 OpenAI 兼容的 API 与 slime 交互——无需修改环境，并且训练与部署的一致性也得到了保留。

在训练方案方面，slime 使用 Ray 进行资源管理，通过一个简单的标志 (`--colocate`)，即可启用**同地**（相同 GPU）或**解耦**（不同 GPU）的设置。

凭借 Ray 通过 `.remote()` 实现的异步执行，slime 自然支持异步训练。改变同步行为就像移动 `ray.get` 操作一样简单。为了便于尝试不同的策略，我们没有将代码封装在训练器类中，而是简单地将训练循环暴露在入口文件 `train.py` 中。

## 为性能而生

**一个合格的强化学习框架必须既快，又持续地快。**

**快**意味着要利用最快的推理和训练框架。

与预训练不同，强化学习工作负载在训练过程中涉及大量的在线采样，这使得推理性能至关重要。因此，slime 专门集成了 SGLang，并刻意提供了 SGLang 原生体验。

那么，“SGLang 原生”意味着什么？这意味着你可以充分利用所有 SGLang 的优化——在 slime 内部使用 SGLang 就像单独使用它一样。为了实现这一点：

  - slime 在内部以**服务器模式**启动 SGLang 服务器。
  - slime 对所有 SGLang 参数实现了**无缝传递**（带有 `--sglang` 前缀），确保所有优化选项都可以启用。例如，你可以传递 `--sglang-enable-ep-moe`、`--sglang-enable-dp-attention` 和 `--sglang-enable-deepep-moe`，以实现强大的多节点 MoE 推理功能。
  - slime 提供了一个**仅限 SGLang 的调试模式** (`--debug-rollout-only`)，以便轻松进行性能调优。

通过这些，我们可以在 slime 内部重现 SGLang 的独立性能。甚至 slime 的基础镜像也是基于 `lmsysorg/sglang:dev` 构建的。

对于训练，slime 集成了久经考验的 Megatron-LM，旨在提供同样原生的预训练体验：

  - slime 也对所有 Megatron 参数实现了**无缝传递**。
  - slime 支持**所有 Megatron 并行策略**（TP, PP, EP, CP），并监控训练 MFU。
  - slime 提供了**仅限 Megatron 的调试模式** (`--debug-train-only`)，并支持存储采样数据以供重现。

Megatron 可能非常复杂，因此我们还提供了 ckpt 转换工具来简化其使用。

**持续地快**意味着要跟上不断发展的推理和训练框架。

如果你曾关注 [SGLang 的 PR 列表](https://github.com/sgl-project/sglang/pulls)，你会被其快速的演进所震惊。另一方面，Megatron 通常被深度定制，每个组织都维护着自己的分叉。slime 旨在跟上游 SGLang 的变化，并适应内部 Megatron 变体中的优化。这也是我们追求对 SGLang 和 Megatron 原生支持的另一个原因。参数传递使得升级毫不费力。

除了优化推理和训练框架，我们还处理了强化学习特有的工作负载。当 SGLang 需要修改以支持这些工作流时，我们与 SGLang 团队紧密合作，将补丁合并到上游——这样即使强化学习逻辑演变，slime 也能保持原生。例如：

**优化权重更新**：与推理任务不同，强化学习训练涉及频繁的模型权重更新。为了解决这个问题，我们在 SGLang 中引入了几项优化：

  - 在各种并行策略下对 MoE 模型进行参数更新（[\#6265](https://github.com/sgl-project/sglang/pull/6265)、[\#6308](https://github.com/sgl-project/sglang/pull/6308)、[\#6311](https://github.com/sgl-project/sglang/pull/6311)）。
  - 支持桶式参数更新以减少开销（[\#7292](https://github.com/sgl-project/sglang/pull/7292)）。

**用于动态采样的 `/abort_request`**：在需要过采样的强化学习算法中，例如 [DAPO](https://arxiv.org/abs/2503.14476)，即使已收集到足够的数据，某些请求可能仍会继续运行。我们与 [AReal](https://github.com/inclusionAI/AReaL) 团队合作，设计了一个新的端点：`/abort_request`。这个端点能够：

  - 立即终止正在进行的请求。
  - 重新获取部分生成的内容，从而实现部分推理。

这些功能在 [\#6698](https://github.com/sgl-project/sglang/pull/6698)、[\#6855](https://github.com/sgl-project/sglang/pull/6855)、[\#6184](https://github.com/sgl-project/sglang/pull/6184)、[\#5966](https://github.com/sgl-project/sglang/pull/5966) 中实现。

## 轻量且可扩展

slime 专注于可定制性和性能：

1.  提供了一个可定制的推理接口。
2.  使用 Ray 进行 GPU 管理和异步执行。
3.  集成 SGLang 用于推理，Megatron 用于训练。
4.  提供训练和推理之间的权重更新。

很简单，对吧？slime 将复杂性从框架转移到用户定义的管道和核心库（SGLang 和 Megatron），从而形成一个轻量、易于维护的代码库。

但它并不仅限于强化学习。

由于其模块化设计和强大的后端，slime 可以通过最少的额外代码自然地扩展到其他后训练工作流：

  - **SFT**：加载 Megatron 并使用 token 预测损失。
  - **Rejection Sampling**：使用 SGLang 进行过滤，然后使用 Megatron SFT。

*（请注意，SFT 功能目前处于实验阶段。）*

除此之外，slime 的原生集成**无缝连接了预训练到在线服务**。我们可以使用 Megatron 进行预训练，切换到 slime（它集成了 Megatron 和 SGLang）进行后训练，最后直接使用 SGLang 进行评估和部署。这消除了转换检查点格式和对齐框架之间精度的繁琐且易出错的步骤。

统一的管道将我们从繁琐的“胶水代码”中解放出来，让我们能够专注于真正重要的事情：更好的强化学习。太棒了！

## 发展蓝图

强化学习扩展的旅程才刚刚开始，slime 也在不断演进。在下一阶段，我们将专注于：

1.  与 SGLang 团队合作，探索大规模 MoE 模型的最佳强化学习训练策略。
2.  支持更广泛的后训练工作流，加强从预训练到生产的桥梁。
3.  添加对原生 PyTorch 训练后端的支持，以降低入门门槛。

我们希望 slime 能加速你的强化学习扩展之旅，并将你的创新想法变为现实。欢迎随时提出贡献和进行交流！

特别感谢 AMD GenAI - Foundation Model Team 在第一天就提供了 AMD 硬件支持。
