# slime

[English](./README.md)

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

**slime** 是为 RL scaling 设计的 LLM post‑training 框架，提供两大核心能力：

1. **高性能训练**：通过连接 Megatron 与 SGLang，支持各种模式的高效训练；
2. **灵活的数据生成**：通过自定义数据生成接口以及 server-based engine，实现任意训练数据生成流程。

slime 的设计目标，是让这两大能力彼此强化，同时避免把系统变成一组割裂的 trainer、rollout service 和 agent framework。Megatron training、SGLang rollout、custom data generation、reward computation、verifier feedback 和 environment interaction 都流经同一条 training / rollout / Data Buffer 路径。

这让 slime 成为最经受实战验证的开源 RL post-training 框架之一：它足够轻量、清晰、易扩展，同时也经过了 SOTA 级模型发布背后的完整训练闭环验证。

## 为什么这个设计重要

- **经过 frontier model 训练验证**：slime 是 [GLM-5.1](https://z.ai/blog/glm-5.1)、[GLM-5](https://z.ai/blog/glm-5)、[GLM-4.7](https://z.ai/blog/glm-4.7)、[GLM-4.6](https://z.ai/blog/glm-4.6)、[GLM-4.5](https://z.ai/blog/glm-4.5) 背后的 RL 训练框架。这验证的是完整 post-training loop，而不是孤立 example。
- **以正确性为先的基础设施**：RL bug 往往不会立刻报错。slime 保持显式的数据流，支持 rollout-only 和 train-only 分离调试，并把可复现性、容错、trace、profiling 和 CI 作为一等工程问题来维护。
- **从设计开始就是 native**：slime 直接透传 Megatron 参数，并通过 `--sglang-` 前缀暴露当前安装版本 SGLang 支持的参数。新的上游训练和 serving 优化可以直接使用，不需要在 slime 里再加一层抽象。
- **最大化的数据生成自由度**：math、code、search、tool、sandbox、verifier、environment、multi-agent system 以及 long-horizon agentic workflow 都可以作为 data generation 或 reward workflow 接入，而不需要 fork training kernel。
- **轻量且有明确取舍**：slime 深度优化 Megatron + SGLang 这条大规模 RL 生产路径。选择单一 rollout backend 让 slime 可以直接发挥 SGLang 的特有能力，而不是为了同时兼容多个推理框架，被迫抽象成 lowest-common-denominator 的公共能力子集。

## 生产验证

slime 已经覆盖发布级模型 post-training 所需要的完整工作流：大规模训练、高吞吐 rollout、权重同步、reward/verifier 数据、checkpoint、debugging 以及长时间运行稳定性。

除 GLM 系列之外，slime 还支持：

- Qwen 系列：Qwen3.6、Qwen3.5、Qwen3Next、Qwen3MoE、Qwen3、Qwen2.5；
- DeepSeek V3 系列：DeepSeek V3、V3.1、DeepSeek R1；
- Llama 3。

## 原生 Engine 透传与 SGLang 部署

slime 不只是一个能调用推理后端的框架。它尽量保留 Megatron 和 SGLang 上游 engine 原生的控制面，同时在其外层加入 RL 所需的数据流：

- 原生 SGLang 参数透传：当前安装版本 SGLang 支持的每个参数，都可以通过增加 `--sglang-` 前缀使用，例如把 `--mem-fraction-static` 写成 `--sglang-mem-fraction-static`；
- 原生 Megatron 参数透传：slime 直接读取 Megatron 参数，因此 Megatron 侧的并行、优化器、checkpoint 和模型配置不需要 wrapper code 也能继续使用；
- [SGLang Config](docs/zh/advanced/sglang-config.md) 作为可选 YAML 扩展，用于 topology-specific control，例如为 prefill/decode/EPD-style 部署设置独立参数、配置 heterogeneous server group、multi-model serving 和 per-group SGLang override；
- 面向 multi-turn 和 agentic workload 的 [PD Disaggregation](docs/zh/advanced/pd-disaggregation.md)，用于处理 prefill/decode 资源需求不同的问题；
- 面向 multi-turn agent 的 session affinity 等 router policy；
- 面向 training/inference disaggregation 和大模型高效更新的 [Delta Weight Sync](docs/zh/advanced/delta-weight-sync.md)；
- [External Rollout Engines](docs/zh/advanced/external-rollout-engines.md)，用于 serving 由训练任务外部管理的部署形态；SGLang serving 侧可以使用独立环境，disk transport 下还可以使用不同型号或不同厂家的 GPU，并通过共享文件系统做 full checkpoint update from disk 或 delta update。

这种透传设计让 slime 从一开始就是 native 的。大多数上游 engine 的优化会随着 engine 升级自然可用，而 slime 可以把主要复杂度集中在 RL loop、dataflow、synchronization 和 correctness check 上。

选择 SGLang 作为单一 rollout backend 也是有意为之。多 backend 框架往往需要在多个 inference engine 的公共能力子集上做抽象，结果会遮住每个 backend 最强的特性。slime 则深度优化 SGLang，让 RL workload 可以直接使用 SGLang-specific 的 serving、routing、caching、disaggregation 和 weight-sync 能力。

## 正确性、稳定性与 CI

slime 被当作 RL 基础设施来开发，因为“脚本能跑起来”远远不够。项目维护 CPU unit test、customization hook contract test，以及 GPU end-to-end test，覆盖 dense 和 MoE 模型、Megatron training path、SGLang deployment config、checkpoint、数值精度、async rollout、OPD、PPO-style workflow，以及 debug rollout-then-train replay。

相关工程文档：

- [CI](docs/zh/developer_guide/ci.md)
- [Debugging](docs/zh/developer_guide/debug.md)
- [Reproducibility](docs/zh/advanced/reproducibility.md)
- [Fault Tolerance](docs/zh/advanced/fault-tolerance.md)
- [Trace Viewer](docs/zh/developer_guide/trace.md)
- [Profiling](docs/zh/developer_guide/profiling.md)

## 博文

- 我们的愿景：[slime：为 RL Scaling 设计的 SGLang-Native 后训练框架](https://thudm.github.io/slime/zh/blogs/introducing_slime.html)
- 关于纯异步 agentic 训练的一些想法：[Agent-Oriented Design: An Asynchronous and Decoupled Framework for Agentic RL](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL-2278e692d081802cbdd5d37cef76a547)
- v0.1.0 日志：[slime v0.1.0: 重新定义高性能 RL 训练框架](https://zhuanlan.zhihu.com/p/1945237948166547268)


## 目录

- [为什么这个设计重要](#为什么这个设计重要)
- [生产验证](#生产验证)
- [原生 Engine 透传与 SGLang 部署](#原生-engine-透传与-sglang-部署)
- [正确性、稳定性与 CI](#正确性稳定性与-ci)
- [架构总览](#架构总览)
- [快速开始](#快速开始)
- [基于 slime 构建的生态](#基于-slime-构建的生态)
- [参数说明](#参数说明)
- [开发指南](#开发指南)
- [常见 Q&A 与致谢](#常见-qa-与致谢)

## 架构总览

![arch](./imgs/arch.png)

**模块说明**：

- **training (Megatron)**：负责主训练流程，从 Data Buffer 读取数据，训练完后将参数同步至 rollout 模块；
- **rollout (SGLang + router)**：生成新数据（含 reward/verifier），存储至 Data Buffer；通过 custom generate 可以在其上叠加 multi-turn loop、tool call、environment/sandbox 交互以及 verifier-based reward；
- **data buffer**：桥梁模块，管理 prompt 初始化、自定义数据与 rollout 生成方法（包括以同一套接口产出 sample 的 agentic workflow）。

## 快速开始

有关环境配置、数据准备、训练启动和关键代码分析的完整快速开始指南，请参考：

- [快速开始指南](./docs/zh/get_started/quick_start.md)

我们还提供了一些未在快速开始中覆盖的使用示例，请查看 [examples](examples/)。

### Agentic RL 示例

下面这些 example 通过 customization 接口接入标准的 rollout / Data Buffer 闭环，而不是独立的 framework：

- [`examples/multi_agent`](examples/multi_agent/README.md)：通过自定义 `--rollout-function-path` 实现多 agent 的 rollout。
- [`examples/search-r1`](examples/search-r1/)：通过 `--custom-generate-function-path` 实现 search/RAG 风格的多轮生成。
- [`examples/fully_async`](examples/fully_async/README.md)：fully-async rollout，适合不同样本生成耗时差异较大的 long-tail agentic 场景。
- [`examples/coding_agent_rl`](examples/coding_agent_rl/README.md)：端到端 SWE coding-agent RL，包含 sandboxed tool use、test-based reward，以及通过 `--custom-generate-function-path` 导出的 token-correct trajectory segments。

如何为某种 agentic workflow 选择合适的接口，请参考 [自定义指南](docs/zh/get_started/customization.md)。

## 基于 slime 构建的生态

这些项目不只是 demo。它们是把 slime 作为可复用 RL substrate 的独立系统，覆盖生产级 post-training、agentic RL、domain RL 和 rollout-system research。

### 🌈 Relax: Asynchronous RL Engine for Omni-Modal Agentic Training

[**Relax**](https://github.com/redai-infra/Relax) (Reinforcement Engine Leveraging Agentic X-modality) 是 RedAI Infra 团队开源的 omni-modal agentic RL framework，构建在结合 Ray、Megatron-LM 和 SGLang 的 slime infrastructure stack 之上。Relax 采用 Ray Serve 上的 service-oriented architecture，以 Megatron-LM 和 SGLang 作为 training/inference backend。它使用 [TransferQueue](https://github.com/Ascend/TransferQueue) 将 Actor、Rollout、ActorFwd、Reference 和 Advantage computation 完全解耦到独立 GPU 集群，并引入 **DCS (Distributed Checkpoint Service)**，通过 NCCL-broadcast weight-sync engine 将更新后的 Actor 权重异步 stream 到 Rollout/ActorFwd/Reference，并与下一步训练重叠，从而在可配置 staleness 下实现 fully-async training。Relax 支持 text、vision、audio（包括 Qwen3-Omni）以及 agentic multi-turn rollout 的端到端 RL。

### 🦞 OpenClaw-RL: Train a Personalized Clawbot Simply by Talking to It

[**OpenClaw-RL**](https://github.com/Gen-Verse/OpenClaw-RL) 是面向 personalized OpenClaw agent 的 RL server。它托管 OpenClaw model，并从跨部署的历史对话中持续改进模型，同时 slime 的 asynchronous RL infrastructure 避免训练过程干扰 API serving。它支持两种自动优化方法：基于后续状态推断 binary feedback 的 GRPO，以及从后续反馈中提取 hindsight hint 的 on-policy distillation。

### ⚛️ P1: Mastering Physics Olympiads with Reinforcement Learning

[**P1**](https://prime-rl.github.io/P1/) 是一系列完全通过 reinforcement learning 训练的开源物理推理模型。P1 使用 slime 作为 RL post-training framework，并提出 multi-stage RL training algorithm，通过 adaptive learnability adjustment 和 stabilization mechanism 逐步增强推理能力。在这一训练范式下，P1 在开源物理推理上取得了突破性表现。

### 📈RLVE: Scaling LM RL with Adaptive Verifiable Environments

[**RLVE**](https://github.com/Zhiyuan-Zeng/RLVE) 提出使用 verifiable environments 来扩展语言模型 RL：环境以程序化方式生成问题，并提供可算法验证的 reward。通过在 400 个 verifiable environment 上联合训练，RLVE 能让每个 environment 随训练进展动态适配 problem difficulty distribution，使其匹配当前 policy model 的能力。

### ⚡ TritonForge: Agentic RL Training Framework for Kernel Generation

[**TritonForge**](https://github.com/RLsys-Foundation/TritonForge) 使用 slime 的 SFT 和 RL 能力训练能够自动生成优化 GPU kernel 的 LLM。通过 supervised fine-tuning 加 reinforcement learning with multi-turn compilation feedback 的两阶段训练，TritonForge 在将 PyTorch operation 转换为高性能 Triton kernel 上取得了显著结果。

### 🚀 APRIL: Accelerating RL Training with Active Partial Rollouts

[**APRIL**](https://github.com/RLsys-Foundation/APRIL) 提出一种可以无缝集成到 slime 的 system-level optimization，用于加速 RL 训练中的 rollout generation 阶段。它通过智能 over-provision request 并主动管理 partial completion，缓解 rollout 生成中常见的 long-tail bottleneck，而这一阶段通常会消耗 RL 训练 90% 以上的时间。

### 🏟️ qqr: Scaling Open-Ended Agents with ArenaRL & MCP

[**qqr**](https://github.com/Alibaba-NLP/qqr) (a.k.a. hilichurl) 是一个用于演化 open-ended agent 的 slime lightweight extension。它实现 **ArenaRL** algorithm，通过 tournament-based relative ranking（例如 Seeded Single-Elimination、Round-Robin）缓解 discriminative collapse，并无缝集成 **Model Context Protocol (MCP)**。qqr 利用 slime 的高吞吐训练能力，在标准化、解耦的 tool environment 中实现可扩展的分布式 agent evolution。

这些项目共同体现了 slime 的核心思路：一个高性能 RL kernel 可以同时支撑 frontier model post-training、online agent optimization、verifiable environment、omni-modal rollout、kernel-generation agent 和 rollout-system research，而不需要改变核心 training loop。

## 参数说明

参数分为三类：

1. **Megatron 参数**：slime 会直接读取 Megatron 参数，可以通过传入如 `--tensor-model-parallel-size 2` 的方式配置 Megatron；
2. **SGLang 参数**：支持当前环境中安装版本 SGLang 的所有参数，这些参数需要以 `--sglang-` 起始，例如 `--mem-fraction-static` 需要通过 `--sglang-mem-fraction-static` 传入。
3. **slime 自身的参数**：请见：[slime/utils/arguments.py](slime/utils/arguments.py)

完整使用说明请查阅 [使用文档](docs/zh/get_started/usage.md)。

## 开发指南

- **欢迎贡献！** 若有功能建议、性能调优或使用体验反馈，欢迎提交 Issue / PR 😊

- 使用 [pre-commit](https://pre-commit.com/) 保证提交代码风格：

  ```bash
  apt install pre-commit -y
  pre-commit install

  # 运行 pre-commit 保证代码风格
  pre-commit run --all-files --show-diff-on-failure --color=always
  ```

- 调试技巧请参考 [debug 指南](docs/zh/developer_guide/debug.md)

## 常见 Q&A 与致谢

- 常见问题请见 [Q&A](docs/zh/get_started/qa.md)
- 特别感谢以下项目 & 社区：SGLang、Megatron‑LM、mbridge、OpenRLHF、veRL、Pai-Megatron-Patch 等。

- 引用 slime 请使用：
```bibtex
@misc{slime_github,
  author       = {Zilin Zhu and Chengxing Xie and Xin Lv and slime Contributors},
  title        = {slime: An LLM post-training framework for RL Scaling},
  year         = {2025},
  howpublished = {\url{https://github.com/THUDM/slime}},
  note         = {GitHub repository. Corresponding author: Xin Lv},
  urldate      = {2025-06-19}
}
```
