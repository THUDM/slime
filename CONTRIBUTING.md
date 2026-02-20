# Contributing to slime

[中文版](#开源协作范围说明)

Thank you for your interest in contributing to slime! We deeply appreciate every contribution from the community. To keep the project healthy and sustainable, please read this document carefully before submitting issues or pull requests.

## Collaboration Scope

slime is the RL training infrastructure behind [GLM-4.5 through GLM-5](https://z.ai) and a large number of internal experiments at Z.ai. We open-sourced slime because we believe the training scenarios used internally cover the majority of cutting-edge RL algorithm requirements, and we hope to provide the community with a correct and efficient large-scale RL training infrastructure.

Our goal for open-source collaboration is focused on **bug fixes** and **general-purpose large-scale RL optimizations**. We have had several successful collaborations with the community in this area, including:

- Speculative decoding in RL ([docs](https://thudm.github.io/slime/en/advanced/speculative_decoding.html))
- Low-precision training: fp8 rollout + bf16/fp8 training, int4 rollout + int4 QAT training ([docs](https://thudm.github.io/slime/en/advanced/low_precision_training.html))
- Deterministic training ([docs](https://thudm.github.io/slime/en/advanced/reproducibility.html))

### What We Welcome

| Category | Examples |
|----------|----------|
| **Bug reports** | Crashes, incorrect results, documentation errors |
| **Bug fixes** | PRs that fix existing issues with tests or clear reproduction |
| **General RL optimizations** | Performance improvements with clear benchmarks that can be verified through CI or standard training runs |

### What's Currently Outside Our Scope

| Category | Reason |
|----------|--------|
| **Large-scale code refactoring** | This would add considerable overhead to syncing between the internal and open-source versions, particularly in coordinating with internal algorithm teams. |
| **Design / abstraction proposals** | e.g., universal data standards, eval standards, tool base classes. Standard-setting involves non-technical factors; slime intentionally avoids such content to keep things flexible for both the community and internal teams. |
| **Features that cannot be clearly verified** | Correctness is critically important for a training framework. If a feature cannot be verified through CI or routine internal training, it becomes difficult for us to ensure timely fixes, which could affect the project's long-term reliability. |
| **Features independent of the RL framework** | e.g., full algorithm reproduction pipelines. While these lower the barrier to entry, they are difficult to include in routine verification. slime aims to be lightweight — more like Flask than Django. We recommend building such pipelines in separate repositories; we are happy to reference them in the README. |
| **Major modifications to Megatron** | We do not plan to maintain a Megatron fork through slime. The goal is to switch Megatron versions relatively painlessly; Megatron performance optimization and feature completion are not primary objectives. |

### Why This Policy?

slime's design and development roadmap must first align with Z.ai's internal requirements — the RL infrastructure design is tightly coupled with post-training plans, and publishing the full post-training R&D roadmap is not within the open-source scope of slime. This focused scope is what we believe is necessary to maintain slime as a long-term, trustworthy project.

We understand this may slow down slime's pace of feature development. We are actively expanding the slime team through hiring to help with this — if you're interested, feel free to reach out directly.

Thank you for your understanding and patience. We truly appreciate the effort community contributors put in, and we're sorry if this policy causes any inconvenience.

---

## 开源协作范围说明

感谢你对 slime 的关注和支持！社区的每一份贡献我们都非常珍视。为了保证项目的长期健康发展，请在提交 issue 或 PR 之前仔细阅读以下内容。

### 背景

slime 承担了智谱内部的大量实验，包括 GLM 4.5 至 5 的全部 RL 流程，以及大量的日常实验。我们开源的初衷是相信智谱内部的训练场景覆盖了大多数前沿算法需求，希望能够为社区提供一套**正确且高效**的大规模 RL 训练 Infra，同时也希望在此基础上和社区进行 Infra 性能优化上的共建。

在我们的已知范围内，目前只有极少的前沿大模型团队愿意公开如此核心且完整的 Infra 组件——这背后是公司对开源社区的极大热情。相应的，从维护者的角度，我们会充分利用这个开明的政策，让智谱内部使用的 slime 与开源版本同步，保持核心竞争力；同时，我们也希望在开源协作的过程中尽量不影响内部的研发节奏。因此，slime 的设计与开发 roadmap 需要优先参考智谱内部的需求，暂时无法完全在开源社区内进行讨论。

### 协作范围

我们将开源协作的范围限制在 **bug fix** 和一些**通用的大规模 RL 优化**上。在这方面我们也和社区达成了多次成功的合作，例如：

- RL 中的投机采样（[文档](https://thudm.github.io/slime/en/advanced/speculative_decoding.html)）
- 低精度训练：fp8 rollout + bf16/fp8 training，int4 rollout + int4 QAT training（[文档](https://thudm.github.io/slime/en/advanced/low_precision_training.html)）
- 确定性训练（[文档](https://thudm.github.io/slime/en/advanced/reproducibility.html)）

### 我们欢迎的

| 类别 | 说明 |
|------|------|
| **Bug 报告** | 崩溃、结果错误、文档错误等 |
| **Bug 修复** | 带有测试或清晰复现步骤的修复 PR |
| **通用 RL 优化** | 有明确 benchmark 且可通过 CI 或常规训练验证的性能优化 |

### 暂时不在协作范围内的

| 类别 | 原因 |
|------|------|
| **较大范围的代码重构** | 会给内外部版本同步带来较多额外工作，尤其是在与内部算法团队的沟通协调上。 |
| **带有项目规划建议的标准或抽象** | 例如引入某种通用数据标准、eval 标准、工具构建基类等。标准的设立在大多数团队中会涉及到非技术因素，slime 的设计中故意避开了类似的内容，一方面不希望将智谱内部的管理偏好投射给社区，另一方面也便于内部不同方向的团队进行合适的选型。 |
| **无法进行明确验证的功能** | 训练框架的正确性至关重要。如果一个功能不能通过 CI 或智谱内部常规训练进行验证，我们就难以及时发现和修复问题，这对项目的长期可靠性会带来不小的风险。 |
| **与 RL 框架较为独立的功能** | 例如整套算法复现流程。这类内容较难纳入日常验证流程，不太容易持续保证正确性。slime 是一个相对轻量的框架，更像是 Flask 而非 Django。建议在独立的 repo 中搭建，我们也非常愿意在 README 中引用所有使用了 slime 的项目链接。 |
| **对 Megatron 的大幅度改动** | 目前我们没有计划通过 slime 维护一套 Megatron fork。slime 的目标是能够相对无痛地切换 Megatron 版本，Megatron 的性能优化和功能补全不在主要目标中。 |

### 为什么需要这样的策略？

RL Infra 的设计与后训练的规划有很强的绑定关系，公开整个后训练的研发规划并不包含在 slime 框架的开源目标内。这种较为聚焦的策略是我们认为将 slime 这个项目更长久地维护下去的必要手段。

暂时无法纳入上述 feature 也许会减慢 slime 的功能迭代速度，我们会通过招聘的方式逐渐扩展 slime 团队来改善这一点——对此有兴趣的朋友欢迎直接私信联系~

感谢大家的理解与支持，如果这一策略给您带来了不便，我们深表歉意。
