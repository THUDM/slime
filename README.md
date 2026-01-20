# slime

[中文版](./README_zh.md)

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

**slime** is an LLM post-training framework for RL scaling, providing two core capabilities:

1.  **High-Performance Training**: Supports efficient training in various modes by connecting Megatron with SGLang;
2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

slime is the RL-framework behind [GLM-4.7](https://z.ai/blog/glm-4.7), [GLM-4.6](https://z.ai/blog/glm-4.6), [GLM-4.5](https://z.ai/blog/glm-4.5) and apart from models from Z.ai, we also supports the following models:
- Qwen3 series (Qwen3Next, Qwen3MoE, Qwen3), Qwen2.5 series;
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1);
- Llama 3.

## Blogs

- Our vision: [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/).
- Our ideas on agentic training: [Agent-Oriented Design: An Asynchronous and Decoupled Framework for Agentic RL](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL-2278e692d081802cbdd5d37cef76a547)
- v0.1.0 release note: [v0.1.0: Redefining High-Performance RL Training Frameworks](https://thudm.github.io/slime/blogs/release_v0.1.0.html)

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Projects Built with slime](#projects-built-with-slime)
- [Arguments Walkthrough](#arguments-walkthrough)
- [Developer Guide](#developer-guide)
- [FAQ & Acknowledgements](#faq--acknowledgements)

## Architecture Overview

![arch](./imgs/arch.png)

**Module Descriptions**:

- **training (Megatron)**: Responsible for the main training process, reads data from the Data Buffer, and synchronizes parameters to the rollout module after training.
- **rollout (SGLang + router)**: Generates new data (including rewards/verifier outputs) and stores it in the Data Buffer.
- **data buffer**: A bridge module that manages prompt initialization, custom data, and rollout generation methods.

## Quick Start

For a comprehensive quick start guide covering environment setup, data preparation, training startup, and key code analysis, please refer to:
- [Quick Start Guide](./docs/en/get_started/quick_start.md)

We also provide examples for some use cases not covered in the quick start guide; please check [examples](examples/).

## Projects Built upon slime

slime has powered several novel research projects and production systems. Here are some notable examples:

### ⚛️ P1: Mastering Physics Olympiads with Reinforcement Learning

[**P1**](https://prime-rl.github.io/P1/) is a family of open-source physics reasoning models trained entirely through reinforcement learning. P1 leverages slime as the RL post training framework, and introduces a multi-stage RL training algorithm that progressively enhances reasoning ability through adaptive learnability adjustment and stabilization mechanisms. Enpowered by this training paradigm, P1 delivers breakthrough performance in open-source physics reasoning.

### 📈RLVE: Scaling LM RL with Adaptive Verifiable Environments

[**RLVE**](https://github.com/Zhiyuan-Zeng/RLVE) introduces an approach using verifiable environments that procedurally generate problems and provide algorithmically verifiable rewards, to scale up RL for language models (LMs). With joint training across 400 verifiable environments, RLVE enables each environment to dynamically adapt its problem difficulty distribution to the policy model's capabilities as training progresses.

### ⚡ TritonForge: Agentic RL Training Framework for Kernel Generation

[**TritonForge**](https://github.com/RLsys-Foundation/TritonForge) leverages slime's SFT & RL capabilities to train LLMs that automatically generate optimized GPU kernels. By using a two-stage training approach—supervised fine-tuning followed by reinforcement learning with multi-turn compilation feedback—TritonForge achieves remarkable results in converting PyTorch operations into high-performance Triton kernels.

### 🚀 APRIL: Accelerating RL Training with Active Partial Rollouts

[**APRIL**](https://github.com/RLsys-Foundation/APRIL) introduces a system-level optimization that seamlessly integrates with slime to accelerate the rollout generation phase in RL training. By intelligently over-provisioning requests and actively managing partial completions, APRIL addresses the long-tail generation bottleneck that typically consumes over 90% of RL training time.

### 🏟️ qqr: Scaling Open-Ended Agents with ArenaRL & MCP

[**qqr**](https://github.com/Alibaba-NLP/qqr) (a.k.a. hilichurl) is a lightweight extension for slime designed to evolve open-ended agents. It implements the **ArenaRL** algorithm to tackle discriminative collapse through tournament-based relative ranking (**e.g., Seeded Single-Elimination, Round-Robin**) and seamlessly integrates the **Model Context Protocol (MCP)**. qqr leverages slime's high-throughput training capabilities to enable scalable, distributed evolution of agents in standardized, decoupled tool environments.

These projects showcase slime's versatility—from training code-generation models to optimizing RL training systems—making it a powerful foundation for both research and production deployments.

## Arguments Walkthrough

Arguments in slime are divided into three categories:

1.  **Megatron arguments**: slime reads all arguments in Megatron. You can configure Megatron by passing arguments like `--tensor-model-parallel-size 2`.
2.  **SGLang arguments**: All arguments for the installed SGLang are supported. These arguments must be prefixed with `--sglang-`. For example, `--mem-fraction-static` should be passed as `--sglang-mem-fraction-static`.
3.  **slime-specific arguments**: Please refer to: [slime/utils/arguments.py](slime/utils/arguments.py)

For complete usage instructions, please refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Developer Guide

- **Contributions are welcome\!** If you have suggestions for new features, performance tuning, or feedback on user experience, feel free to submit an Issue or PR 😊

- Use [pre-commit](https://pre-commit.com/) to ensure code style consistency for your commits:

```bash
apt install pre-commit -y
pre-commit install

# run pre-commit to ensure code style consistency
pre-commit run --all-files --show-diff-on-failure --color=always
```

- For debugging tips, please refer to the [Debugging Guide](docs/en/developer_guide/debug.md)

## FAQ & Acknowledgements

- For frequently asked questions, please see the [Q\&A](docs/en/get_started/qa.md)
- Special thanks to the following projects & communities: SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch and others.
- To quote slime, please use:

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
