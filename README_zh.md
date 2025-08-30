# slime

[English](./README.md)

**slime** 是为 RL scaling 设计的 LLM post‑training 框架，提供两大核心能力：

1. **高性能训练**：通过连接 Megatron 与 SGLang，支持各种模式的高效训练；
2. **灵活的数据生成**：通过自定义数据生成接口以及 server based engine，实现任意的数据训练数据生成流程。

## 目录

- [架构总览](#架构总览)
- [快速开始](#快速开始)
- [Checkpoint 格式转换](#checkpoint-格式转换)
- [启动训练流程](#启动训练流程)
- [参数说明](#参数说明)
- [开发指南](#开发指南)
- [常见 Q&A 与致谢](#常见-qa-与致谢)

## 架构总览

![arch](./imgs/arch.png)

**模块说明**：

- **training (Megatron)**：负责主训练流程，从 Data Buffer 读取数据，训练完后将参数同步至 rollout 模块；
- **rollout (SGLang + router)**：生成新数据（含 reward/verifier），存储至 Data Buffer；
- **data buffer**：桥梁模块，管理 prompt 初始化、自定义数据与 rollout 生成方法。

## 快速开始

有关环境配置、数据准备、训练启动和关键代码分析的完整快速开始指南，请参考：

- [快速开始指南](./docs/zh/quick_start.md)

## 参数说明

参数分为三类：

1. **megatron 参数**：slime 会读取 `PYTHONPATH` 中的 megatron 里设置的所有参数，可以通过传入如 `--tensor-model-parallel-size 2` 的方式配置 megatron；
2. **sglang 参数**：支持环境中安装的 sglang 的所有参数，这些参数需要以 `--sglang` 起始，例如 `--mem-fraction-static` 需要通过 `--sglang-mem-fraction-static` 传入。
3. **slime 自身的参数**：请见：[slime/utils/arguments.py](slime/utils/arguments.py)

完整使用说明请查阅 [使用文档](docs/zh/usage.md)。

## 开发指南

- **欢迎贡献！** 若有功能建议、性能调优或使用体验反馈，欢迎提交 Issue / PR 😊

- 使用 [pre-commit](https://pre-commit.com/) 保证提交代码风格：

  ```bash
  apt install pre-commit -y
  pre-commit install
  ```

- 调试技巧请参考 [debug 指南](docs/zh/debug.md)

## 常见 Q&A 与致谢

- 常见问题请见 [Q&A](docs/zh/qa.md)
- 特别感谢以下项目 & 社区：SGLang、Megatron‑LM、mbridge、OpenRLHF、veRL、Pai-Megatron-Patch 等。
