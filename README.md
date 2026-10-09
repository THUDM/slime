# slime

[中文版](./README_zh.md)

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![CI](https://img.shields.io/github/actions/workflow/status/THUDM/slime/pr-test.yml?branch=main&event=push&label=CI&logo=github)](https://github.com/THUDM/slime/actions/workflows/pr-test.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

**slime** is an LLM post-training framework for RL scaling. It combines Megatron training with SGLang rollout and exposes custom interfaces for data generation and rewards.

Training, rollout, the data buffer, and environment feedback share one dataflow. This supports math, code, search, tool use, and long-horizon agent workflows within the same training loop.

## Why This Design Matters

- **Production experience**: slime is the RL framework behind [GLM-5.3-Flash](https://z.ai/blog/glm-5.3-flash), [GLM-5.3](https://z.ai/blog/glm-5.3), [GLM-5.2](https://z.ai/blog/glm-5.2), [GLM-5.1](https://z.ai/blog/glm-5.1), [GLM-5](https://z.ai/blog/glm-5), [GLM-4.7](https://z.ai/blog/glm-4.7), [GLM-4.6](https://z.ai/blog/glm-4.6), and [GLM-4.5](https://z.ai/blog/glm-4.5).
- **Direct engine integration**: Megatron arguments are available directly; SGLang arguments use the `--sglang-` prefix.
- **Customizable data generation**: generation functions, reward functions, verifiers, and environments connect through documented interfaces.
- **Correctness and reliability**: separate rollout-only and train-only debugging, reproducibility controls, recovery, and CPU/GPU tests support long-running experiments.

## Supported Models

Alongside the GLM family, slime supports Qwen (Qwen3.6, Qwen3.5, Qwen3-Next, Qwen3 MoE, Qwen3, Qwen2.5), DeepSeek (V3, V3.1, R1), and Llama 3. See the model configurations in [scripts/models](scripts/models/) and the [training examples](https://thudm.github.io/slime/).

## Engine Configuration and Deployment

Use Megatron arguments directly for parallelism, optimizers, checkpoints, and model configuration. Prefix installed SGLang arguments with `--sglang-`; for example, `--mem-fraction-static` becomes `--sglang-mem-fraction-static`.

For more involved deployments:

- [SGLang Config](docs/en/advanced/sglang-config.md): YAML configuration for server groups, multiple models, and per-group overrides.
- [PD Disaggregation](docs/en/advanced/pd-disaggregation.md): separate prefill and decode resources.
- [Delta Weight Sync](docs/en/advanced/delta-weight-sync.md): send changed weight bytes over shared storage.
- [External Rollout Engines](docs/en/advanced/external-rollout-engines.md): connect serving processes managed outside the training job, including disk-based updates across different GPU fleets.

## Correctness, Stability, and CI

CPU tests cover core behavior and customization contracts. GPU tests exercise dense and MoE training, rollout deployment, checkpointing, precision, fully asynchronous rollout, distillation, and debug replay. See [CI](docs/en/developer_guide/ci.md) for the test matrix and how to run it.

Engineering guides: [Debugging](docs/en/developer_guide/debug.md), [Reproducibility](docs/en/advanced/reproducibility.md), [Fault Tolerance](docs/en/advanced/fault-tolerance.md), [Tracing](docs/en/developer_guide/trace.md), and [Profiling](docs/en/developer_guide/profiling.md).

## Blogs

- [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/)
- [Agent-Oriented Design: An Asynchronous and Decoupled Framework for Agentic RL](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL-2278e692d081802cbdd5d37cef76a547)
- [slime v0.1.0: Redefining High-Performance RL Training Frameworks](https://thudm.github.io/slime/blogs/release_v0.1.0.html)

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Ecosystem Built on slime](#ecosystem-built-on-slime)
- [Arguments Walkthrough](#arguments-walkthrough)
- [Code Reading Path](#code-reading-path)
- [Developer Guide](#developer-guide)
- [FAQ & Acknowledgements](#faq--acknowledgements)

## Architecture Overview

![arch](./imgs/arch.png)

**Module Descriptions**:

- **training (Megatron)**: Responsible for the main training process, reads data from the Data Buffer, and synchronizes parameters to the rollout module after training.
- **rollout (SGLang + router)**: Generates new data (including rewards/verifier outputs) and stores it in the Data Buffer. Custom generate functions can wrap this with multi-turn loops, tool calls, environment/sandbox interaction, and verifier-based reward.
- **data buffer**: A bridge module that manages prompt initialization, custom data, and rollout generation methods (including agentic workflows that produce samples through the same interface).

The default payload transport is Ray `object-store`. Selecting `--rollout-data-transport straw` uses [straw](https://github.com/zhuzilin/straw) for persistent prompt tasks, rollout continuations and training batches on shared JuiceFS storage. Ray carries control messages and references; generation and training processes read and write packed payloads directly. See the [straw usage and recovery guide](docs/en/advanced/straw.md).

## Quick Start

Follow the [Quick Start Guide](docs/en/get_started/quick_start.md) for environment setup, data preparation, and your first training run.

We also provide examples for some use cases not covered in the quick start guide; please check [examples](examples/).

### Agentic RL examples

These agentic RL examples use the standard rollout and data buffer interfaces:

- [`examples/multi_agent`](examples/multi_agent/README.md): Multi-agent generation via `--custom-generate-function-path` inside the standard rollout loop.
- [`examples/search-r1`](examples/search-r1/): Search/RAG-style multi-turn generation via `--custom-generate-function-path`.
- [`examples/fully_async`](examples/fully_async/README.md): Fully-async rollout, useful for long-tail agentic generation where some samples take much longer than others.
- [`examples/coding_agent_rl`](examples/coding_agent_rl/README.md): End-to-end SWE coding-agent RL with sandboxed tool use, test-based rewards, and token-correct trajectory segments via `--custom-generate-function-path`.

See the [Customization Guide](docs/en/get_started/customization.md) for which interface to use for a given agentic workflow.

## Ecosystem Built on slime

These independent projects build on slime for model post-training, agentic RL, domain applications, and rollout research.

### Dressage

[Dressage](https://github.com/Accio-Lab/Dressage) — Alibaba Accio’s agentic RL framework, with Paddock, Sandbox, and Proxy layers for interaction, execution placement, and token-level trajectory capture. It supports black-box agents and configurable sandboxes.

### Miles

[Miles](https://github.com/radixark/miles) — RadixArk’s large-model post-training framework, extending slime with SGLang integration, deployment and operations tooling, LoRA, TITO, and low-precision training.

### vime

[vime](https://github.com/vllm-project/vime) — A post-training framework maintained by the vLLM project. It retains slime’s Megatron training and data generation design while using vLLM and vllm-router for rollout.

### Relax

[Relax](https://github.com/redai-infra/Relax) — RedAI Infra’s multimodal agentic RL framework, using Ray Serve, TransferQueue, and asynchronous checkpoint synchronization to separate training, rollout, and teacher/reference computation.

### OpenClaw-RL

[OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) — Personalized OpenClaw training from conversation feedback, using GRPO or on-policy distillation alongside ongoing API serving.

### P1

[P1](https://prime-rl.github.io/P1/) — Physics reasoning models trained with multi-stage RL, adaptive task difficulty, and training stabilization.

### RLVE

[RLVE](https://github.com/Zhiyuan-Zeng/RLVE) — RL across 400 procedurally generated, verifiable environments, with task difficulty adapted to the current policy.

### TritonForge

[TritonForge](https://github.com/RLsys-Foundation/TritonForge) — GPU kernel generation using SFT followed by RL with multi-turn compilation feedback.

### APRIL

[APRIL](https://github.com/RLsys-Foundation/APRIL) — Rollout throughput optimization through additional in-flight requests and active management of partial generations.

### qqr

[qqr](https://github.com/Alibaba-NLP/qqr) — Open-ended agent training with ArenaRL tournament ranking and MCP tool environments.

### ART

[ART](https://github.com/awslabs/agentcore-rl-toolkit) — An SDK for training production agents on AWS Bedrock AgentCore Runtime. It reuses agent workflows, captures trajectories at the model gateway, and offers slime as a training backend.

## Arguments Walkthrough

Arguments in slime are divided into three categories:

1.  **Megatron arguments**: slime reads Megatron arguments directly. You can configure Megatron by passing arguments like `--tensor-model-parallel-size 2`.
2.  **SGLang arguments**: All arguments for the installed SGLang are supported through pass-through. These arguments must be prefixed with `--sglang-`. For example, `--mem-fraction-static` should be passed as `--sglang-mem-fraction-static`.
3.  **slime-specific arguments**: Please refer to: [slime/utils/arguments.py](slime/utils/arguments.py)

For complete usage instructions, please refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Code Reading Path

Start from the training loop and follow the calls only as deep as needed:

```text
train.py: train
├─ slime/ray/placement_group.py       Ray resource and worker initialization
├─ slime/ray/rollout.py              RolloutManager.generate: rollout orchestration
│  └─ slime/rollout/sglang_rollout.py  Sample generation and reward computation
└─ slime/ray/actor_group.py          RayTrainGroup.async_train: training dispatch
   └─ slime/backends/megatron_utils/actor.py
      ├─ model.py                    Megatron model execution
      └─ loss.py                     RL losses and advantages
```

On a first pass, treat `slime/utils/arguments.py` as the configuration entry point. The deployment details in `slime/backends/sglang_utils/` and the weight-sync implementations under `slime/backends/megatron_utils/update_weight/` can also wait until you need to change those areas.

## Developer Guide

- Read the [contribution guidelines](CONTRIBUTING.md) before submitting an issue or PR.

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
