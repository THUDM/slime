# Rollout Buffer

## 概述

Rollout Buffer 是用于辅助纯异步 agent 训练的独立组件，其主要功能是使用 slime 训练启动的 LLM OpenAI Server 进行智能体轨迹的生成。

### 工作流程

```
slime Training Process ←─── HTTP API ───→ Rollout Buffer
        ↓                                      ↓
   LLM Server ←─────── HTTP Requests ─────── Agent Framework
        ↓                                      ↓
   Model Response ──────────────────────→ Trajectory Generation
```

对于每一个不同的 Agent 任务，都应该对应一个独立的 Generator 类，负责生成该类任务的轨迹。Rollout Buffer 会自动读取并加载不同类型的 Generator。

## 快速开始

### 基本使用流程

1. **复制模板**：将 `base_generator.py` 作为模板进行复制
2. **修改任务类型**：将 `TASK_TYPE` 修改为您的任务名称（不能与其他 Generator 重复）
3. **实现核心函数**：实现 `run_rollout()` 函数
4. **可选定制**：根据需要重写五个可选函数


Generator 文件必须以 `_generator.py` 结尾，并放置在 `generator/` 目录下：

```
generator/
├── base_generator.py      # Math 任务实现（默认模板）
└── your_task_generator.py # 您的自定义任务
```

每个 Generator 文件必须定义 `TASK_TYPE` 与 `run_rollout()`。

此外，Rollout Buffer 还提供了一些可自定义的函数来满足不同任务的特殊需求。如果不提供自定义实现，系统将使用默认实现（位于 `slime_plugins/rollout_buffer/default_func.py`）。

### 示例脚本

请仿照 [示例：Qwen3-4B 模型](../../docs/zh/models/qwen3-4B.md) 文档中配置好 slime 的运行环境，下载数据，并转换模型 ckpt。之后分别运行

```bash
cd slime_plugins/rollout_buffer
bash rollout_buffer_example.sh

# In a different terminal
python buffer.py
```
