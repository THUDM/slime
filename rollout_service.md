# Rollout Service 实现研究

> 研究对象：SLIME 框架中的 Rollout Service
> 研究日期：2026-02-25

---

## 一、概述

Rollout Service 是 SLIME（一个用于大模型 RL 后训练的分布式框架）中的核心推理与数据生成组件。它的主要职责是：

1. **数据生成**：基于语言模型推理，生成新的训练数据
2. **奖励计算**：对生成的样本计算奖励/验证器输出
3. **数据缓冲**：在训练模块消费之前管理训练数据
4. **多轮对话**：支持 partial rollout 和持续生成场景
5. **性能优化**：负责内存管理、offload/onload、健康监控

---

## 二、核心文件

| 文件路径 | 行数 | 功能 |
|---|---|---|
| `/slime/ray/rollout.py` | ~1002 | 核心 RolloutManager 和 RolloutServer 逻辑 |
| `/slime/rollout/sglang_rollout.py` | ~581 | 推理生成和 RM（奖励模型）pipeline |
| `/slime/rollout/data_source.py` | ~228 | 数据加载与 buffer 管理 |
| `/slime/utils/health_monitor.py` | — | 推理引擎健康监控 |
| `/slime/rollout/rm_hub/` | — | 奖励模型实现集合 |
| `/slime/rollout/filter_hub/` | — | 生成过程中的样本过滤 |

---

## 三、架构层次

### 3.1 总体数据流

```
DataSource（RolloutDataSource / RolloutDataSourceWithBuffer）
    │
    ├─→ 按 rollout_batch_size 拉取 samples
    ↓
RolloutManager.generate()
    ├─→ generate_rollout_async()   异步生成
    ├─→ async_rm()                  奖励计算
    ├─→ _convert_samples_to_train_data()  转为训练数据
    ├─→ 按 DP size 分片
    ↓
Training Module（Megatron）
```

### 3.2 组件层次结构

```
RolloutManager（Ray Actor，全局协调器）
  └─ RolloutServer（模型服务抽象，含 Router）
       ├─ EngineGroup（同质引擎组，普通推理）
       └─ EngineGroup（PD 分离时的 Prefill/Decode 引擎组）
            └─ SGLang Engine（实际推理 Actor）
```

---

## 四、核心类详解

### 4.1 `RolloutManager`（`/slime/ray/rollout.py`）

`@ray.remote` 装饰，作为 Ray Actor 运行，是整个 Rollout 流程的入口和调度中心。

**关键方法：**

| 方法 | 功能 |
|---|---|
| `generate(rollout_id)` | 执行一轮 rollout：生成 → 奖励计算 → 转换为训练数据 → 按 DP 分片 |
| `eval(rollout_id)` | 评估 rollout（不用于训练） |
| `offload()` | 释放 GPU 内存（保留权重，释放 KV cache 和 CUDA graphs） |
| `onload()` | 恢复 GPU 内存占用 |
| `recover_rollout_engines()` | 故障恢复，重启死亡的推理引擎 |
| `save()` / `load()` | 数据源状态持久化 |

### 4.2 `RolloutServer`（Dataclass）

管理一个或多个 EngineGroup，为单个模型提供服务，带共享 Router。

**关键特性：**
- 支持 PD（Prefill-Decode）分离：prefill 和 decode 阶段用不同的 EngineGroup
- Router 可选 SGLang Router 或 SLIME 原生 Router
- 统一管理各 EngineGroup 的生命周期

### 4.3 `EngineGroup`（Dataclass）

一组同质的 SGLang 推理引擎，配置相同。

**关键属性：**

| 属性 | 含义 |
|---|---|
| `engines` | node-0 上的引擎列表（多节点服务时仅含主节点引擎） |
| `all_engines` | 所有节点的完整引擎列表 |

**关键方法：**

| 方法 | 功能 |
|---|---|
| `start_engines()` | 异步创建 Ray Actor 并初始化 |
| `offload()` | 内存释放 |
| `onload()` | 恢复内存 |

### 4.4 `GenerateState`（`/slime/rollout/sglang_rollout.py`，单例）

维护推理状态：
- tokenizer、processor
- 采样参数（sampling parameters）
- 异步信号量（concurrency control）
- DP rank 均衡追踪

---

## 五、生成 Pipeline 详解

### 5.1 调用链

```
generate_rollout_async()         ← 主异步协调器
  └─ generate_and_rm_group()     ← 对一组样本生成+奖励
       ├─ generate()             ← 单个样本通过 SGLang Router 生成
       └─ async_rm()             ← 对样本应用奖励模型
```

### 5.2 生成特性

- **多轮对话支持**：可继续已有的对话历史（partial rollout）
- **多模态输入**：通过 processor 处理图片输入
- **Log Probability 跟踪**：用于 off-policy 修正
- **动态样本过滤**：生成过程中实时过滤低质量样本
- **确定性推理**：支持 seeded sampling，保证可复现性
- **Session ID**：保证一致路由（一致性哈希）

---

## 六、数据源管理

### 6.1 两种数据源

| 类 | 特性 |
|---|---|
| `RolloutDataSource` | 从固定数据集读取 |
| `RolloutDataSourceWithBuffer` | 支持 partial rollout 和 replay buffer 策略 |

### 6.2 Buffer 策略

- 当 rollout 被中断时，收集未完成的样本
- 将样本存入 buffer，供下一个 iteration 使用
- 保留 loss mask，用于部分生成场景

---

## 七、样本（Sample）数据结构

`Sample` 对象（定义于 `/slime/utils/types.py`）包含：

| 字段 | 说明 |
|---|---|
| `tokens` | token ids |
| `response` | 生成的响应文本 |
| `prompt` | 输入 prompt |
| `reward` | 奖励值 |
| `loss_mask` | loss 计算掩码 |
| `metadata` | 元数据（支持多轮场景） |

**Sample 状态机：**
```
PENDING → COMPLETED
         → TRUNCATED
         → ABORTED
```

### 样本转换为训练数据（`_convert_samples_to_train_data()`）

- 提取：tokens、response_lengths、rewards、loss_masks
- 奖励归一化：支持 group-level normalization（用于 GRPO/GSPO）
- 支持自定义奖励后处理函数
- 保留 metadata 用于多轮场景

---

## 八、Router 机制

### 8.1 SGLang Router

- 标准 Router，多种负载均衡策略
- Round-robin / 一致性哈希

### 8.2 SLIME 原生 Router

- 支持 Middleware（如 RadixTree 用于前缀缓存）
- 与 SLIME 框架更紧密集成

---

## 九、内存管理机制

### 9.1 两级卸载策略

| 操作 | 效果 |
|---|---|
| `offload` | 释放 KV cache 和 CUDA graphs，保留权重 |
| `onload` | 恢复完整内存占用 |

### 9.2 生命周期

```
Rollout 生成
    → offload engines（训练期间节约内存）
    → Training 执行
    → onload engines（恢复推理能力）
    → 下一轮 Rollout
```

这种设计减少了在非重叠阶段的峰值内存占用。

---

## 十、故障容错（Fault Tolerance）

### 10.1 健康监控（`RolloutHealthMonitor`）

- 作为后台线程运行
- 持续检测推理引擎健康状态
- 检测到死亡引擎后自动标记为 `None`
- 在 offload/onload 期间自动暂停检测，避免误判

### 10.2 恢复机制

- `RolloutServer.recover()`：重启死亡引擎
- 多个 EngineGroup 的初始化可以交叠进行（重叠优化）
- 恢复后自动执行 offload → onload 内存管理

---

## 十一、配置参数（关键）

| 参数 | 说明 |
|---|---|
| `rollout_num_gpus` | 分配给 rollout 的总 GPU 数 |
| `rollout_num_gpus_per_engine` | 每个推理引擎使用的 GPU 数 |
| `rollout_batch_size` | 每轮 rollout 的样本数量 |
| `n_samples_per_prompt` | 每个 prompt 生成多少个响应 |
| `sglang_router_ip/port` | Router 地址（分布式推理） |

---

## 十二、与 SLIME 框架的集成

### 12.1 部署方式

- RolloutManager 作为 Ray Actor 创建，部署在专用 GPU 节点
- 通过 Ray Object Store 与训练 Actor 通信
- 通过 PlacementGroup 实现协同部署（co-located）

### 12.2 并行优化

- Prefill 和 Decode 引擎分离运行（PD disaggregation）
- 多节点推理，支持 Tensor Parallelism 分布式
- 并发执行健康检测和生成任务

---

## 十三、创新亮点总结

| 创新点 | 说明 |
|---|---|
| **Async-First 架构** | 使用 asyncio 充分利用 GPU，减少等待 |
| **生成与奖励解耦** | 生成和奖励计算分离，提高灵活性 |
| **内存感知调度** | Offload/Onload 策略优化 GPU 内存使用 |
| **自动故障恢复** | 引擎崩溃后自动重启，无需人工介入 |
| **弹性数据 Pipeline** | 支持自定义生成函数、奖励模型和样本过滤 |
| **PD 分离推理** | Prefill 和 Decode 阶段解耦，提升吞吐 |

---

## 十四、总结

SLIME 的 Rollout Service 是一个面向大规模 RL 训练设计的高吞吐、高容错推理服务。它通过以下设计实现了工业级别的稳定性和性能：

- 用 **Ray Actor** 实现分布式协调
- 用 **AsyncIO** 实现高效并发
- 用 **SGLang** 作为推理后端
- 用 **Health Monitor** 实现自动故障恢复
- 用 **Offload/Onload** 实现内存复用

该架构已被用于支持 GLM-5 等生产模型的训练，以及 P1、RLVE、TritonForge 等研究项目。
