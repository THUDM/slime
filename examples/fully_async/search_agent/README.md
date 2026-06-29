# Fully Async Search Agent — 设计与实现分析

> 基于对 [SearchAgent-Zero](https://github.com/NLPJCL/SearchAgent-Zero) 代码的完整阅读，分析如何在 slime 中复刻全异步 Search Agent RL。

---

## 1. 关键发现

**SearchAgent-Zero 的 "fully async" 指的是训练级异步**（Rollouter + Trainer 通过 MessageQueue 解耦），其 Agent Loop 内部仍然是同步的：

```
每轮: await generate() → 等待完整输出 → 解析 <tool_call> → asyncio.gather 并发执行 tool → 下一轮
```

它并没有实现流式 token 级别的 tool call 增量检测+提前调度（即原 README 中的 Layer 2 A1-A4）。因此复刻的核心目标是：**把 SearchAgent-Zero 的训练级异步架构 + Agent Loop 状态机 + Staleness 管理，搬进 slime 现有的 fully_async_rollout 框架中**。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      train_async.py                         │
│  ┌──────────────────┐          ┌──────────────────────────┐ │
│  │  Actor Model     │          │  SGLang Rollout (Ray)    │ │
│  │  (本地 GPU)       │          │                          │ │
│  │                  │ 权重同步  │  generate_and_rm_group()  │ │
│  │  async_train() ──┼─────────►│    ↓                     │ │
│  │                  │          │  --custom-generate-fn ───┼─┼──► fully_async_agent_loop.py
│  │  update_weights()│          │    ↓                     │ │    (状态机 Agent Loop)
│  │       ↓          │          │  Sample → output_queue   │ │
│  │  reset_staleness │          │    ↓                     │ │
│  │       ↓          │          │  data_buffer (wrap)      │ │
│  │  继续训练        │          │    ↓                     │ │
│  └──────────────────┘          │  AsyncRolloutWorker      │ │
│                                │   _loop() 后台持续跑      │ │
│                                └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

async_hack.py (Patch 层):
  ┌─────────────────────┐
  │ StalenessDataBuffer │  ← 包装 data_buffer，注入 T5/T6
  │ monkey-patch worker │  ← 替换 _get_global_worker
  │ reset_staleness()   │  ← 导出给 train_async.py 调用
  └─────────────────────┘
```

## 3. 文件结构与职责

| 文件 | 职责 | 状态 |
|------|------|------|
| `fully_async_agent_loop.py` | Agent Loop 状态机（`generate()` 函数） | **待实现** |
| `async_hack.py` | Staleness + 流控 Patch（T5/T6）+ Partial Rollout 辅助 | **待实现** |
| `run-qwen3.5-9B-fully_async.sh` | 启动脚本 | **待实现** |
| `slime/rollout/fully_async_rollout.py` | AsyncRolloutWorker（框架，不改） | 已有 |
| `train_async.py` | 训练主循环（仅 +1 行 reset_staleness 调用） | 已有 |

---

## 4. 各模块详细设计

### 4.1 Agent Loop 状态机 (`fully_async_agent_loop.py`)

参考 SearchAgent-Zero [`tool_agent_loop_credit_assignment.py`](https://github.com/Meituan-Dianping/SearchAgent-Zero/blob/main/verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py) 的状态机设计：

```
PENDING ──► GENERATING ──► PROCESSING_TOOLS ──► GENERATING ──► ... ──► TERMINATED
            │                                     │
            └── 无 tool call → TERMINATED         └── tool 结果拼入 prompt，loss_mask=0
```

每轮 `GENERATING` 调用 slime 已有的 `sglang_streaming_rollout.generate_streaming()` 做流式生成，abort 时 sample 上已有 partial token。

**与 SearchAgent-Zero 的差异：**

| | SearchAgent-Zero | Slime 实现 |
|--|-----------------|-----------|
| 生成 API | vLLM `server_manager.generate()` | SGLang `/generate` HTTP SSE 流 |
| 流式消费 | 不支持（一次性返回 TokenOutput） | ✅ `generate_streaming()` 逐 chunk 写入 sample |
| Tool 解析 | `HermesToolParser.extract_tool_calls()` decode 完整 text → regex | 同逻辑，复用 regex 模式 |
| Tool 并发 | `asyncio.gather(*tasks)` | 同逻辑 |
| 搜索结果摘要 | `_generate_single_summary()` 调用 LLM | 先用 truncate 简化，后续加摘要 |
| Credit Assignment | `_mask_previous_response_tokens()` 只保留最后一轮 trainable | 对应 `loss_mask` 设置 |
| 异常轨迹过滤 | `abnormal_trajectory_dic` 追踪各种异常 | 直接复用 |

**需要实现的核心方法：**

```
generate(args, sample, sampling_params) → Sample
├── _handle_pending_state()       # apply chat template + tools schema
├── _handle_generating_state()    # 调用 generate_streaming() → 解析 tool call
│   ├── 异常检测: 重复 query / 解析错误 / 过多 turn / 超长
│   └── 正常: 提取 FunctionCall → 进入 PROCESSING_TOOLS
├── _handle_processing_tools_state()
│   ├── asyncio.gather 并发执行 search
│   ├── 搜索结果去重检测 (document signature overlap)
│   ├── 结果截断 (truncate by tokens)
│   └── 拼入 prompt_ids，loss_mask=0
└── 终止条件检查: response_length / max_turns / tool_parse_error / ...
```

预计 **~400 行**。

### 4.2 Staleness + 流控 Patch (`async_hack.py`)

**设计原则：不改 `fully_async_rollout.py` 框架代码，通过包装 `data_buffer` 注入 T5/T6。**

#### 切入方式

参照 [fully_async_rollout.py:82-91](../../slime/rollout/fully_async_rollout.py#L82-L91)，`_get_global_worker` 是全局 worker 的单例工厂：

```python
# fully_async_rollout.py (不改)
def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            _global_worker = AsyncRolloutWorker(args, data_buffer, ...)
            _global_worker.start()
        return _global_worker
```

参照 [fully_async_rollout.py:166](../../slime/rollout/fully_async_rollout.py#L166)，`_loop()` 中唯一从 buffer 取样本的点：

```python
# fully_async_rollout.py:166 (不改)
groups = self.data_buffer.get_samples(1)
```

#### 实现方案

```python
# async_hack.py

class StalenessDataBuffer:
    """包装 data_buffer，注入 staleness 计数 + pause/resume。"""

    def __init__(self, real_buffer, threshold: int):
        self._real = real_buffer
        self._threshold = threshold
        self._counter = 0
        self._lock = threading.Lock()
        self._resume = threading.Event()
        self._resume.set()

    def get_samples(self, n):
        self._resume.wait()          # block if paused
        samples = self._real.get_samples(n)
        with self._lock:
            self._counter += len(samples) if samples else 0
            if self._counter >= self._threshold:
                self._resume.clear()  # pause: 不再取新样本
        return samples

    def reset_staleness(self):
        with self._lock:
            self._counter = 0
        self._resume.set()           # resume

    def __getattr__(self, name):     # 透传其他方法
        return getattr(self._real, name)


# Monkey-patch _get_global_worker
_staleness_buffer: StalenessDataBuffer | None = None

def _patched_get_worker(args, data_buffer):
    global _staleness_buffer
    _staleness_buffer = StalenessDataBuffer(
        data_buffer,
        threshold=args.ppo_mini_batch_size * (args.staleness_threshold + 1),
    )
    return _original_get_worker(args, _staleness_buffer)

def reset_staleness():
    """train_async.py 在 update_weights() 后调用。"""
    if _staleness_buffer:
        _staleness_buffer.reset_staleness()

# Apply
import slime.rollout.fully_async_rollout as _fr
_original_get_worker = _fr._get_global_worker
_fr._get_global_worker = _patched_get_worker
```

#### 训练循环对接

[train_async.py:69](../../train_async.py#L69) 加一行：

```python
# train_async.py L69 之后
actor_model.update_weights()
from examples.fully_async.search_agent.async_hack import reset_staleness
reset_staleness()  # ← 新增: 唤醒生成
```

配合 [arguments.py](../../slime/utils/arguments.py) 加 `--staleness-threshold`（默认 1）。

#### Staleness 阈值计算

```
staleness_threshold = 1  →  允许 1 个 batch 的 stale 样本
max_stale = ppo_mini_batch_size × (staleness_threshold + 1)
```

即 staleness_threshold=1 时，最多允许 2 个 batch 的样本用旧权重生成，之后暂停。

参照 SearchAgent-Zero [`fully_async_rollouter.py:690-712`](https://github.com/Meituan-Dianping/SearchAgent-Zero/blob/main/verl/experimental/fully_async_policy/fully_async_rollouter.py#L690-L712) 的 `_should_pause_generation()`：

```python
# 暂停条件（二选一即停）：
# 1. queue_size >= max_queue_size（输出队列满，下游消费慢）
# 2. staleness_samples >= max_required_samples（旧权重样本过多）
```

预计 `async_hack.py` **~80 行**。

### 4.3 Partial Rollout 恢复 (C1)

这是 slime 相对于 verl **最大的风险点**。核心问题是：

> 权重同步 abort 一条轨迹时，sample 上已有 model token + tool token 混排。
> tool 已执行的 token 不可回滚（搜索 HTTP 请求已发出），只能跳过；
> model token 可以重新生成（新权重下）— 但需要知道边界在哪里。
>
> 当前 slime 的 `--partial-rollout` 只知道 "requeue 样本到 data_buffer"，
> 不知道 sample 上有哪些 token 是 model token（可重生成）、
> 哪些是 tool token（不可回滚）。

**解决思路**：在 Agent Loop 的 `agent_data` 中维护 `model_token_boundary`：

```
[turn 1: model tokens (trainable)] [turn 1: tool result tokens (non-trainable)]
[turn 2: model tokens (trainable)] [turn 2: tool result tokens (non-trainable)]
                                     ↑
                              如果 abort 在这里，
                              只重新生成 turn 2 的 model tokens
```

具体实现：
1. 每次 `_handle_processing_tools_state` 完成后，记录 tool token 结束位置
2. 每次 `_handle_generating_state` 开始生成前，记录 model token 起始位置
3. 当 sample.status == ABORTED 被 requeue 后再次进入 generate()：
   - 检查 `loss_mask` 找到最后一个 `loss_mask=1` 的 model token 位置
   - 从该位置之后恢复生成（即只重新生成被 abort 截断的那段 model token）
   - tool token 保留，skip over

这部分逻辑嵌入在 Agent Loop 的 `_handle_generating_state` 中，预计 **~60 行**。

---

## 5. 工作量汇总

| 模块 | 文件 | 行数 | 难度 |
|------|------|------|------|
| Agent Loop 状态机 | `fully_async_agent_loop.py` | ~400 | 🔴 高 |
| Staleness + 流控 Patch | `async_hack.py` | ~80 | 🟡 中 |
| Partial Rollout 恢复 | Agent Loop 内 | ~60 | 🔴 高（最大风险） |
| 训练循环对接 | `train_async.py` +1 行, `arguments.py` +1 行 | ~5 | 🟢 低 |
| 启动脚本 + 配置 | `run-*.sh`, tool config | ~60 | 🟢 低 |
| **合计** | | **~605 行** | |

**不改任何 slime 框架代码**（`fully_async_rollout.py`、`sglang_rollout.py` 等），所有增量在 `examples/fully_async/search_agent/` 目录内闭环。仅 `train_async.py` 加一行 `reset_staleness()` 调用。

---

## 6. 与 SearchAgent-Zero 的架构对比

| | SearchAgent-Zero | Slime（复刻后） |
|--|-----------------|----------------|
| 训练级异步 | Rollouter + Trainer 对等 Ray Actor，MessageQueue 解耦 | Trainer 主循环驱动，AsyncRolloutWorker 后台线程辅助 |
| 样本传递 | MessageQueue (Ray Actor, asyncio.Condition) | `output_queue.Queue(maxsize=1000)` |
| 权重同步 | `CheckpointEngineManager.update_weights()` → `reset_staleness()` | `actor_model.update_weights()` → `reset_staleness()` |
| Agent Loop | `ToolAgentLoop` 状态机，注册到 verl 框架 | `generate()` 函数，通过 `--custom-generate-function-path` 接入 |
| Tool 执行 | verl.tools 框架（create/execute/release 生命周期） | 直接调用 search API（复用 search-r1 模式） |
| 配置 | Hydra YAML | argparse + Python dict |
| Staleness | Rollouter 内置 | `async_hack.py` Patch |
| Partial Rollout | `FullyAsyncLLMServerManager.generate()` 内 while loop | Agent Loop 内 model_token_boundary 追踪 |
| 接入方式 | `--config-name search_multiturn_grpo` | `--custom-generate-function-path examples/fully_async/search_agent/fully_async_agent_loop.py` |

---

## 7. 参考文件

- SearchAgent-Zero Agent Loop: [`verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py`](https://github.com/Meituan-Dianping/SearchAgent-Zero/blob/main/verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py)
- SearchAgent-Zero FullyAsync Rollouter: [`verl/experimental/fully_async_policy/fully_async_rollouter.py`](https://github.com/Meituan-Dianping/SearchAgent-Zero/blob/main/verl/experimental/fully_async_policy/fully_async_rollouter.py)
- SearchAgent-Zero FullyAsync Trainer: [`verl/experimental/fully_async_policy/fully_async_trainer.py`](https://github.com/Meituan-Dianping/SearchAgent-Zero/blob/main/verl/experimental/fully_async_policy/fully_async_trainer.py)
- SearchAgent-Zero MessageQueue: [`verl/experimental/fully_async_policy/message_queue.py`](https://github.com/Meituan-Dianping/SearchAgent-Zero/blob/main/verl/experimental/fully_async_policy/message_queue.py)
- SearchAgent-Zero FullyAsyncLLMServerManager: [`verl/experimental/fully_async_policy/agent_loop/agent_loop.py`](https://github.com/Meituan-Dianping/SearchAgent-Zero/blob/main/verl/experimental/fully_async_policy/agent_loop/agent_loop.py)
- Slime Fully Async Rollout: [`slime/rollout/fully_async_rollout.py`](../../slime/rollout/fully_async_rollout.py)
- Slime Streaming Rollout: [`slime/rollout/sglang_streaming_rollout.py`](../../slime/rollout/sglang_streaming_rollout.py)
- Slime Train Async: [`train_async.py`](../../train_async.py)
- Slime Search-R1 (同步参考): [`examples/search-r1/generate_with_search.py`](../search-r1/generate_with_search.py)
