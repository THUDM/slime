# Trainer 通信时间线

slime 可以把训练与权重同步阶段写成低开销 JSONL 时间线。该功能默认关闭，不会改变 collective 顺序，也不会在训练路径中增加同步。

使用每 rank 独立路径启用：

```bash
python train.py \
  ... \
  --communication-timeline /path/to/traces/slime-{role}-{rank}.jsonl
```

也可以设置等价环境变量 `SLIME_COMMUNICATION_TIMELINE`。路径支持 `{rank}`、`{trainer_rank}`、`{local_rank}`、`{pid}`、`{hostname}`、`{role}` 和 `{world_size}`。多 rank 运行若没有 rank 占位符，slime 会自动追加 `.rank-N`；省略 `{role}` 时，actor、critic 与磁盘同步 orchestrator 也会自动使用独立后缀。该机制适用于任意 world size，不依赖特定 rank 拓扑。slime 会为同一 actor group 生成共享 `run_id`；也可通过 `--communication-timeline-run-id` 或 `SLIME_COMMUNICATION_TIMELINE_RUN_ID` 显式指定。

## 内置阶段

trainer 会记录：

- `train_forward_backward`：Megatron forward/backward schedule；
- `grad_sync`：Megatron 最终梯度同步回调；
- `optimizer_step`；
- `weight_convert`：生成每个 HF 权重 bucket 的工作；
- `weight_bucket_ready` 与 `weight_bucket_send`；
- `engine_bucket_receive`：trainer 侧观察到传输完成的边界；
- `engine_load_weights`：等待 engine 更新请求返回的时间；
- `weight_sync_complete`。

`engine_bucket_receive` 是 trainer 侧观测，不是 engine 内部时间戳；记录中的 `observation` metadata 会说明它对应的具体边界。

每条记录都有 `global_step`、`rollout_id`、`weight_version`、`bucket_id`、`trainer_rank`、`engine_id`、`message_bytes` 和 `transport` 等共享字段。不适用的字段为 `null`。`sequence_id` 在进程内单调递增，`logical_operation_id` 则由当前可用的生命周期 ID 与操作名组合而成。

CUDA span 在框架当前 stream 上记录 event。正常运行中只做非阻塞查询，在进程退出时才排空尚未完成的 event。schema v2 用 `gpu_timestamp_semantics="event-bracket"`、`timestamp_domain="process-realtime-projected-cuda-event"` 和 `clock_sync_error_bound_us=null` 显式标注时间来源。投影后的 GPU 区间表示框架操作可执行到观测完成的边界，并不宣称等于 ProcessGroupNCCL 内部 stream 上 NCCL kernel 的精确起点；各进程 realtime clock 也没有经过测量的跨 rank 误差上界。每个 span 同时生成名为 `slime.comm/<operation>` 的 NVTX range，可在 Nsight Systems 中精确关联 kernel。

自动通信策略不得直接使用这些 event-bracket 时间戳进行选择或执行；在 adapter 为每个参与 rank 提供 kernel-observed 时间戳和实测时钟同步误差上界之前，策略消费者必须 fail closed。内置时间线仍可用于生命周期诊断，以及通过 NVTX range 与精确 profiler trace 做关联。

未配置 timeline 时，公开 helper 会复用同一个 no-op phase，不读取时钟、不导入 torch、不分配 CUDA event、不 push NVTX、不扫描 bucket tensor 大小、不更新 context variable、不序列化 JSON、不打开文件，也不做同步。host-only、CUDA-event 和外部 profiler 三种开销必须分别测量；带 profiler 的样本不得混入不带 profiler 的吞吐对比。

## 从自定义代码补充语义阶段

### wave/credit 集成路径

集成后的 distributed/colocated updater 会让发送 span 跨越异步 API 返回，
持续到传输完成边界。`mark_api_return()` 仅关闭 API/NVTX range，保留待完成
的 GPU event bracket，避免多个 bucket 的异步提交造成 NVTX 栈错误嵌套。
engine load ACK 等待单独记录为 trainer-side span；CUDA IPC 在 ACK 前不
臆造传输完成。wave ID 保留在 metadata，并加入 logical operation ID，不
冒充新 bucket。

`weight_bucket_reusable` 跟随实际 credit/资源释放，`weight_sync_complete`
跟随最终版本 commit；失败则记录 `weight_sync_failed`，未完成 span 标记
error。转换迭代器耗尽时取消的末尾 probe 会使全局 trace sequence 出现空洞，
它不是 communicator sequence ID。启用/关闭 timeline 的 CPU 组合测试覆盖
2/4 个逻辑 engine group，但不代表完整 Ray/SGLang GPU 或 engine 内核验证。

slime 不会修改 Megatron 内部 MoE 实现。自定义 Megatron hook 或 plugin 可以直接按同一 schema 补充 `ep_dispatch` 与 `ep_combine`：

```python
from slime.observability.communication_timeline import communication_context, communication_phase

with communication_context(global_step=step, rollout_id=rollout_id):
    with communication_phase(
        "ep_dispatch",
        message_bytes=dispatch_bytes,
        transport="nccl",
        layer=layer_id,
    ):
        dispatch_tokens()
```

瞬时边界使用 `communication_event(...)`；观察到第一个 consumer 时，可对 `communication_phase(...)` 返回的对象调用 `mark_consumer()`。

这条时间线面向进程级训练/通信阶段，与按 sample 展示的 [Trace Viewer](./trace.md) 互补。kernel 级分析请把 NVTX range 与[性能分析](./profiling.md)结合使用。
