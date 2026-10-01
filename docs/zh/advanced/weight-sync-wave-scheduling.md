# 权重同步 Engine Wave 调度

大规模 rollout 部署可能让所有 SGLang engine 同时接收或加载同一个权重
bucket。这样可以最大化 fan-out，但也可能瞬间占满 GPU、PCIe、NVLink、网络
或存储带宽。slime 可以在不引入硬件相关固定延迟的情况下限制这种 fan-out：

```bash
--update-weight-max-inflight-engine-groups 2
```

这里的 engine group 指一个逻辑 SGLang engine。它内部的 tensor parallel、
pipeline parallel 或多节点 worker 始终作为不可拆分的整体。调度器按解析后的
engine 列表稳定分波，每波最多放行配置数量。例如四个 engine、上限为三时，
wave 依次为 `(0, 1, 2)` 和 `(3)`；实现不假设特定 world size 或 engine
数量。

默认值为 `0`，保持原有 all-at-once 行为。配置值大于等于 engine 数量时也
等价于默认行为；负数会在启动参数校验阶段被拒绝。

## 支持的传输路径

- 非 colocate NCCL 权重同步只在确实要求并发上限时，为每个逻辑 engine
  创建独立 process group。同一 wave 内的 broadcast 异步发起，NCCL work
  与 engine 侧加载请求全部完成后才放行下一波；默认行为仍使用原来的聚合
  process group。
- Colocate tensor/IPC 同步在所有 trainer rank 间使用相同的确定性 wave；
  每波结束用 trainer 控制组 barrier 封口，然后才允许复用源 buffer。
- Full 和 delta disk 同步会对 checkpoint pull 与 engine reload 请求应用
  同一并发上限。
- 量化模型的 post-load 处理也遵守同一 engine-group 上限。

现有的 pause、flush 与权重版本边界保持不变：某个 engine 不会仅仅因为自己的
wave 先完成就提前恢复 generation。disk checkpoint pull 可以在 pause 前预取，
但该过程不会修改 serving 权重。因此 wave 放行本身不会引入新的新旧权重混用
窗口。

## 如何选择上限

先用 `0` 建立吞吐基线，再在实际部署拓扑上比较 `1`、`2`、`4` 等较小上限。
应选择既能消除观测到的流量或显存峰值、又不会增加权重同步总时间或尾延迟的
最大值。该参数只限制并发 engine group 数量；每个权重 bucket 的大小仍由
`--update-weight-buffer-size` 独立控制。

## 生产调用点确认

### 同时启用 wave 与 bucket credit

集成 updater 支持 wave 与 `--update-weight-max-inflight-buckets`、
`--update-weight-max-inflight-bytes` 同时启用。一个逻辑 bucket 的 credit
跨越该 bucket 的全部 engine wave；每波完成设备传输、engine load ACK 和
staging 释放后才推进，最后一波结束后才返还逻辑 credit。所有消费者恢复后
才发布最终版本。

存在多个 wave 时，缓存窗口中的 bucket 逐个完成全部波次；逻辑字节窗口仍
限制准入量，但不会将 engine 并发上限乘以 bucket 数量，也不承诺跨 bucket
重叠。单个聚合 communicator 保留异步 bucket-window 路径。逻辑字节不是
物理显存硬上限或实际网络线速字节。返还 transport credit 前，在 Work.wait
插入的流依赖后执行主机可观察的流完成等待。load 失败会使版本不可提交、
阻止下一波，并在 updater 上保留当前波的资源；恢复需要重建 updater/进程，
不自动重试部分写入的版本。

CPU 组合测试覆盖两种生产 updater、2/4 个逻辑 engine group、对端 load
失败传播和 staging 保留，不等价于完整 Ray/SGLang GPU 训练验证。

`tools/benchmark_weight_sync_callsite.py` 直接调用生产函数
`update_weights_in_engine_group_waves`，使用真实 process group 和合成 payload。
它是调用点/模块级探针：不会重写调度器、改变默认值，也不声称测到了
Ray/SGLang 加载性能。

四进程启动会保留三个真实的二 rank `[trainer, engine]` 通信组。A/B 是
engine group 0 和 1；第三组是区分 all-at-once 与 window-2 的竞争操作。
只使用当前测试环境允许的设备数量。

下列每条命令只生成一个独立进程运行制品：

```bash
torchrun --standalone --nproc-per-node=4 \
  tools/benchmark_weight_sync_callsite.py \
  --backend gloo --policy candidate_windowed \
  --evidence-role selection --run-id selection-windowed-00 \
  --order ab --output-json /tmp/selection-windowed-00.json
```

若最小 PyTorch 容器缺少 slime 的 Ray/Megatron 控制面依赖，可显式设置
`SLIME_CALLSITE_SOURCE_LOAD=1`。该模式加载完全相同的生产源文件，只替换本
探针不会调用的 actor/conversion import；制品会记录
`source_with_control_plane_stubs`，不能静默混入完整运行时 campaign。
若容器内没有挂载 checkout 的 `.git` 元数据，请用
`SLIME_BENCHMARK_SOURCE_COMMIT` 显式传入被测 revision。

对每个 policy 分别以 `selection`、`confirmation` 角色启动独立进程（默认
每个 policy/role 五次），并交替使用 `--order ab`、`--order ba`。随后校验、
汇总不可变原始制品：

```bash
python tools/benchmark_weight_sync_callsite.py \
  --summarize /tmp/weight-sync-callsite/*.json \
  --min-runs-per-role 5 \
  --summary-json /tmp/weight-sync-callsite-summary.json
```

遇到重复进程/运行身份、rank 覆盖不完整、payload 不一致、混用运行时/
消息/拓扑 cell，或独立运行不足时，汇总器会 fail closed。它分别报告通信
A/B、rank-local pair makespan、接收端 consumer wait、调用点返回、设备
ready 和整个同步阶段 ready。NCCL 区间只标记为 `event_bracket`，绝不冒充
`kernel_observed`。工具记录 PyTorch/CUDA/NCCL 版本、launch-order 配置、
dtype、消息几何、graph 状态、主机/设备身份与精确 PG membership。

这些证据不能证明端到端训练吞吐、SGLang 加载延迟、多机行为或某个生产
策略必胜；汇总器不会自动选择或应用策略。
