# 权重同步 Engine Wave 调度

大规模 rollout 部署可能让所有 SGLang engine 同时接收或加载同一个权重
bucket。这样可以最大化 fan-out，但也可能瞬间占满 GPU、PCIe、NVLink、网络
或存储带宽。slime 可以在不引入硬件相关固定延迟的情况下限制这种 fan-out：

```bash
--update-weight-max-inflight-engine-groups 2
```

这里的 engine group 指一个逻辑 SGLang engine。它内部的 tensor parallel、
pipeline parallel 或多节点 worker 始终作为不可拆分的整体。调度器按解析后的
engine 列表稳定分波，每波最多放行配置数量。例如五个 engine、上限为二时，
wave 依次为 `(0, 1)`、`(2, 3)` 和 `(4)`；实现不假设特定 world size 或
engine 数量。

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
