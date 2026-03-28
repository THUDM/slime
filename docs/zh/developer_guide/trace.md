# Trace 可视化

slime 可以为每条 rollout sample 挂上轻量级执行 trace。它会记录生成、奖励模型等 span 事件，并且可以在保存下来的 rollout debug dump 中离线查看。

![trace 时间线查看器](../../_static/image/trace.png)

## 保存 rollout trace 数据

如果想在运行结束后查看 trace，可以在训练时打开 rollout debug dump：

```bash
python train.py \
    ... \
    --save-debug-rollout-data /path/to/debug/rollout_{rollout_id}.pt
```

每个保存出来的 `.pt` 文件都会包含 rollout samples，以及对应的 `trace` 数据。之后也可以通过 `--load-debug-rollout-data` 复用同一份 dump。

## 打开时间线查看器

对保存好的 rollout dump 运行：

```bash
python tools/trace_timeline_viewer.py /path/to/debug/rollout_0.pt
```

脚本会生成：

- `rollout_0.trace_timeline_cache.json`
- `rollout_0.trace_timeline_viewer.html`

默认情况下，它还会启动一个本地静态文件服务，方便直接在浏览器里打开。如果只想生成文件，可以加 `--no-serve`。

## 如何理解可视化结果

- 每一行对应一条 sample。
- 条形块表示 span，点表示瞬时事件。
- `trace_span(...)` 在开始和结束时记录的属性，都会显示在详情面板里。
- 当 SGLang 返回 PD 分离相关时延时，viewer 会自动补出 `[P]` 和 `[D]` 两条虚拟 lane，用来拆开展示 prefill/decode。
- 如果没有开启 PD，这两条虚拟 lane 不会出现，基础 trace 也仍然可以正常渲染。

## 给自定义代码打点

在自定义 rollout 或 reward 逻辑中，可以直接复用 `slime.utils.trace_utils` 里的工具：

- `trace_span(target, name, attrs=...)`：记录一段持续时间。
- `trace_event(target, name, attrs=...)`：记录一个瞬时事件。
- `bind_trace(sample)`：在 sample 被传递到其他 helper 或任务之前，确保它已经绑定好 trace carrier。

如果想统一记录 SGLang 返回的 generation 元信息，可以复用 `build_sglang_meta_trace_attrs`：

```python
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_span

with trace_span(sample, "sglang_generate") as span:
    output = await post(url, payload)
    span.update(build_sglang_meta_trace_attrs(output["meta_info"]))
```

## 使用建议

- 先保存少量 rollout；单个 dump 的 sample 数量适中时，viewer 会更容易阅读。
- viewer 直接基于保存下来的 `.pt` dump 工作，因此可以把文件拷到别的机器离线分析。
- 如果你想看的是 SGLang 自身的 GPU / kernel 级 profiling trace，请参考 [性能分析](./profiling.md)。

