# 观测

slime 的默认观测路径很简单：训练指标继续进 W&B / TensorBoard；SGLang 的高频 Prometheus metrics 不再上传 W&B；request timing 从 SGLang response `meta_info` 写进 sample trace，并在 rollout 结束时聚合成少量 `perf/...` 指标。

## W&B / TensorBoard 里会看到什么

W&B 和 TensorBoard 仍然记录 reward、loss、KL、entropy、eval 等训练指标。额外的 SGLang request timing 会放在 `perf/` 下，例如：

```text
perf/request/e2e_latency/mean
perf/request/queue_time/median
perf/request/count
perf/request/profiled_count
perf/decode/throughput/mean
perf/prefill/bootstrap_queue_duration/mean
perf/prefill/bootstrap_duration/mean
perf/prefill/alloc_wait_duration/mean
perf/prefill/forward_duration/max
perf/prefill/transfer_speed_gb_s/mean
perf/decode/prealloc_duration/mean
perf/decode/bootstrap_duration/mean
perf/decode/alloc_wait_duration/mean
perf/decode/transfer_duration/max
perf/decode/forward_duration/mean
```

这些指标是每个 rollout step 聚合一次，不是每个 request 上报一次，所以不会像上传完整 Prometheus metrics 那样拖慢 W&B。

不开 PD 时仍然会有通用的 `perf/request/...` 和可用的 `perf/decode/throughput/...`。`perf/prefill/...` 和更细的 `perf/decode/...duration` 只有在 SGLang 返回对应 `pd_*` timing 字段时才会出现。

## Prometheus metrics 存在哪里

slime 自己不存 Prometheus 的每秒数据。SGLang / router 只暴露 `/metrics` 和 `/engine_metrics` HTTP endpoint；Prometheus 定期 scrape 这些 endpoint，并把时间序列写进 Prometheus 自己的 TSDB。

因此：

- 如果没有启动 Prometheus，这些 serving metrics 只存在于 SGLang 进程内存和当前 endpoint 输出里，不会形成历史记录。
- 如果启动了 Prometheus，历史数据存放在 Prometheus 的 `--storage.tsdb.path`。
- slime 不把这些高频 metrics 上传到 W&B。

常用的 SGLang metrics 包括：

```text
sglang:num_queue_reqs
sglang:num_running_reqs
sglang:num_prefill_bootstrap_queue_reqs
sglang:num_prefill_inflight_queue_reqs
sglang:num_decode_prealloc_queue_reqs
sglang:num_decode_transfer_queue_reqs
sglang:kv_transfer_speed_gb_s_bucket
sglang:kv_transfer_latency_ms_bucket
sglang:kv_transfer_total_mb_bucket
```

这些适合在 Grafana / Prometheus 里看实时 queue buildup、transfer speed、latency histogram、失败计数等 serving 状态。

## 如何启动 Prometheus

Prometheus 需要在训练运行时启动，因为它只能 scrape 当前正在暴露的 endpoint，不能在训练结束后从 SGLang endpoint 里补回过去的数据。它不需要放进训练 Python 进程里，推荐作为同一台机器或同一个作业里的旁路进程运行。

最小配置如下，把 `ROUTER_IP:ROUTER_PORT` 换成 slime 日志里的 router 地址，或者用户显式设置的 `--sglang-router-ip` / `--sglang-router-port`：

```yaml
global:
  scrape_interval: 10s

scrape_configs:
  - job_name: slime-sglang
    metrics_path: /engine_metrics
    static_configs:
      - targets:
          - "ROUTER_IP:ROUTER_PORT"
```

启动 Prometheus 时把 TSDB 目录放到持久化路径上：

```bash
prometheus \
  --config.file=/path/to/prometheus.yml \
  --storage.tsdb.path=/path/to/prometheus-data \
  --storage.tsdb.retention.time=7d \
  --web.listen-address=0.0.0.0:9090
```

slime 镜像里已经安装了 `prometheus`，可以直接在容器里用上面的命令启动。也可以从同一个镜像再起一个旁路容器，只要它能访问 router 地址，并把 `/path/to/prometheus-data` 挂到持久化目录即可。

如果 `--storage.tsdb.path` 在容器本地盘里，容器回收后数据也会丢；如果挂到 NFS、持久化卷或作业输出目录，训练结束后可以重新启动 Prometheus 指向同一个 TSDB 目录，再用 Prometheus UI 或 Grafana 查询历史时间段。这里的“回放”是时间序列回放和图表分析，不是 per-request trace 的完整重放；per-sample request timing 仍然走 sample trace / debug rollout 数据。

## Trace viewer

`--save-debug-rollout-data` 保存的 sample trace 会包含 SGLang `meta_info` 里的 timing 字段。trace viewer 直接读取这些 attrs，并用 `pd_*` 字段展示 `[P]` / `[D]` 虚拟 lane。

```bash
python tools/trace_timeline_viewer.py /path/to/debug/rollout_0.pt
```

默认路径不需要单独保存 `ReqTimeStats(...)` 日志，也不需要 Loki 或 compact 工具。
