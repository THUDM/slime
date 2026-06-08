# Observability Bundle 设计

slime 的 observability 设计是一个 per-run 本地 bundle。它的目标是让长时间 RL 任务在训练中和崩溃后都容易检查，同时不要把高频 SGLang serving metrics 上传到 W&B，也不要把 Prometheus、Grafana、Loki、OTel 这些后端 SDK 放进 trainer 的关键路径。

当前默认只有一个生产模式：`--enable-observability`。这个模式会收集对 profiling 最有用、同时开销可控的本地 artifacts。

- W&B 继续记录算法指标，并额外记录从 `meta_info` / sample trace 聚合出来的少量 request perf 统计，例如 P/D duration、transfer speed、retry count。
- Prometheus 同时从 SGLang router 的 `/metrics` 和 `/engine_metrics` 抓 serving/system 指标，分别覆盖 router 自身指标和 router 聚合的 engine 指标。
- SGLang response `meta_info` 中的 request timing 字段写入 sample trace；trace viewer 直接用其中的 `pd_*` 字段恢复 PD prefill/decode timeline。
- slime 生成 run metadata、Prometheus 配置、file-based service discovery 文件，以及 fail-open 的本地 status/component_state 文件。
- slime 不自动打开 token-level trace、payload dump、远端日志系统或 replay。
- slime 默认不向训练主循环注入 step/weight-sync/request lifecycle 事件；这类 trainer/rollout coupling telemetry 后续应通过显式、非侵入的 hook/helper 接口接入。
- trainer 进程不负责启动或管理 Prometheus、Grafana、Loki、OTel。面向普通用户时，可以在 slime 镜像里内置本机 helper，由 helper 在同一台机器或同一个容器里启动这些标准工具。

## 为什么不要把 SGLang metrics 传到 W&B

SGLang Prometheus metrics 是高频系统指标，直接通过 W&B 上传会让 W&B 变卡，也会把不同类型的数据混在一起。slime 现在明确分层：

| 数据类型 | 目标系统 | 默认状态 | 说明 |
| --- | --- | --- | --- |
| 算法指标 | W&B / TensorBoard | 开启方式不变 | reward、loss、KL、entropy、eval |
| SGLang request perf 统计 | W&B / TensorBoard | rollout perf 聚合时写入 | 从 sample trace 中的 timing 字段汇总，低频低基数 |
| SGLang 系统指标 | Prometheus | `--enable-observability` 后可 scrape | 从 router `/metrics` 和 `/engine_metrics` 抓取 |
| Run metadata | 本地 JSON | `--enable-observability` 后写入 | run id、路径、环境、版本、allowlist 参数 |
| Component status | 本地 JSON/JSONL | `--enable-observability` 后写入 | `status.json`、`component_state.json`、`errors.jsonl` |
| Traces / replay dumps | TODO | 默认不开 | 后续必须显式 opt-in |

这条边界很重要：监控系统挂了，训练不能挂。

## 会不会影响训练或推理速度

默认 `--enable-observability` 的开销应该很低，但不是数学意义上的 0。

- bundle 文件只在启动和 router 注册时写入，和 rollout 请求热路径无关。
- Prometheus 每 5 秒 scrape 一次 router `/metrics` 和 `/engine_metrics`。这是 HTTP 拉取，不在 trainer 进程里运行，也不会同步阻塞 rollout。
- SGLang metrics endpoint 本身需要维护计数器和 histogram。slime 当前一直开启 SGLang metrics，因为 router `/engine_metrics` 要依赖它。这个开销通常远小于模型推理。
- SGLang `meta_info` 里的 request timing 已经随 response 回到 Python 侧。slime 在 sample trace 中保存这些字段，并在 rollout 完成时只上报 `perf/...` 聚合统计到 W&B/TensorBoard，不上报每条 request。
- 默认不写 SGLang request metrics JSONL、request logs 或 `ReqTimeStats(...)` engine logs，也不会让这些内容进入用户日常 stdout。
- 如果 Prometheus 没启动，训练也会继续跑。bundle 和 target 写失败也会 fail open。

日常 profiling 的主数据源是 SGLang response `meta_info` 里的 request timing 字段。它们会进入 sample trace，trace viewer 用 `pd_*` 字段画 `[P]` / `[D]` lane；W&B/TensorBoard 只收到每个 rollout step 的聚合值，例如 `perf/prefill/forward_duration/mean`、`perf/decode/transfer_duration/max`、`perf/request/queue_time/median`。

所以 slime 的日常生产默认是：

```text
生产 profiling 默认 = manifest + status + Prometheus config + file_sd + /metrics scrape target + /engine_metrics scrape target + sample trace timing attrs + W&B/TensorBoard perf/* 聚合
```

这个模式把高频 SGLang Prometheus metrics 留给 Prometheus，把少量 request perf 汇总给 W&B/TensorBoard。

## 用户接口

启用方式：

```bash
python train.py \
  ... \
  --enable-observability \
  --run-dir /mnt/runs/ppo_qwen3_001
```

也可以用环境变量：

```bash
export SLIME_ENABLE_OBSERVABILITY=1
export SLIME_RUN_DIR=/mnt/runs/ppo_qwen3_001
export SLIME_PROMETHEUS_TSDB_DIR=/local_nvme/prometheus/{run_id}
```

如果没有传 `--run-id`，slime 会自动生成。如果没有传 `--run-dir`，默认是：

```text
/tmp/slime-runs/{run_id}
```

旧的 `--observability-profile` 已废弃并隐藏。`basic/debug/replay` 不再作为日常用户接口，因为实际训练应该只有一个生产 profiling 配置。

## 生成的目录结构

例如 `--run-dir /mnt/runs/ppo_qwen3_001` 会生成：

```text
/mnt/runs/ppo_qwen3_001/
  manifest.json
  observability/
    status.json
    component_state.json
    errors.jsonl
  prometheus/
    prometheus.yml
    file_sd/
      sglang_router_metrics.json
      sglang_router_engine_metrics.json
```

`manifest.json` 记录 run id、关键路径、环境信息、组件版本，以及 allowlist 后的 slime 参数。默认不把完整 argparse namespace 写进 manifest；key、token、secret、password、authorization、credential 这类字段即使出现在 allowlist 中也会被打码。

生成的 Prometheus 配置指向当前 run 的 file service discovery 文件：

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 15s

scrape_configs:
  - job_name: slime_sglang_router
    metrics_path: /metrics
    file_sd_configs:
      - files:
          - /mnt/runs/ppo_qwen3_001/prometheus/file_sd/sglang_router_metrics.json

  - job_name: slime_sglang_engine_aggregated
    metrics_path: /engine_metrics
    file_sd_configs:
      - files:
          - /mnt/runs/ppo_qwen3_001/prometheus/file_sd/sglang_router_engine_metrics.json
```

SGLang router 启动后，slime 会写入两个 scrape target 文件。它们通常指向同一个 router 地址，但 label 中的 `metrics_endpoint` 不同：

```json
[
  {
    "targets": ["10.0.0.1:3456"],
    "labels": {
      "component": "sglang",
      "metrics_endpoint": "router",
      "role": "router",
      "run_id": "..."
    }
  }
]
```

## 本机启动 Prometheus

最小可用流程不需要单独部署服务。slime 镜像里可以直接安装 Prometheus，用户只需要给任务一个持久化 `run_dir`，然后用 run-local config 在同一台机器上启动：

```bash
prometheus \
  --config.file=/mnt/runs/ppo_qwen3_001/prometheus/prometheus.yml \
  --storage.tsdb.path=/local_nvme/prometheus/ppo_qwen3_001 \
  --web.listen-address=:19090
```

建议把 `--storage.tsdb.path` 放在本地盘上，例如 `SLIME_PROMETHEUS_TSDB_DIR` 指向的路径。Prometheus local storage 更适合 POSIX-like local storage；NFS/EFS 这类共享文件系统可能很慢，也可能不适合 TSDB 写入。

Docker 镜像里安装了 Prometheus，并放了一个 all-runs 示例配置：

```text
/etc/prometheus/slime-example-prometheus.yml
```

单个任务最推荐的还是用 run 目录里生成的 `prometheus/prometheus.yml`。

后续更适合普通用户的形态是提供一个镜像内 helper，例如：

```bash
slime-observability local \
  --run-dir /mnt/runs/ppo_qwen3_001 \
  --prometheus-port 19090
```

这个 helper 只是把标准组件作为本机辅助进程启动，例如 Prometheus、可选 Grafana、可选日志 tailer；训练进程仍然只写本地 bundle 和 profiling 文件。这样用户不需要懂 Kubernetes、Helm 或单独部署 Loki/Prometheus，但也不会把远端日志发送、查询后端 SDK 或重试逻辑放进 rollout 热路径。

需要注意：只在镜像里安装依赖不能解决持久化。如果容器文件系统或节点本地盘会在任务失败后被回收，`run_dir` 必须挂到持久路径，例如 NFS/并行文件系统、PVC/hostPath，或者由本机日志 tailer 及时送到 Loki、ClickHouse、对象存储等后端。

## Request perf metrics 如何工作

SGLang response `meta_info` 会提供不开 PD 也有的 request timing，例如：

```text
queue_time
e2e_latency
decode_throughput
```

PD disaggregation 下，SGLang patch 还会把 request time stats 中对训练 profiling 最有用的字段放进 response `meta_info`，例如：

```text
pd_prefill_bootstrap_queue_duration
pd_prefill_forward_duration
pd_prefill_transfer_queue_duration
pd_decode_prealloc_duration
pd_decode_transfer_duration
pd_decode_forward_duration
pd_bootstrap_duration
pd_alloc_waiting_duration
pd_transfer_speed_gb_s
pd_transfer_total_mb
pd_prefill_retry_count
```

slime 在 `sglang_generate` span 结束时把这些字段写进 sample trace。rollout 汇总时，`slime/ray/rollout.py` 会从 sample trace 中抽取所有 `sglang_generate` span，并把内部 `pd_*` 字段映射成对用户更清楚的 `perf/prefill/...`、`perf/decode/...`、`perf/request/...`。每个字段会上报 `mean`、`median`、`max`、`min` 和 `count`：

```text
perf/request/e2e_latency/mean
perf/request/queue_time/median
perf/prefill/bootstrap_queue_duration/mean
perf/prefill/forward_duration/max
perf/prefill/transfer_speed_gb_s/mean
perf/decode/prealloc_duration/mean
perf/decode/transfer_duration/max
perf/decode/forward_duration/mean
perf/request/count
perf/request/profiled_count
```

这些指标会走现有 `logging_utils.log()`，所以 W&B 和 TensorBoard 都能看到。它们是每个 rollout step 的聚合统计，不是每个 request 一条；因此不会像上传 Prometheus 原始 metrics 那样把 W&B 卡住。不开 PD 时，`perf/request/...` 和可用的 `perf/decode/throughput/...` 仍然会存在；`perf/prefill/...` 和 `perf/decode/...` 的细分 duration 只在 SGLang 返回对应 timing 字段时出现。

## Prometheus 能看到哪些 P/D queue

Prometheus 负责 live serving 状态。当前 SGLang `/engine_metrics` 里已经有 PD disaggregation 的队列 gauge，适合看某个时间点的拥塞和积压：

```text
sglang:num_queue_reqs
sglang:num_running_reqs
sglang:num_prefill_bootstrap_queue_reqs
sglang:num_prefill_inflight_queue_reqs
sglang:num_decode_prealloc_queue_reqs
sglang:num_decode_transfer_queue_reqs
```

这些指标带有低基数 label，例如 `engine_type`、`dp_rank`、`tp_rank`、`pp_rank`、`model_name`。因此可以在 Grafana 里分别看 prefill engine 和 decode engine 的 queue buildup。

Prometheus 也能看一部分请求阶段 latency 和 KV transfer 统计：

```text
sglang:per_stage_req_latency_seconds_bucket{stage="prefill_bootstrap"}
sglang:per_stage_req_latency_seconds_bucket{stage="prefill_transfer_kv_cache"}
sglang:per_stage_req_latency_seconds_bucket{stage="decode_prepare"}
sglang:per_stage_req_latency_seconds_bucket{stage="decode_bootstrap"}
sglang:per_stage_req_latency_seconds_bucket{stage="decode_transferred"}
sglang:kv_transfer_speed_gb_s_bucket
sglang:kv_transfer_latency_ms_bucket
sglang:kv_transfer_total_mb_bucket
sglang:num_prefill_retries_total
sglang:num_bootstrap_failed_reqs_total
sglang:num_transfer_failed_reqs_total
```

这里和 W&B/TensorBoard 的 `perf/...` 指标是互补关系：Prometheus 看在线 queue depth、histogram 和错误计数；sample trace / W&B 聚合看已经完成请求的 min/max/mean/median，以及 trace viewer 能重建的 P/D timeline。

## Trace viewer 如何读取 duration

trace viewer 默认直接读取 debug rollout dump 里 sample trace 的 `pd_*` attrs，并用它们画 `[P]` / `[D]` 虚拟 lane。也就是说，主路径不需要单独保存 `ReqTimeStats(...)` 日志，不需要 Loki，也不需要 compact request-time-stats 文件。

后续要做更像 Grafana 的分析，可以优先在 trace viewer 里加原生面板：按 request duration 做 histogram、p95/p99、`input_len` vs transfer duration scatter、transfer speed 时间序列等。这样用户只需要打开 trace viewer，不需要额外部署日志系统。

## slime 自身 telemetry

SGLang metrics 能回答 serving 层面的很多问题，但不能完整回答 trainer 是在等 rollout、weight sync、data buffer，还是 Ray/object store。这个层面的 telemetry 很有价值，但默认实现不应该直接改写训练主循环。

v1 只在 observability bundle 边界内做低侵入改动：生成标准配置、注册 scrape target、写 status。后续如果要补 slime 自身的 step/rollout/weight-sync/request lifecycle 事件，推荐通过显式的轻量接口接入，例如：

- 使用已有 `Timer` / trace 工具导出低频 phase summary。
- 在 rollout/training 已有 log/perf 汇总点旁边增加可关闭的本地 writer。
- 对自定义 rollout 函数提供 opt-in correlation headers 或 hook，而不是默认包裹每个请求。
- 所有 slime-local events 都写本地 JSONL 或 textfile，fail open，不引入远端 SDK。

这样可以补上 RL 系统因果链，同时不把 observability 变成训练流程的硬依赖。

## RDMA 和通信速度

当前默认采集能看到一部分和通信相关的 request-level 症状，但还不能完整回答 RDMA 硬件层问题。

Prometheus scrape 的 `/engine_metrics` 能看到 SGLang serving 层面的现象，例如 P/D queue buildup、TTFT、decode/prefill latency、cache、吞吐、KV transfer speed/latency/size histogram、retry 和失败计数等。sample trace 里的 `pd_*` 字段还能在 rollout 维度汇总 transfer duration、transfer speed、transfer total MB、retry 等完成请求信息。如果 RDMA 或网络有问题，这些指标会更容易暴露症状。

但 RDMA 带宽、重传、packet drop、NIC 错误计数、link state 仍然需要 node-level counters。要看这些，需要后续增加低频、带外、fail-open 的采集：

- 每 10 到 30 秒读取 `rdma` / `ibstat` / `perfquery` / sysfs counters。
- 采集 NIC bytes、packets、errors、discards、retransmit、link state、link speed。
- 写到本地 `network/rdma/*.jsonl`，或通过 node exporter textfile collector 暴露给 Prometheus。
- 不在 rollout request 路径里调用这些命令。
- 采样失败只记录 warning，不影响训练。

这样可以在 dashboard 里把 SGLang latency/throughput、request-level transfer stats 和 RDMA counters 放在同一时间轴上看。

## 运行规则

后续扩展这个 bundle 时要守住这些规则：

- 不要把 SGLang Prometheus metrics 上传到 W&B。
- 默认模式不要写 request-level profiling 文件、不要写 prompt/output payload、不要写 stdout、不要同步发送远端日志。
- 不要把 `request_id`、`sample_id`、prompt hash、raw prompt、raw output、user id 放进 Prometheus labels。
- Prometheus labels 只放低基数字段，例如 `run_id`、`role`、`node`、`rank`、`worker_id`、`model_name`、`phase`。
- 高基数字段放 JSONL、Parquet 或 trace attributes，用来 join，不要做 label。
- Observability 必须 fail open。bundle 创建失败或 target 写失败时，训练继续跑。
- trainer 不拥有 Prometheus/Grafana/Loki 生命周期。slime 生成标准文件；镜像内 helper 可以把这些标准工具作为本机辅助进程启动，方便不熟悉云原生的用户。

## 当前代码位置

- `slime/profiling/observability.py`：负责 bundle 创建、manifest 写入、Prometheus config 渲染、file service discovery target 写入。
- `slime/utils/arguments.py`：提供 `--enable-observability`、`--run-id`、`--run-dir`、`--observability-prometheus-tsdb-dir` 参数。
- `train.py` 和 `train_async.py`：在 tracking 之前初始化 bundle，在 rollout server 启动后注册 router target。
- `slime/utils/wandb_utils.py`：不再 reinit W&B 来上传 open metrics endpoints。
- `slime/utils/logging_utils.py`：只有 `--use-wandb` 时才 lazy import W&B。
- `slime/backends/sglang_utils/sglang_engine.py`：强制打开 SGLang metrics，保证 router `/metrics` 和 `/engine_metrics` 可被 Prometheus scrape。
- `slime/utils/trace_utils.py`：把 `meta_info["id"]` 保存为 `sglang_request_id`，并把 `meta_info` 里的 request timing 字段保存到 sample trace。
- `slime/ray/rollout.py`：从 sample trace 聚合 request / prefill / decode perf 指标，并通过现有 W&B/TensorBoard logging 路径上报 min/max/mean/median/count。
- `tools/trace_timeline_viewer.py`：直接读取 sample trace 里的 `pd_*` 字段，展示 `[P]` / `[D]` 虚拟 lane。

## TODO

短期：

- 给 `prepare_observability_args`、manifest redaction、Prometheus config 渲染、file service discovery 输出、rollout request perf 聚合补单测。
- 在 quick start 或 profiling 文档里加一个最小使用例子。
- 增加 `slime-observability local` helper：读取 `run_dir` 和 manifest 中的 `prometheus_tsdb_dir`，在本机启动 Prometheus，写 pid/log，退出时能清理辅助进程。
- 确认不同规模 rollout 下 Prometheus scrape、SGLang metrics enable、sample trace request perf 聚合的开销，用 micro benchmark 写进文档。

中期：

- 增加 RDMA / network node-level sampler，低频采集 counters，默认 fail open，并支持 node exporter textfile collector。
- 生成 Grafana provisioning 和基础 dashboards，默认指向 run-local Prometheus。
- 如果后续使用 SGLang tokenizer metric custom labels，加低基数 allowlist。
- 给 rollout 请求加 correlation headers，例如 `x-slime-run-id`、`x-slime-rollout-id`、`x-slime-weight-version`、request id。这些字段用于 join logs/traces/request facts，不应该直接作为 Prometheus labels。
- 增加 slime 自己的轻量 metrics endpoint 或本地 metrics 文件，记录 rollout latency、weight sync latency、data buffer size、global step 等 training/rollout coupling 指标；这部分需要显式 opt-in 或挂在已有汇总点上，避免侵入训练主循环。

长期：

- 可选接入本机 Loki 或远端 Loki / ClickHouse / 对象存储，给需要日志检索和长期保存的用户使用，但默认 artifact 仍然是本地文件。
- 可选接入 OpenTelemetry tracing，必须显式 opt-in，并带采样、脱敏、retention。
- 生成标准 `docker-compose.yaml` 作为启动 Prometheus/Grafana/Loki 的 convenience wrapper；主路径仍然是 slime 镜像内 helper，避免要求用户掌握额外部署系统。
- 增加能回答 RL-serving 问题的 dashboard：queue 堆积、TTFT spike、cache hit rate 变化、prefill/decode 负载、weight sync pause、worker hot spot、request-level transfer 慢点、RDMA link/counter 异常。
