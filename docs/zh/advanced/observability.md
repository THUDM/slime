# Observability Bundle 设计

slime 的 observability 设计是一个 per-run 本地 bundle。它的目标是让长时间 RL 任务在训练中和崩溃后都容易检查，同时不要把高频 SGLang serving metrics 上传到 W&B，也不要把 Prometheus、Grafana、Loki、OTel 这些后端 SDK 放进 trainer 的关键路径。

当前默认只有一个生产模式：`--enable-observability`。这个模式会收集对 profiling 最有用、同时开销可控的本地 artifacts。

- W&B 只继续记录算法指标，例如 loss、reward、KL、entropy、eval。
- Prometheus 同时从 SGLang router 的 `/metrics` 和 `/engine_metrics` 抓 serving/system 指标，分别覆盖 router 自身指标和 router 聚合的 engine 指标。
- SGLang request metrics 写到本地 JSONL 文件，每个完成请求一条，用来重建 request-level profiling 状态。
- SGLang request logs 以 `level=0` metadata-only 形式写到本地文件目录，不写到用户日常 stdout。
- SGLang `ReqTimeStats(...)` 日志写到本地 engine 日志文件，trace viewer 可以读取这些日志来恢复 PD prefill/decode timeline。
- slime 生成 run metadata、Prometheus 配置、file-based service discovery 文件，以及 fail-open 的本地 status/component_state 文件。
- 高频 request-level artifacts 默认进入 node-sharded scratch 路径，而不是直接混进 `run_dir` 根目录。
- slime 不自动打开 token-level trace、payload dump、远端日志系统或 replay。
- slime 默认不向训练主循环注入 step/weight-sync/request lifecycle 事件；这类 trainer/rollout coupling telemetry 后续应通过显式、非侵入的 hook/helper 接口接入。
- trainer 进程不负责启动或管理 Prometheus、Grafana、Loki、OTel。面向普通用户时，可以在 slime 镜像里内置本机 helper，由 helper 在同一台机器或同一个容器里启动这些标准工具。

## 为什么不要把 SGLang metrics 传到 W&B

SGLang Prometheus metrics 是高频系统指标，直接通过 W&B 上传会让 W&B 变卡，也会把不同类型的数据混在一起。slime 现在明确分层：

| 数据类型 | 目标系统 | 默认状态 | 说明 |
| --- | --- | --- | --- |
| 算法指标 | W&B / TensorBoard | 开启方式不变 | reward、loss、KL、entropy、eval |
| SGLang 系统指标 | Prometheus | `--enable-observability` 后可 scrape | 从 router `/metrics` 和 `/engine_metrics` 抓取 |
| Run metadata | 本地 JSON | `--enable-observability` 后写入 | run id、路径、环境、版本、allowlist 参数 |
| Component status | 本地 JSON/JSONL | `--enable-observability` 后写入 | `status.json`、`component_state.json`、`errors.jsonl` |
| Request facts | 本地 JSONL | `--enable-observability` 后写入 | SGLang `export_metrics_to_file`，每个完成请求一条 |
| Request logs | 本地 JSON log | `--enable-observability` 后写入 | metadata-only，文件 target，不进 stdout |
| Request time stats | 本地 engine log | `--enable-observability` 后写入 | SGLang `ReqTimeStats(...)`，给 profiling 和 trace viewer 使用 |
| Traces / replay dumps | TODO | 默认不开 | 后续必须显式 opt-in |

这条边界很重要：监控系统挂了，训练不能挂。

## 会不会影响训练或推理速度

默认 `--enable-observability` 的开销应该很低，但不是数学意义上的 0。

- bundle 文件只在启动和 router 注册时写入，和 rollout 请求热路径无关。
- Prometheus 每 5 秒 scrape 一次 router `/metrics` 和 `/engine_metrics`。这是 HTTP 拉取，不在 trainer 进程里运行，也不会同步阻塞 rollout。
- SGLang metrics endpoint 本身需要维护计数器和 histogram。slime 当前一直开启 SGLang metrics，因为 router `/engine_metrics` 要依赖它。这个开销通常远小于模型推理。
- SGLang request metrics exporter 只在请求完成时写一条 JSONL，不是每个 token 写一条。写入通过 `asyncio.create_task` 触发，内部格式化 JSON、写文件并 flush。
- SGLang request logger 会在 request receive/finish 时写 metadata-only JSON log。slime 把 target 指向本地文件目录，所以用户日常 stdout 看不到这些日志。
- SGLang request-time-stats logging 只在请求完成时写一条 `ReqTimeStats(...)`。slime 给本地 engine 注入 run-local logging config，让这些 SGLang 子进程日志进入 scratch 目录下的 `request_time_stats/sglang`，不依赖 Ray driver stdout 的 repeated 聚合。
- 如果 Prometheus 没启动，训练也会继续跑。bundle 和 target 写失败也会 fail open。

request metrics 和 request-time-stats log 分工不同。request metrics JSONL 记录 request 参数和最终 `meta_info`，适合崩溃后做 request facts 查询；`ReqTimeStats(...)` 日志直接来自 prefill/decode engine，包含 queue、forward、PD transfer、transfer speed、retry 等 duration，是重建 PD timeline 和 trace viewer `[P]` / `[D]` lane 的主数据源。

所以 slime 的日常生产默认是：

```text
生产 profiling 默认 = manifest + status + Prometheus config + file_sd + /metrics scrape target + /engine_metrics scrape target + request metrics JSONL + metadata-only request logs + ReqTimeStats engine logs
```

这个模式把路径语义拆开：`run_dir` 放低频、持久 metadata/config/status；`observability_scratch_dir` 放高频 request-level 文件；`observability_prometheus_tsdb_dir` 给 Prometheus local TSDB；`observability_export_dir` 放 compact 后的持久输出。高 QPS、慢盘、共享文件系统、日志量很大或磁盘快满时，request-level 文件仍可能影响 tail latency 或吞吐。后续 TODO 会补 benchmark、buffering/rotation，以及更窄的 SGLang request metrics privacy option。

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
export SLIME_OBS_SCRATCH_DIR=/local_nvme/slime-obs/{run_id}/node={hostname}
export SLIME_PROMETHEUS_TSDB_DIR=/local_nvme/prometheus/{run_id}
export SLIME_OBS_EXPORT_DIR=/mnt/runs/ppo_qwen3_001/export
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
  logging_configs/
    sglang/
      {hostname}_{worker_type}_rank{rank}.json
  export/
  nodes/
    node={hostname}/
      request_metrics/
        sglang/
          sglang-request-metrics-YYYYMMDD_HH.log
      request_time_stats/
        sglang/
          worker_type=prefill/rank=0/pid=1234.jsonl
          worker_type=decode/rank=0/pid=5678.jsonl
      logs/
        sglang/
          ...
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

## request metrics 如何工作

slime 在 `--enable-observability` 下自动注入这些 SGLang 参数：

```text
sglang_export_metrics_to_file = true
sglang_export_metrics_to_file_dir = ${observability_scratch_dir}/request_metrics/sglang
sglang_log_requests = true
sglang_log_requests_level = 0
sglang_log_requests_format = json
sglang_log_requests_target = [${observability_scratch_dir}/logs/sglang]
sglang_enable_request_time_stats_logging = true
```

`export_metrics_to_file` 是 request facts artifact。它在每个完成请求输出时写一条 JSONL，字段来自 request 参数和最终 `meta_info`。它适合做 request 级查询和崩溃后取证。

`enable_request_time_stats_logging` 是 PD profiling 和 trace viewer 的主 artifact。SGLang 在 prefill/decode engine 请求完成时写类似下面的日志：

```text
ReqTimeStats(rid=..., bootstrap_room=..., input_len=..., output_len=..., type=decode): prealloc_queue_duration(...ms) = bootstrap(...ms) + alloc_wait(...ms); transfer_duration=...ms; queue_duration=...ms, forward_duration=...ms
```

slime 会给本地 SGLang engine 生成 `SGLANG_LOGGING_CONFIG_PATH` 指向的 logging config，把这些日志写进 `${observability_scratch_dir}/request_time_stats/sglang`。它不从 Ray driver stdout 读日志，因为 stdout 可能被聚合、截断或显示成 repeated 形式。

默认落盘是每个 engine 进程一个 append-only 分片文件，例如：

```text
${observability_scratch_dir}/request_time_stats/sglang/
  worker_type=prefill/rank=0/pid=1234.jsonl
  worker_type=decode/rank=0/pid=5678.jsonl
```

每个进程独占自己的 shard，避免同一节点上 prefill/decode engine 并行 append 一个文件。文件数增加后由带外 compact 工具处理，而不是在请求完成路径里合并。日志行仍然带 `worker_type` 和 `rank`，文件路径里也有这些维度，方便后处理。

如果任务挂掉后容器或本地盘会被回收，不要只依赖节点本地盘。`run_dir` 可以指向共享 NFS/并行文件系统，但高频 request-level artifacts 应优先进入 `observability_scratch_dir`，再由日志 agent 或 compact/export 工具带外同步到持久位置。另一种做法是写节点本地盘，再由日志 agent tail 到 Loki/ClickHouse/object storage；这时本地分片只是短期 buffer。

不建议在训练热路径里让所有节点直接写同一个全局文件：跨节点共享文件 append 的原子性、顺序和性能都依赖文件系统，NFS/EFS 这类路径尤其容易变成瓶颈。需要一个文件分析时，用 compact 工具在带外生成：

```bash
python tools/compact_sglang_request_time_stats.py \
  /local_nvme/slime-obs/ppo_qwen3_001/node=node-a/request_time_stats/sglang \
  --output /mnt/runs/ppo_qwen3_001/export/sglang_request_time_stats.jsonl
```

trace viewer 会按 `sglang_request_id` / `rid` 读取并合并这些日志；如果旧 trace dump 里已经有 `pd_*` 字段，则优先使用旧字段，保证兼容。

这里同时打开 `sglang_log_requests_level=0` 是当前 SGLang request metrics exporter 的安全边界：它仍会复用 request logger 的 skip list；如果不打开 metadata-only logger，exporter 可能把更宽的 request 参数写入 JSONL。slime 侧不再把完整 argparse namespace 写入 manifest，而是使用 allowlist，并在 manifest 中记录 effective request metrics privacy mode。后续如果 SGLang upstream 提供 exporter 自己的 allowlist/privacy 参数，slime 应切到那个参数，减少对 request logger 副作用的依赖。

request logs 只是辅助 artifact，并且 target 被设置成本地文件目录，不包含 `stdout`。这些文件仍然可能包含 request metadata，所以 run 目录应该按训练数据来保护。

如果使用 external rollout engines，slime 只能写 Prometheus target，不能改外部服务进程的 SGLang 参数或日志落盘方式。需要在外部 engine 启动参数里同样设置上面的 `--export-metrics-to-file`、`--log-requests-level 0`、文件 target、`--enable-request-time-stats-logging`，并把 SGLang `ReqTimeStats(...)` 日志保存到可被 trace viewer 读取的目录或 compacted JSONL 文件。

## Trace viewer 如何读取新日志

SGLang 会为请求生成随机 `rid`，response 的 `meta_info["id"]` 和 `ReqTimeStats(rid=...)` 日志都会使用这个 id。slime 在收到 response 后把 `meta_info["id"]` 保存为 span attribute `sglang_request_id`。`tools/trace_timeline_viewer.py` 在生成 cache 时会读取 request-time-stats 日志，按日志里的 `rid` 合并回对应的 `sglang_generate` span，然后沿用原来的 `[P]` / `[D]` 虚拟 lane 展示逻辑。

常用方式：

```bash
python tools/trace_timeline_viewer.py \
  /path/to/debug/rollout_0.pt \
  --request-time-stats-path /mnt/runs/ppo_qwen3_001/request_time_stats/sglang
```

如果没有显式传路径，viewer 会依次尝试 `SLIME_REQUEST_TIME_STATS_PATH`、`SLIME_RUN_DIR/request_time_stats/sglang`，以及 `.pt` 文件附近的 `request_time_stats/sglang`。

`--request-time-stats-path` 可以指向目录，也可以指向 compact 后的单个 JSONL/log 文件。

latest SGLang patch 不再为了 trace viewer 在 P/D 之间额外传输 prefill timing buffer。viewer 同时支持两种来源：trace dump 里已有的 `pd_*` 字段，以及 request-time-stats 日志或 compacted JSONL 中按 `rid` 合并出的字段。SGLang `ReqTimeStats(...)` 文本是兼容输入；如果 SGLang 以后直接写结构化 JSONL，loader 也可以直接读取。

## 日志后端怎么选

Loki 是适合存日志的系统，尤其适合“按 run、node、worker_type、时间范围查日志”。它不适合作为 SGLang 请求热路径里的同步写入目标。推荐方式是：

```text
SGLang ReqTimeStats
  -> 每 engine 进程 append-only 分片（本地短期 buffer 或 node-sharded scratch）
  -> Promtail / Grafana Alloy / Vector / Fluent Bit tail
  -> Loki
```

这样不需要在共享盘里长期保存大量碎日志，Loki 或 agent 挂了也不会阻塞训练/推理。agent 可以配置 queue、batch、retry、drop policy 和 retention；如果使用易失本地盘，agent 就是持久化链路的一部分。`rid`、`bootstrap_room` 这类高基数字段不要做 Loki label，应该留在日志 body 里。Loki labels 只放低基数字段，例如：

```text
run_id, component=sglang, log_type=req_time_stats, node, worker_type
```

如果目标是做 request duration、transfer speed、retry、P/D breakdown 的大规模统计，Loki 可以用，但不是最强项。更适合的后端是：

| 后端 | 适合场景 | 说明 |
| --- | --- | --- |
| Loki | 日志检索、按时间范围看原始行、和 Grafana dashboard 联动 | 不要把 `rid` 做 label；适合作为集中日志后端 |
| ClickHouse | 大规模结构化 profiling 查询 | 适合按 request 聚合 duration、p99、transfer speed、worker hotspot |
| Object storage + Parquet | 低成本长期归档和离线分析 | 例如 S3/MinIO + compact 成 Parquet，适合训练后分析 |
| Kafka/Pulsar + sink | 高吞吐缓冲和多后端分发 | 通常作为中间层，再落 Loki/ClickHouse/object storage |
| Elasticsearch/OpenSearch | 全文检索 | 能用，但成本和运维通常比 Loki/ClickHouse 高 |

所以默认设计保留每进程分片作为 fail-open artifact，并通过 `observability_scratch_dir` 做 node-sharded 布局。生产集群可以加一个 agent 把本地分片送到 Loki/ClickHouse/object storage，并用 retention 清掉本地文件；也可以带外 compact 到 `observability_export_dir`。后续可以补一个标准 Grafana Alloy / Vector 配置模板，避免用户自己在共享盘里攒碎文件。

如果不希望用户单独部署服务，可以把 Grafana Alloy、Vector 或 Fluent Bit 这类 tailer 放进 slime 镜像，并由 `slime-observability local` 在本机启动。对用户来说这仍然是“跑一个 slime 镜像”，只是镜像里多了一个辅助进程；对系统设计来说，它仍然保持异步 tail、batch、retry、retention，不会让 SGLang 请求完成时同步 push 到 Loki。

## slime 自身 telemetry

SGLang metrics 能回答 serving 层面的很多问题，但不能完整回答 trainer 是在等 rollout、weight sync、data buffer，还是 Ray/object store。这个层面的 telemetry 很有价值，但默认实现不应该直接改写训练主循环。

v1 只在 observability bundle 边界内做低侵入改动：生成标准配置、注册 scrape target、注入 SGLang 本地 artifact 路径、写 status。后续如果要补 slime 自身的 step/rollout/weight-sync/request lifecycle 事件，推荐通过显式的轻量接口接入，例如：

- 使用已有 `Timer` / trace 工具导出低频 phase summary。
- 在 rollout/training 已有 log/perf 汇总点旁边增加可关闭的本地 writer。
- 对自定义 rollout 函数提供 opt-in correlation headers 或 hook，而不是默认包裹每个请求。
- 所有 slime-local events 都写本地 JSONL 或 textfile，fail open，不引入远端 SDK。

这样可以补上 RL 系统因果链，同时不把 observability 变成训练流程的硬依赖。

## RDMA 和通信速度

当前默认采集能看到一部分和通信相关的 request-level 症状，但还不能完整回答 RDMA 硬件层问题。

Prometheus scrape 的 `/engine_metrics` 能看到 SGLang serving 层面的现象，例如 queue、TTFT、decode/prefill latency、cache、吞吐等。request-time-stats 日志能看到更细的 request breakdown，PD disaggregation 下会包含 transfer duration、transfer speed、transfer total MB、retry 等字段。如果 RDMA 或网络有问题，这些指标会更容易暴露症状。

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
- 默认模式可以写 request-level profiling 文件，但不要写 prompt/output payload、不要写 stdout、不要同步发送远端日志。
- 不要把 `request_id`、`sample_id`、prompt hash、raw prompt、raw output、user id 放进 Prometheus labels。
- Prometheus labels 只放低基数字段，例如 `run_id`、`role`、`node`、`rank`、`worker_id`、`model_name`、`phase`。
- 高基数字段放 JSONL、Parquet 或 trace attributes，用来 join，不要做 label。
- Observability 必须 fail open。bundle 创建失败或 target 写失败时，训练继续跑。
- trainer 不拥有 Prometheus/Grafana/Loki 生命周期。slime 生成标准文件；镜像内 helper 可以把这些标准工具作为本机辅助进程启动，方便不熟悉云原生的用户。

## 当前代码位置

- `slime/profiling/observability.py`：负责 bundle 创建、manifest 写入、Prometheus config 渲染、file service discovery target 写入、SGLang request profiling 默认值注入。
- `slime/utils/arguments.py`：提供 `--enable-observability`、`--run-id`、`--run-dir`、`--observability-scratch-dir`、`--observability-prometheus-tsdb-dir`、`--observability-export-dir` 参数。
- `train.py` 和 `train_async.py`：在 tracking 之前初始化 bundle，在 rollout server 启动后注册 router target。
- `slime/utils/wandb_utils.py`：不再 reinit W&B 来上传 open metrics endpoints。
- `slime/utils/logging_utils.py`：只有 `--use-wandb` 时才 lazy import W&B。
- `slime/backends/sglang_utils/sglang_engine.py`：强制打开 SGLang metrics，保证 router `/metrics` 和 `/engine_metrics` 可被 Prometheus scrape；在 observability 模式下为本地 SGLang engine 写 logging config，使 `ReqTimeStats(...)` 落到每进程分片文件。
- `slime/profiling/request_time_stats.py`：提供通用 request-time-stats JSONL loader、SGLang `ReqTimeStats(...)` 兼容 parser、append-only log handler，以及 trace viewer/compact 工具共用的加载逻辑；compact 工具会保留 parse error 记录，避免静默丢掉格式漂移的行。
- `slime/utils/trace_utils.py`：把 `meta_info["id"]` 保存为 `sglang_request_id`，供 trace viewer join 日志。
- `tools/trace_timeline_viewer.py`：读取 request-time-stats 日志，按 `rid` 合并到 trace span，并保留旧 `pd_*` trace 字段兼容。
- `tools/compact_sglang_request_time_stats.py`：把 request-time-stats 分片目录合成一个结构化 JSONL 文件，供离线分析使用。

## TODO

短期：

- 给 `prepare_observability_args`、manifest redaction、Prometheus config 渲染、file service discovery 输出、request profiling 参数注入补单测。
- 在 quick start 或 profiling 文档里加一个最小使用例子。
- 增加 `slime-observability local` helper：读取 `run_dir` 和 manifest 中的 `prometheus_tsdb_dir`，在本机启动 Prometheus，写 pid/log，退出时能清理辅助进程。
- 确认不同规模 rollout 下 Prometheus scrape、SGLang metrics enable、request metrics JSONL、metadata-only request logs 的开销，用 micro benchmark 写进文档。
- 补 request-time-stats parser、append-only handler、compact 工具的单测，覆盖 prefill、decode、旧格式和当前 SGLang 格式。
- 验证所有本地 SGLang 子进程都继承 run-local logging config，尤其是多节点、多 DP/TP/EP、PD prefill/decode 分离场景。

中期：

- 增加 RDMA / network node-level sampler，低频采集 counters，默认 fail open，并支持 node exporter textfile collector。
- 生成 Grafana provisioning 和基础 dashboards，默认指向 run-local Prometheus。
- 在镜像里可选内置 Grafana Alloy / Vector / Fluent Bit 配置模板，让 helper 可以本机 tail request-time-stats 分片并异步发送到 Loki、ClickHouse 或对象存储。
- 增加 JSONL 到 Parquet 的 compact 工具，用于 request facts 和 request-time-stats 后处理。
- 增加 log rotation / max-size / retention 控制，避免 request-level profiling 文件撑满磁盘；fail open 需要更新 `observability/component_state.json`，不能静默停写。
- 如果 SGLang upstream 支持 request-time-stats 专用文件 target，就改用专用 target，避免通过普通 SGLang logging config 捕获。
- 如果 SGLang upstream 支持结构化 request-time-stats JSONL，就优先读结构化记录，只保留 `ReqTimeStats(...)` 文本 parser 作为兼容路径。
- 如果 SGLang upstream 支持更窄的 request metrics privacy option，把 exporter 的 skip list 从 request logger 解耦，减少辅助 request log 文件。
- 如果后续使用 SGLang tokenizer metric custom labels，加低基数 allowlist。
- 给 rollout 请求加 correlation headers，例如 `x-slime-run-id`、`x-slime-rollout-id`、`x-slime-weight-version`、request id。这些字段用于 join logs/traces/request facts，不应该直接作为 Prometheus labels。
- 增加 slime 自己的轻量 metrics endpoint 或本地 metrics 文件，记录 rollout latency、weight sync latency、data buffer size、global step 等 training/rollout coupling 指标；这部分需要显式 opt-in 或挂在已有汇总点上，避免侵入训练主循环。

长期：

- 可选接入本机 Loki 或远端 Loki / ClickHouse / 对象存储，给需要日志检索和长期保存的用户使用，但默认 artifact 仍然是本地文件。
- 可选接入 OpenTelemetry tracing，必须显式 opt-in，并带采样、脱敏、retention。
- 生成标准 `docker-compose.yaml` 作为启动 Prometheus/Grafana/Loki 的 convenience wrapper；主路径仍然是 slime 镜像内 helper，避免要求用户掌握额外部署系统。
- 增加能回答 RL-serving 问题的 dashboard：queue 堆积、TTFT spike、cache hit rate 变化、prefill/decode 负载、weight sync pause、worker hot spot、request-level transfer 慢点、RDMA link/counter 异常。
