# Observability Bundle

slime's observability path is a per-run local bundle. The goal is to make long-running RL jobs inspectable during training and after a crash without uploading high-volume SGLang serving metrics to W&B or putting Prometheus, Grafana, Loki, or OpenTelemetry SDKs in the trainer critical path.

There is one default production mode: `--enable-observability`. This mode collects local artifacts that are useful for profiling while keeping overhead controlled.

- W&B keeps algorithm metrics such as loss, reward, KL, entropy, and eval scores.
- Prometheus scrapes SGLang serving/system metrics from the router's `/metrics` and `/engine_metrics` endpoints, covering router-native metrics and router-aggregated engine metrics.
- SGLang request metrics are written to local JSONL files, one record per completed request, so request-level profiling state can be reconstructed after the run.
- SGLang request logs are written as `level=0` metadata-only JSON files under the run directory, not to the user's daily stdout.
- SGLang `ReqTimeStats(...)` logs are written to local engine log files, and the trace viewer can read them to reconstruct PD prefill/decode timelines.
- slime writes run metadata, Prometheus config, file-based service discovery files, and fail-open local status/component-state files.
- High-frequency request-level artifacts default to a node-sharded scratch path instead of the `run_dir` root.
- slime does not automatically enable token-level traces, payload dumps, remote logging backends, or replay artifacts.
- slime does not inject step, weight-sync, or request-lifecycle events into the training loop by default; trainer/rollout coupling telemetry should be added through explicit, non-invasive hooks or helpers.
- The trainer process does not start or own Prometheus, Grafana, Loki, or OpenTelemetry collectors. For non-cloud-native users, the slime image can include a local helper that starts these standard tools on the same machine or inside the same container.

## Why Not W&B for SGLang Metrics

SGLang Prometheus metrics are high-volume system metrics. Uploading them through W&B can make W&B slow and mixes very different data types in one place. slime keeps the split explicit:

| Data type | Destination | Default state | Notes |
| --- | --- | --- | --- |
| Algorithm metrics | W&B / TensorBoard | unchanged | reward, loss, KL, entropy, eval |
| SGLang system metrics | Prometheus | scrapeable with `--enable-observability` | scraped from `/metrics` and `/engine_metrics` |
| Run metadata | local JSON | written with `--enable-observability` | run id, paths, environment, versions, allowlisted args |
| Component status | local JSON/JSONL | written with `--enable-observability` | `status.json`, `component_state.json`, `errors.jsonl` |
| Request facts | local JSONL files | written with `--enable-observability` | SGLang `export_metrics_to_file`, one record per completed request |
| Request logs | local JSON logs | written with `--enable-observability` | metadata-only, file target, no stdout |
| Request time stats | local engine logs | written with `--enable-observability` | SGLang `ReqTimeStats(...)`, used for profiling and the trace viewer |
| Traces / replay dumps | TODO | disabled by default | future explicit opt-in only |

This boundary matters: monitoring failures must not stop training.

## Performance Impact

The default `--enable-observability` path should be low overhead, but it is not mathematically zero.

- Bundle files are written at startup and when the router target is registered. They are not on the rollout request hot path.
- Prometheus scrapes the router `/metrics` and `/engine_metrics` endpoints every 5 seconds. This is HTTP pull-based, does not run in the trainer process, and does not synchronously block rollout.
- SGLang still maintains counters and histograms for the metrics endpoint. slime currently keeps SGLang metrics enabled so the router `/engine_metrics` endpoint is available. This overhead is typically much smaller than model inference.
- SGLang's request metrics exporter writes one JSONL record when a request finishes. It does not write once per token. The write is triggered through `asyncio.create_task`, then formats JSON, writes a file record, and flushes.
- SGLang's request logger writes metadata-only JSON logs on request receive/finish. slime points the target at a local file directory, so the user's daily stdout does not show these logs.
- SGLang request-time-stats logging writes one `ReqTimeStats(...)` line when a request finishes. slime injects a run-local logging config for local engines so these SGLang subprocess logs go to `request_time_stats/sglang` under the scratch directory instead of relying on Ray driver stdout, which can aggregate repeated lines.
- If Prometheus is not running, training still runs. Bundle and target writing are fail-open.

Request metrics and request-time-stats logs have different jobs. Request metrics JSONL records request parameters and final `meta_info`, which is useful for request facts and post-crash inspection. `ReqTimeStats(...)` logs come directly from the prefill/decode engines and carry queue, forward, PD transfer, transfer speed, retry, and related durations. They are the primary source for rebuilding PD timelines and trace viewer `[P]` / `[D]` lanes.

The default production path is therefore:

```text
production profiling default = manifest + status + Prometheus config + file_sd + /metrics scrape target + /engine_metrics scrape target + request metrics JSONL + metadata-only request logs + ReqTimeStats engine logs
```

The path contract is split: `run_dir` holds durable low-frequency metadata/config/status; `observability_scratch_dir` holds high-frequency request-level files; `observability_prometheus_tsdb_dir` is for Prometheus local TSDB; and `observability_export_dir` is for compacted durable outputs. Under high QPS, slow disks, shared filesystems, large log volume, or near-full disks, request-level files can still affect tail latency or throughput. TODOs track benchmarking, buffering/rotation, and a narrower SGLang request metrics privacy option.

## User Interface

Enable the bundle with:

```bash
python train.py \
  ... \
  --enable-observability \
  --run-dir /mnt/runs/ppo_qwen3_001
```

You can also use environment variables:

```bash
export SLIME_ENABLE_OBSERVABILITY=1
export SLIME_RUN_DIR=/mnt/runs/ppo_qwen3_001
export SLIME_OBS_SCRATCH_DIR=/local_nvme/slime-obs/{run_id}/node={hostname}
export SLIME_PROMETHEUS_TSDB_DIR=/local_nvme/prometheus/{run_id}
export SLIME_OBS_EXPORT_DIR=/mnt/runs/ppo_qwen3_001/export
```

If `--run-id` is not provided, slime generates one. If `--run-dir` is not provided, slime uses:

```text
/tmp/slime-runs/{run_id}
```

The old `--observability-profile` argument is deprecated and hidden. `basic/debug/replay` are no longer the daily user interface because production training should have one profiling configuration.

## Generated Layout

For `--run-dir /mnt/runs/ppo_qwen3_001`, slime writes:

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

`manifest.json` records the run id, important paths, environment information, component versions, and an allowlisted subset of the slime argument namespace. The full argparse namespace is not written by default; keys containing key, token, secret, password, authorization, or credential are redacted even if they appear in the allowlist.

The generated Prometheus config points to the run-local file service discovery file:

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

After the SGLang router starts, slime writes two scrape target files. They usually point at the same router address, but their `metrics_endpoint` labels differ:

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

## Starting Prometheus Locally

The minimal workflow does not require a separately deployed service. The slime image can install Prometheus, and the user only needs to choose a durable `run_dir` and start Prometheus on the same machine with the run-local config:

```bash
prometheus \
  --config.file=/mnt/runs/ppo_qwen3_001/prometheus/prometheus.yml \
  --storage.tsdb.path=/local_nvme/prometheus/ppo_qwen3_001 \
  --web.listen-address=:19090
```

Prefer local disk for `--storage.tsdb.path`, for example the path in `SLIME_PROMETHEUS_TSDB_DIR`. Prometheus local storage expects POSIX-like local storage semantics; shared NFS/EFS-style paths can be slow or unsafe for TSDB writes.

The Docker images install Prometheus and include an example all-runs config at:

```text
/etc/prometheus/slime-example-prometheus.yml
```

The preferred path for one job is still the generated run-local `prometheus/prometheus.yml`.

A better user-facing shape is a helper bundled in the image, for example:

```bash
slime-observability local \
  --run-dir /mnt/runs/ppo_qwen3_001 \
  --prometheus-port 19090
```

The helper would only start standard components as local helper processes, such as Prometheus, optional Grafana, and an optional log tailer. The training process still only writes the local bundle and profiling files. Users do not need Kubernetes, Helm, or a separate Loki/Prometheus deployment, while remote log shipping, query backend SDKs, and retry logic stay out of the rollout hot path.

Installing dependencies in the image does not by itself make artifacts durable. If the container filesystem or node-local disk is reclaimed after a failed job, `run_dir` must be mounted on durable storage such as NFS/a parallel filesystem, PVC/hostPath, or be drained promptly by a local log tailer into Loki, ClickHouse, object storage, or another backend.

## Request Metrics Behavior

With `--enable-observability`, slime injects these SGLang arguments:

```text
sglang_export_metrics_to_file = true
sglang_export_metrics_to_file_dir = ${observability_scratch_dir}/request_metrics/sglang
sglang_log_requests = true
sglang_log_requests_level = 0
sglang_log_requests_format = json
sglang_log_requests_target = [${observability_scratch_dir}/logs/sglang]
sglang_enable_request_time_stats_logging = true
```

`export_metrics_to_file` is the request facts artifact. It writes one JSONL record when each request finishes. Fields come from request parameters and the final `meta_info`, which makes it useful for request-level lookup and post-crash inspection.

`enable_request_time_stats_logging` is the main artifact for PD profiling and the trace viewer. SGLang writes lines like this when prefill/decode engine requests finish:

```text
ReqTimeStats(rid=..., bootstrap_room=..., input_len=..., output_len=..., type=decode): prealloc_queue_duration(...ms) = bootstrap(...ms) + alloc_wait(...ms); transfer_duration=...ms; queue_duration=...ms, forward_duration=...ms
```

For local engines, slime generates a logging config referenced by `SGLANG_LOGGING_CONFIG_PATH` so these lines land under `${observability_scratch_dir}/request_time_stats/sglang`. It does not read Ray driver stdout because stdout can be aggregated, truncated, or displayed as repeated lines.

The default hot-path layout is one append-only shard per engine process, for example:

```text
${observability_scratch_dir}/request_time_stats/sglang/
  worker_type=prefill/rank=0/pid=1234.jsonl
  worker_type=decode/rank=0/pid=5678.jsonl
```

Each process owns its shard, so prefill/decode engines on the same node do not append to one shared file. The larger file count is handled by out-of-band compaction instead of by merging in the request-completion path. Lines still include `worker_type` and `rank`, and those dimensions also appear in the path for easier post-processing.

If containers or node-local disks are reclaimed after a failed job, do not rely only on ephemeral local disk. `run_dir` can point at shared NFS or a parallel filesystem, while high-frequency request-level artifacts should prefer `observability_scratch_dir` and then be copied to durable storage by an agent or compaction/export step. Another option is to write node-local shards and have a log agent tail them into Loki, ClickHouse, or object storage; in that setup the local shard is only a short-lived buffer.

Do not make all nodes append directly to one global file in the training hot path. Cross-node append atomicity, ordering, and throughput depend on the filesystem, and NFS/EFS-style shared paths are especially risky. If you want one file for analysis, compact the shards out of band:

```bash
python tools/compact_sglang_request_time_stats.py \
  /local_nvme/slime-obs/ppo_qwen3_001/node=node-a/request_time_stats/sglang \
  --output /mnt/runs/ppo_qwen3_001/export/sglang_request_time_stats.jsonl
```

The trace viewer joins records by `sglang_request_id` / `rid`; if an older trace dump already contains `pd_*` attrs, those existing attrs win for compatibility.

`sglang_log_requests_level=0` is enabled for the current SGLang exporter privacy boundary. The request metrics exporter still reuses the request logger skip list; without the metadata-only logger, it may write a wider set of request parameters. slime also records the effective privacy mode in the manifest and writes only an allowlisted config subset there. If SGLang upstream adds an exporter-owned allowlist/privacy option, slime should switch to that and reduce reliance on request logger side effects.

Request logs are auxiliary artifacts, and their target is a local file directory rather than `stdout`. The files can still contain request metadata, so protect the run directory as training data.

If you use external rollout engines, slime can only write the Prometheus target; it cannot change the SGLang flags or logging setup of an already-running external service. Set the same `--export-metrics-to-file`, `--log-requests-level 0`, file target, `--enable-request-time-stats-logging`, and save SGLang `ReqTimeStats(...)` logs to a directory or compacted JSONL file that the trace viewer can read.

## Trace Viewer and Request-Time-Stats Logs

SGLang generates a random `rid` for each request, and both the response `meta_info["id"]` and `ReqTimeStats(rid=...)` logs use that id. After slime receives the response, it records `meta_info["id"]` as the span attribute `sglang_request_id`. `tools/trace_timeline_viewer.py` reads request-time-stats logs while building the cache, joins log records by `rid`, and then uses the existing synthetic `[P]` / `[D]` lane rendering.

Common usage:

```bash
python tools/trace_timeline_viewer.py \
  /path/to/debug/rollout_0.pt \
  --request-time-stats-path /mnt/runs/ppo_qwen3_001/request_time_stats/sglang
```

If no path is provided, the viewer checks `SLIME_REQUEST_TIME_STATS_PATH`, then `SLIME_RUN_DIR/request_time_stats/sglang`, then `request_time_stats/sglang` near the `.pt` file.

`--request-time-stats-path` can point to a directory or to one compacted JSONL/log file.

The latest SGLang patch no longer sends an extra prefill timing buffer between P/D workers only for the trace viewer. The viewer supports both sources: existing `pd_*` attrs in trace dumps and fields joined by `rid` from request-time-stats logs or compacted JSONL. SGLang `ReqTimeStats(...)` text is the compatibility input; if SGLang writes structured JSONL later, the loader can read it directly.

## Choosing a Log Backend

Loki is a good log storage system, especially for searching raw logs by run, node, worker type, and time range. It should not be the synchronous write target in the SGLang request hot path. The recommended shape is:

```text
SGLang ReqTimeStats
  -> one append-only shard per engine process (local short-lived buffer or node-sharded scratch)
  -> Promtail / Grafana Alloy / Vector / Fluent Bit tailing
  -> Loki
```

This avoids long-lived piles of small files on a shared filesystem, and Loki or agent failures do not block training or inference. The agent can own queueing, batching, retry, drop policy, and retention; if the shard is on ephemeral local disk, the agent is part of the durability path. High-cardinality fields such as `rid` and `bootstrap_room` must stay in the log body, not labels. Loki labels should stay low-cardinality, for example:

```text
run_id, component=sglang, log_type=req_time_stats, node, worker_type
```

If the goal is large-scale statistics over request duration, transfer speed, retry count, or P/D breakdowns, Loki works but is not the strongest backend. Better fits are:

| Backend | Best fit | Notes |
| --- | --- | --- |
| Loki | Log search, raw lines by time range, Grafana integration | Do not label by `rid`; good centralized log backend |
| ClickHouse | Large-scale structured profiling queries | Good for request aggregation, p99, transfer speed, and worker hotspots |
| Object storage + Parquet | Low-cost long-term archive and offline analysis | For example S3/MinIO plus compaction into Parquet |
| Kafka/Pulsar + sinks | High-throughput buffering and fan-out | Usually an intermediate layer before Loki/ClickHouse/object storage |
| Elasticsearch/OpenSearch | Full-text search | Usable, but often higher cost and operational overhead than Loki/ClickHouse |

The default design therefore keeps one shard per engine process as the fail-open artifact and uses `observability_scratch_dir` for node-sharded placement. Production clusters can add an agent to ship local shards to Loki, ClickHouse, or object storage and clean local files by retention, or compact them out of band into `observability_export_dir`. A future TODO is to provide a standard Grafana Alloy / Vector config template so users do not need to accumulate fragmented logs on a shared disk.

If users should not deploy a separate service, Grafana Alloy, Vector, or Fluent Bit can be bundled in the slime image and started by `slime-observability local`. To users this is still "run one slime image"; internally it remains an asynchronous tail, batch, retry, and retention path, not a synchronous Loki push from the SGLang request completion path.

## slime-Side Telemetry

SGLang metrics answer many serving-layer questions, but they cannot fully explain whether the trainer is waiting on rollout, weight sync, the data buffer, or Ray/object-store behavior. That telemetry is useful, but the default implementation should not rewrite the training loop.

v1 keeps implementation inside the observability bundle boundary: standard config generation, scrape target registration, SGLang local artifact paths, and status files. If slime-side step, rollout, weight-sync, or request-lifecycle events are added later, they should use explicit lightweight interfaces, for example:

- Export low-frequency phase summaries from existing `Timer` / trace utilities.
- Add optional local writers beside existing rollout/training log summary points.
- Provide opt-in correlation headers or hooks for custom rollout functions instead of wrapping every request by default.
- Keep slime-local events in local JSONL or textfiles, fail open, and avoid remote SDKs in the hot path.

This preserves the RL-system causal chain without making observability a hard dependency of the training flow.

## RDMA and Network Speed

The current default collection can show some communication-related request symptoms, but it cannot fully answer hardware-level RDMA questions.

Prometheus scraping `/engine_metrics` can show SGLang serving symptoms such as queue buildup, TTFT, prefill/decode latency, cache behavior, and throughput. Request-time-stats logs add finer request breakdowns; under PD disaggregation they include transfer duration, transfer speed, transfer total MB, and retry fields. RDMA or network problems should be more visible through these symptoms.

RDMA bandwidth, retransmits, packet drops, NIC error counters, and link state still require node-level counters. That needs a future low-frequency, out-of-band, fail-open collector:

- Sample `rdma` / `ibstat` / `perfquery` / sysfs counters every 10 to 30 seconds.
- Collect NIC bytes, packets, errors, discards, retransmits, link state, and link speed.
- Write local `network/rdma/*.jsonl`, or expose values through node exporter's textfile collector.
- Never run those commands from the rollout request path.
- Treat sampling failures as warnings, not training failures.

That lets dashboards put SGLang latency/throughput, request-level transfer stats, and RDMA counters on the same timeline.

## Operational Rules

Keep these rules when extending the bundle:

- Do not upload SGLang Prometheus metrics to W&B.
- The default mode may write request-level profiling files, but it must not write prompt/output payloads, write to stdout, or synchronously send remote logs.
- Do not put `request_id`, `sample_id`, prompt hashes, raw prompts, raw outputs, or user ids into Prometheus labels.
- Keep Prometheus labels low-cardinality: `run_id`, `role`, `node`, `rank`, `worker_id`, `model_name`, and `phase` are acceptable examples.
- Put high-cardinality join fields in JSONL, Parquet, or trace attributes instead.
- Observability should fail open. If bundle creation or target writing fails, training should continue.
- Do not make the trainer own Prometheus, Grafana, or Loki lifecycle. slime writes standard files; an image-bundled helper can start those tools as local helper processes for users who are not familiar with cloud-native deployments.

## Current Implementation Points

- `slime/profiling/observability.py` owns bundle creation, manifest writing, Prometheus config rendering, file service discovery target writing, and SGLang request profiling defaults.
- `slime/utils/arguments.py` exposes `--enable-observability`, `--run-id`, `--run-dir`, `--observability-scratch-dir`, `--observability-prometheus-tsdb-dir`, and `--observability-export-dir`.
- `train.py` and `train_async.py` initialize the bundle before tracking and register the router target after rollout servers start.
- `slime/utils/wandb_utils.py` no longer re-initializes W&B with open metrics endpoints.
- `slime/utils/logging_utils.py` imports W&B lazily only when `--use-wandb` is enabled.
- `slime/backends/sglang_utils/sglang_engine.py` still always enables SGLang metrics so the router `/metrics` and `/engine_metrics` endpoints are available; in observability mode it also writes a run-local logging config so `ReqTimeStats(...)` lands in per-process shard files.
- `slime/profiling/request_time_stats.py` provides the generic request-time-stats JSONL loader, SGLang `ReqTimeStats(...)` compatibility parser, append-only log handler, and shared loading logic used by the trace viewer and compaction tool; the compaction path preserves parse-error records so log format drift is not silently dropped.
- `slime/utils/trace_utils.py` stores `meta_info["id"]` as `sglang_request_id` for trace viewer joins.
- `tools/trace_timeline_viewer.py` reads request-time-stats logs, joins by `rid`, and keeps compatibility with old `pd_*` trace attrs.
- `tools/compact_sglang_request_time_stats.py` compacts request-time-stats shards into one structured JSONL file for offline analysis.

## TODO

Short-term:

- Add tests for `prepare_observability_args`, manifest redaction, Prometheus config rendering, file service discovery output, and request profiling argument injection.
- Add a small example command to the quick-start or profiling docs.
- Add a `slime-observability local` helper that reads `run_dir` and the manifest's `prometheus_tsdb_dir`, starts local Prometheus, writes pid/log files, and can clean up helper processes on exit.
- Benchmark Prometheus scraping, SGLang metrics enablement, request metrics JSONL, and metadata-only request logs at different rollout scales, then document the measured overhead.
- Add tests for the request-time-stats parser, append-only handler, and compaction tool, covering prefill, decode, older log format, and the current SGLang format.
- Verify that all local SGLang subprocesses inherit the run-local logging config in multi-node, multi-DP/TP/EP, and PD prefill/decode deployments.

Medium-term:

- Add an RDMA / network node-level sampler that collects counters at low frequency, fails open, and can integrate with node exporter's textfile collector.
- Generate Grafana provisioning and starter dashboards that point at the run-local Prometheus TSDB.
- Optionally bundle Grafana Alloy / Vector / Fluent Bit config templates in the image so the helper can tail request-time-stats shards locally and asynchronously ship them to Loki, ClickHouse, or object storage.
- Add optional JSONL-to-Parquet compaction for request facts and request-time-stats files.
- Add log rotation, max-size, and retention controls so request-level profiling files cannot fill disks silently; fail-open degradation should update `observability/component_state.json`.
- If SGLang upstream supports a dedicated request-time-stats file target, switch to it instead of capturing through the general SGLang logging config.
- If SGLang upstream supports structured request-time-stats JSONL, prefer structured records and keep the `ReqTimeStats(...)` text parser only as a compatibility path.
- If SGLang upstream supports a narrower request metrics privacy option, decouple exporter skip lists from request logging and reduce auxiliary request log files.
- Add low-cardinality label allowlists for any future custom SGLang tokenizer metric labels.
- Add correlation headers for rollout requests, such as `x-slime-run-id`, `x-slime-rollout-id`, `x-slime-weight-version`, and a request id. These should be used for joining logs/traces/request facts, not as Prometheus labels.
- Expose a small slime-side metrics endpoint or local metrics file for rollout/training coupling metrics such as rollout latency, weight sync latency, data buffer size, and global step; this should be explicit opt-in or attached to existing summary points, not a default training-loop wrapper.

Long-term:

- Add optional local Loki or remote Loki / ClickHouse / object-storage integration for users who want log search and long-term retention, while keeping local files as the default artifact.
- Add optional OpenTelemetry tracing with explicit opt-in, sampling, redaction, and retention.
- Generate a standard `docker-compose.yaml` for Prometheus/Grafana/Loki as a convenience wrapper; the primary path should still be the image-bundled helper so users do not need to learn another deployment system.
- Add dashboard panels that answer RL-serving questions: queue buildup, TTFT spikes, cache hit rate changes, prefill/decode balance, weight-sync pauses, worker hot spots, slow request-level transfer, and RDMA link/counter anomalies.
