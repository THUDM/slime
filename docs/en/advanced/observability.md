# Observability Bundle

slime's observability path is a per-run local bundle. The goal is to make long-running RL jobs inspectable during training and after a crash without uploading high-volume SGLang serving metrics to W&B or putting Prometheus, Grafana, Loki, or OpenTelemetry SDKs in the trainer critical path.

There is one default production mode: `--enable-observability`. This mode collects local artifacts that are useful for profiling while keeping overhead controlled.

- W&B keeps algorithm metrics and also logs a small set of request perf summaries aggregated from `meta_info` / sample traces, such as P/D duration, transfer speed, and retry count.
- Prometheus scrapes SGLang serving/system metrics from the router's `/metrics` and `/engine_metrics` endpoints, covering router-native metrics and router-aggregated engine metrics.
- SGLang response `meta_info` request timing fields are stored in sample traces, and the trace viewer uses the `pd_*` fields directly to reconstruct PD prefill/decode timelines.
- slime writes run metadata, Prometheus config, file-based service discovery files, and fail-open local status/component-state files.
- slime does not automatically enable token-level traces, payload dumps, remote logging backends, or replay artifacts.
- slime does not inject step, weight-sync, or request-lifecycle events into the training loop by default; trainer/rollout coupling telemetry should be added through explicit, non-invasive hooks or helpers.
- The trainer process does not start or own Prometheus, Grafana, Loki, or OpenTelemetry collectors. For non-cloud-native users, the slime image can include a local helper that starts these standard tools on the same machine or inside the same container.

## Why Not W&B for SGLang Metrics

SGLang Prometheus metrics are high-volume system metrics. Uploading them through W&B can make W&B slow and mixes very different data types in one place. slime keeps the split explicit:

| Data type | Destination | Default state | Notes |
| --- | --- | --- | --- |
| Algorithm metrics | W&B / TensorBoard | unchanged | reward, loss, KL, entropy, eval |
| SGLang request perf summaries | W&B / TensorBoard | written during rollout perf aggregation | Aggregated from sample trace timing fields; low frequency and low cardinality |
| SGLang system metrics | Prometheus | scrapeable with `--enable-observability` | scraped from `/metrics` and `/engine_metrics` |
| Run metadata | local JSON | written with `--enable-observability` | run id, paths, environment, versions, allowlisted args |
| Component status | local JSON/JSONL | written with `--enable-observability` | `status.json`, `component_state.json`, `errors.jsonl` |
| Traces / replay dumps | TODO | disabled by default | future explicit opt-in only |

This boundary matters: monitoring failures must not stop training.

## Performance Impact

The default `--enable-observability` path should be low overhead, but it is not mathematically zero.

- Bundle files are written at startup and when the router target is registered. They are not on the rollout request hot path.
- Prometheus scrapes the router `/metrics` and `/engine_metrics` endpoints every 5 seconds. This is HTTP pull-based, does not run in the trainer process, and does not synchronously block rollout.
- SGLang still maintains counters and histograms for the metrics endpoint. slime currently keeps SGLang metrics enabled so the router `/engine_metrics` endpoint is available. This overhead is typically much smaller than model inference.
- SGLang `meta_info` request timings already return to Python with the response. slime stores them in sample traces and logs only aggregate `perf/...` summaries to W&B/TensorBoard after rollout completes, not one point per request.
- The default path does not write SGLang request metrics JSONL, request logs, or `ReqTimeStats(...)` engine logs, and those records do not appear in daily stdout.
- If Prometheus is not running, training still runs. Bundle and target writing are fail-open.

The daily profiling source is the request timing fields in SGLang response `meta_info`. They land in sample traces, the trace viewer renders `[P]` / `[D]` lanes from `pd_*` fields, and W&B/TensorBoard receive one aggregate set per rollout step, such as `perf/prefill/forward_duration/mean`, `perf/decode/transfer_duration/max`, and `perf/request/queue_time/median`.

The default production path is therefore:

```text
production profiling default = manifest + status + Prometheus config + file_sd + /metrics scrape target + /engine_metrics scrape target + sample trace timing attrs + W&B/TensorBoard perf/* aggregates
```

This keeps high-frequency SGLang Prometheus metrics in Prometheus and sends only compact request perf summaries to W&B/TensorBoard.

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
export SLIME_PROMETHEUS_TSDB_DIR=/local_nvme/prometheus/{run_id}
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

## Request Perf Metrics Behavior

SGLang response `meta_info` provides request timings that also exist without PD, for example:

```text
queue_time
e2e_latency
decode_throughput
```

Under PD disaggregation, the SGLang patch also puts the request-time-stats fields that matter most for training profiling into response `meta_info`, for example:

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

slime stores those fields in the `sglang_generate` span attrs. During rollout metric aggregation, `slime/ray/rollout.py` extracts every `sglang_generate` span from sample traces and maps internal `pd_*` fields to clearer user-facing `perf/prefill/...`, `perf/decode/...`, and `perf/request/...` metrics. Each field logs `mean`, `median`, `max`, `min`, and `count`:

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

These metrics go through the existing `logging_utils.log()` path, so both W&B and TensorBoard can see them. They are aggregated once per rollout step, not emitted once per request, so they should not make W&B slow like uploading raw Prometheus metrics would. Without PD, `perf/request/...` and available `perf/decode/throughput/...` still exist; detailed `perf/prefill/...` and `perf/decode/...` durations only appear when SGLang returns the corresponding timing fields.

## Prometheus P/D Queue Metrics

Prometheus owns live serving state. Current SGLang `/engine_metrics` already exposes PD-disaggregation queue gauges that are useful for spotting congestion at a point in time:

```text
sglang:num_queue_reqs
sglang:num_running_reqs
sglang:num_prefill_bootstrap_queue_reqs
sglang:num_prefill_inflight_queue_reqs
sglang:num_decode_prealloc_queue_reqs
sglang:num_decode_transfer_queue_reqs
```

These metrics carry low-cardinality labels such as `engine_type`, `dp_rank`, `tp_rank`, `pp_rank`, and `model_name`, so Grafana can split prefill-engine and decode-engine queue buildup.

Prometheus can also show part of the request-stage latency and KV-transfer picture:

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

This complements W&B/TensorBoard `perf/...` metrics: Prometheus shows online queue depth, histograms, and error counters; sample traces and W&B aggregates show min/max/mean/median for completed requests, plus trace-viewer P/D timelines.

## Trace Viewer Duration Input

The trace viewer reads `pd_*` attrs directly from sample traces in debug rollout dumps and renders the synthetic `[P]` / `[D]` lanes from those attrs. The primary path does not need a separate `ReqTimeStats(...)` log file, Loki, or request-time-stats compaction.

Future Grafana-like analysis should be added as native trace viewer panels first: request-duration histograms, p95/p99, `input_len` vs transfer-duration scatter plots, transfer-speed timelines, and similar views. That keeps the user flow to opening the trace viewer instead of deploying a separate log stack.

## slime-Side Telemetry

SGLang metrics answer many serving-layer questions, but they cannot fully explain whether the trainer is waiting on rollout, weight sync, the data buffer, or Ray/object-store behavior. That telemetry is useful, but the default implementation should not rewrite the training loop.

v1 keeps implementation inside the observability bundle boundary: standard config generation, scrape target registration, and status files. If slime-side step, rollout, weight-sync, or request-lifecycle events are added later, they should use explicit lightweight interfaces, for example:

- Export low-frequency phase summaries from existing `Timer` / trace utilities.
- Add optional local writers beside existing rollout/training log summary points.
- Provide opt-in correlation headers or hooks for custom rollout functions instead of wrapping every request by default.
- Keep slime-local events in local JSONL or textfiles, fail open, and avoid remote SDKs in the hot path.

This preserves the RL-system causal chain without making observability a hard dependency of the training flow.

## RDMA and Network Speed

The current default collection can show some communication-related request symptoms, but it cannot fully answer hardware-level RDMA questions.

Prometheus scraping `/engine_metrics` can show SGLang serving symptoms such as P/D queue buildup, TTFT, prefill/decode latency, cache behavior, throughput, KV-transfer speed/latency/size histograms, retries, and failure counters. Sample-trace `pd_*` fields add completed-request transfer duration, transfer speed, transfer total MB, and retry summaries at rollout granularity. RDMA or network problems should be more visible through these symptoms.

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
- The default mode must not write request-level profiling files, prompt/output payloads, stdout request logs, or synchronous remote logs.
- Do not put `request_id`, `sample_id`, prompt hashes, raw prompts, raw outputs, or user ids into Prometheus labels.
- Keep Prometheus labels low-cardinality: `run_id`, `role`, `node`, `rank`, `worker_id`, `model_name`, and `phase` are acceptable examples.
- Put high-cardinality join fields in JSONL, Parquet, or trace attributes instead.
- Observability should fail open. If bundle creation or target writing fails, training should continue.
- Do not make the trainer own Prometheus, Grafana, or Loki lifecycle. slime writes standard files; an image-bundled helper can start those tools as local helper processes for users who are not familiar with cloud-native deployments.

## Current Implementation Points

- `slime/profiling/observability.py` owns bundle creation, manifest writing, Prometheus config rendering, and file service discovery target writing.
- `slime/utils/arguments.py` exposes `--enable-observability`, `--run-id`, `--run-dir`, and `--observability-prometheus-tsdb-dir`.
- `train.py` and `train_async.py` initialize the bundle before tracking and register the router target after rollout servers start.
- `slime/utils/wandb_utils.py` no longer re-initializes W&B with open metrics endpoints.
- `slime/utils/logging_utils.py` imports W&B lazily only when `--use-wandb` is enabled.
- `slime/backends/sglang_utils/sglang_engine.py` still always enables SGLang metrics so the router `/metrics` and `/engine_metrics` endpoints are available.
- `slime/utils/trace_utils.py` stores `meta_info["id"]` as `sglang_request_id` and stores request timing fields from `meta_info` in sample traces.
- `slime/ray/rollout.py` aggregates request / prefill / decode perf metrics from sample traces and logs min/max/mean/median/count through the existing W&B/TensorBoard path.
- `tools/trace_timeline_viewer.py` reads `pd_*` fields directly from sample traces and renders synthetic `[P]` / `[D]` lanes.

## TODO

Short-term:

- Add tests for `prepare_observability_args`, manifest redaction, Prometheus config rendering, file service discovery output, and rollout request perf aggregation.
- Add a small example command to the quick-start or profiling docs.
- Add a `slime-observability local` helper that reads `run_dir` and the manifest's `prometheus_tsdb_dir`, starts local Prometheus, writes pid/log files, and can clean up helper processes on exit.
- Benchmark Prometheus scraping, SGLang metrics enablement, and sample trace request perf aggregation at different rollout scales, then document the measured overhead.

Medium-term:

- Add an RDMA / network node-level sampler that collects counters at low frequency, fails open, and can integrate with node exporter's textfile collector.
- Generate Grafana provisioning and starter dashboards that point at the run-local Prometheus TSDB.
- Add low-cardinality label allowlists for any future custom SGLang tokenizer metric labels.
- Add correlation headers for rollout requests, such as `x-slime-run-id`, `x-slime-rollout-id`, `x-slime-weight-version`, and a request id. These should be used for joining logs/traces/request facts, not as Prometheus labels.
- Expose a small slime-side metrics endpoint or local metrics file for rollout/training coupling metrics such as rollout latency, weight sync latency, data buffer size, and global step; this should be explicit opt-in or attached to existing summary points, not a default training-loop wrapper.

Long-term:

- Add optional local Loki or remote Loki / ClickHouse / object-storage integration for users who want log search and long-term retention, while keeping local files as the default artifact.
- Add optional OpenTelemetry tracing with explicit opt-in, sampling, redaction, and retention.
- Generate a standard `docker-compose.yaml` for Prometheus/Grafana/Loki as a convenience wrapper; the primary path should still be the image-bundled helper so users do not need to learn another deployment system.
- Add dashboard panels that answer RL-serving questions: queue buildup, TTFT spikes, cache hit rate changes, prefill/decode balance, weight-sync pauses, worker hot spots, slow request-level transfer, and RDMA link/counter anomalies.
