# Observability

slime's default observability path is intentionally small: training metrics still go to W&B / TensorBoard; high-frequency SGLang Prometheus metrics are no longer uploaded to W&B; request timings from SGLang response `meta_info` are stored in sample traces and aggregated once per rollout step as compact `perf/...` metrics.

## W&B / TensorBoard Metrics

W&B and TensorBoard still receive reward, loss, KL, entropy, eval, and other training metrics. SGLang request timing summaries are logged under `perf/`, for example:

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

These metrics are aggregated once per rollout step, not emitted once per request, so they should not slow W&B like uploading raw Prometheus metrics would.

Without PD, common `perf/request/...` metrics and available `perf/decode/throughput/...` metrics still exist. Detailed `perf/prefill/...` and `perf/decode/...duration` metrics only appear when SGLang returns the corresponding `pd_*` timing fields.

## Where Prometheus Data Is Stored

slime does not store per-second Prometheus data. SGLang / router only expose `/metrics` and `/engine_metrics` HTTP endpoints. Prometheus scrapes those endpoints periodically and stores the time series in Prometheus's own TSDB.

That means:

- If Prometheus is not running, serving metrics are only available from the current SGLang process memory and endpoint output, with no historical storage.
- If Prometheus is running, history is stored under Prometheus's `--storage.tsdb.path`.
- slime does not upload these high-frequency metrics to W&B.

Useful SGLang metrics include:

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

These are useful in Prometheus / Grafana for live queue buildup, transfer speed, latency histograms, failure counters, and other serving-side symptoms.

## Starting Prometheus

Prometheus must run while training is running because it can only scrape endpoints that are currently alive. It does not need to run inside the training Python process; run it as a side process in the same machine or job.

A minimal config is:

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

Replace `ROUTER_IP:ROUTER_PORT` with the router address printed by slime, or with the explicit `--sglang-router-ip` / `--sglang-router-port` values.

Start Prometheus with its TSDB path on persistent storage:

```bash
prometheus \
  --config.file=/path/to/prometheus.yml \
  --storage.tsdb.path=/path/to/prometheus-data \
  --storage.tsdb.retention.time=7d \
  --web.listen-address=0.0.0.0:9090
```

The slime image includes the `prometheus` binary, so this command can run directly inside the container. You can also start a side container from the same image as long as it can reach the router address and mounts `/path/to/prometheus-data` on persistent storage.

If `--storage.tsdb.path` points to container-local disk, the data is lost when the container is removed. If it points to NFS, a persistent volume, or a job output directory, you can restart Prometheus with the same TSDB directory after training and query the historical time range in the Prometheus UI or Grafana. This is time-series replay, not full per-request trace replay; per-sample request timings still come from sample traces / debug rollout data.

## Trace Viewer

Debug rollout dumps saved with `--save-debug-rollout-data` include sample traces. The trace viewer reads SGLang timing attrs directly from those traces and uses `pd_*` fields to render synthetic `[P]` / `[D]` lanes.

```bash
python tools/trace_timeline_viewer.py /path/to/debug/rollout_0.pt
```

The default path does not require separate `ReqTimeStats(...)` logs, Loki, or a compaction tool.
