import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


@dataclass
class MetricsSnapshot:
    """A snapshot of metrics at a specific point in time"""

    timestamp: datetime
    metrics: Dict[str, float]

    def __repr__(self):
        return f"MetricsSnapshot(timestamp={self.timestamp.isoformat()}, metrics_count={len(self.metrics)})"


@dataclass
class WorkerStats:
    """Statistical results for a single worker over a time period"""

    worker_url: str
    time_window_seconds: float

    # Token statistics (delta over time window)
    generated_tokens: float = 0.0
    generated_tokens_rate: float = 0.0  # tokens/sec
    prompt_tokens: float = 0.0
    prompt_tokens_rate: float = 0.0
    total_requests: float = 0.0
    requests_rate: float = 0.0  # requests/sec

    # Current status (gauge metrics - latest values)
    current_throughput: float = 0.0  # tokens/sec
    current_running_requests: float = 0.0
    token_usage_ratio: float = 0.0
    cache_hit_rate: float = 0.0

    # Latency statistics (averaged over time window)
    avg_time_to_first_token: Optional[float] = None  # seconds
    avg_e2e_latency: Optional[float] = None  # seconds
    avg_time_per_output_token: Optional[float] = None  # seconds

    # Metadata
    snapshot_available: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (e.g., wandb logging)"""
        return asdict(self)


@dataclass
class AggregatedStats:
    """Aggregated statistics across all workers"""

    timestamp: datetime
    time_window_seconds: float
    num_workers: int
    num_successful_workers: int

    # Aggregated token statistics (sum across all workers)
    total_generated_tokens: float = 0.0
    total_generated_tokens_rate: float = 0.0
    total_prompt_tokens: float = 0.0
    total_requests: float = 0.0
    total_requests_rate: float = 0.0

    # Statistical summaries across workers (mean/min/max)
    throughput_mean: float = 0.0
    throughput_min: float = 0.0
    throughput_max: float = 0.0

    running_requests_mean: float = 0.0
    running_requests_min: float = 0.0
    running_requests_max: float = 0.0

    token_usage_mean: float = 0.0
    token_usage_min: float = 0.0
    token_usage_max: float = 0.0

    cache_hit_rate_mean: float = 0.0
    cache_hit_rate_min: float = 0.0
    cache_hit_rate_max: float = 0.0

    ttft_mean: Optional[float] = None
    ttft_min: Optional[float] = None
    ttft_max: Optional[float] = None

    e2e_latency_mean: Optional[float] = None
    e2e_latency_min: Optional[float] = None
    e2e_latency_max: Optional[float] = None

    time_per_token_mean: Optional[float] = None
    time_per_token_min: Optional[float] = None
    time_per_token_max: Optional[float] = None

    # Per-worker details (for debugging)
    worker_stats: List[WorkerStats] = field(default_factory=list)
    failed_workers: List[str] = field(default_factory=list)

    def to_dict(self, include_worker_details: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization (e.g., wandb logging).

        Args:
            include_worker_details: If True, include per-worker stats (can be large)

        Returns:
            Dictionary representation of the aggregated stats
        """
        result = asdict(self)

        # Convert timestamp to ISO format string
        result["timestamp"] = self.timestamp.isoformat()

        # Optionally exclude worker details to reduce size
        if not include_worker_details:
            result.pop("worker_stats", None)

        return result


class WorkerMetricsCollector:
    """
    Collects and tracks metrics for a single SGLang worker.

    This class fetches metrics from a worker's /metrics endpoint,
    records snapshots over time, and calculates incremental statistics
    between snapshots.
    """

    def __init__(self, worker_url: str):
        """
        Initialize collector for a single worker.

        Args:
            worker_url: Base URL of the worker (e.g., "http://localhost:30000")
        """
        self.worker_url = worker_url.rstrip("/")
        self.metrics_url = f"{self.worker_url}/metrics"
        self.snapshots: List[MetricsSnapshot] = []
        self.last_snapshot: Optional[MetricsSnapshot] = None

    def fetch_metrics(self) -> Dict[str, float]:
        """
        Fetch current metrics from the worker's /metrics endpoint.

        Returns:
            Dictionary mapping metric names to values

        Raises:
            requests.RequestException: If the request fails
        """
        response = requests.get(self.metrics_url, timeout=5)
        response.raise_for_status()

        metrics = {}
        for line in response.text.split("\n"):
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue

            # Parse metric line: metric_name{labels} value
            # We ignore labels for now and just extract metric_name and value
            if "{" in line:
                metric_name = line.split("{")[0]
                value_str = line.split(" ")[-1]
                try:
                    metrics[metric_name] = float(value_str)
                except ValueError:
                    continue
            elif " " in line:
                # Handle metrics without labels: metric_name value
                parts = line.split()
                if len(parts) == 2:
                    try:
                        metrics[parts[0]] = float(parts[1])
                    except ValueError:
                        continue

        return metrics

    def record_snapshot(self) -> MetricsSnapshot:
        """
        Record a snapshot of current metrics.

        Returns:
            The recorded snapshot

        Raises:
            requests.RequestException: If fetching metrics fails
        """
        metrics = self.fetch_metrics()
        snapshot = MetricsSnapshot(timestamp=datetime.now(), metrics=metrics)

        self.snapshots.append(snapshot)
        self.last_snapshot = snapshot

        return snapshot

    def calculate_stats_since(
        self, old_snapshot: MetricsSnapshot, new_snapshot: Optional[MetricsSnapshot] = None
    ) -> WorkerStats:
        """
        Calculate statistics between two snapshots.

        Args:
            old_snapshot: Earlier snapshot
            new_snapshot: Later snapshot (if None, uses the latest recorded snapshot)

        Returns:
            WorkerStats containing calculated statistics
        """
        if new_snapshot is None:
            if not self.snapshots:
                raise ValueError("No snapshots available")
            new_snapshot = self.snapshots[-1]

        time_diff = (new_snapshot.timestamp - old_snapshot.timestamp).total_seconds()

        if time_diff <= 0:
            raise ValueError(f"Invalid time window: {time_diff} seconds")

        old_m = old_snapshot.metrics
        new_m = new_snapshot.metrics

        stats = WorkerStats(worker_url=self.worker_url, time_window_seconds=time_diff)

        # Calculate Counter deltas (cumulative metrics)
        # Generated tokens
        if "sglang:generation_tokens_total" in old_m and "sglang:generation_tokens_total" in new_m:
            delta = new_m["sglang:generation_tokens_total"] - old_m["sglang:generation_tokens_total"]
            stats.generated_tokens = delta
            stats.generated_tokens_rate = delta / time_diff

        # Prompt tokens
        if "sglang:prompt_tokens_total" in old_m and "sglang:prompt_tokens_total" in new_m:
            delta = new_m["sglang:prompt_tokens_total"] - old_m["sglang:prompt_tokens_total"]
            stats.prompt_tokens = delta
            stats.prompt_tokens_rate = delta / time_diff

        # Requests
        if "sglang:num_requests_total" in old_m and "sglang:num_requests_total" in new_m:
            delta = new_m["sglang:num_requests_total"] - old_m["sglang:num_requests_total"]
            stats.total_requests = delta
            stats.requests_rate = delta / time_diff

        # Get current Gauge values (latest snapshot)
        stats.current_throughput = new_m.get("sglang:gen_throughput", 0.0)
        stats.current_running_requests = new_m.get("sglang:num_running_reqs", 0.0)
        stats.token_usage_ratio = new_m.get("sglang:token_usage", 0.0)
        stats.cache_hit_rate = new_m.get("sglang:cache_hit_rate", 0.0)

        # Calculate Histogram averages over the time window
        histogram_metrics = [
            ("sglang:time_to_first_token_seconds", "avg_time_to_first_token"),
            ("sglang:e2e_request_latency_seconds", "avg_e2e_latency"),
            ("sglang:time_per_output_token_seconds", "avg_time_per_output_token"),
        ]

        for metric_name, attr_name in histogram_metrics:
            sum_key = f"{metric_name}_sum"
            count_key = f"{metric_name}_count"

            if all(k in old_m and k in new_m for k in [sum_key, count_key]):
                sum_delta = new_m[sum_key] - old_m[sum_key]
                count_delta = new_m[count_key] - old_m[count_key]

                if count_delta > 0:
                    avg = sum_delta / count_delta
                    setattr(stats, attr_name, avg)

        return stats

    def get_stats_since_last_snapshot(self) -> WorkerStats:
        """
        Get statistics since the last recorded snapshot.

        This is a convenience method that calculates stats from the last snapshot
        to the current moment (by recording a new snapshot).

        Returns:
            WorkerStats for the time period since last snapshot

        Raises:
            ValueError: If no previous snapshot exists
        """
        if self.last_snapshot is None:
            raise ValueError("No previous snapshot available. Call record_snapshot() first.")

        new_snapshot = self.record_snapshot()
        return self.calculate_stats_since(self.last_snapshot, new_snapshot)

    def clear_history(self):
        """Clear all recorded snapshots (useful for memory management)"""
        self.snapshots.clear()
        self.last_snapshot = None


class RouterMetricsAggregator:
    """
    Aggregates metrics from all workers managed by a Slime Router.

    This class discovers workers via the router's /list_workers endpoint,
    creates a WorkerMetricsCollector for each, and provides aggregated
    statistics across all workers.
    """

    def __init__(self, router_url: str, auto_discover: bool = True):
        """
        Initialize aggregator for a router.

        Args:
            router_url: Base URL of the router (e.g., "http://localhost:8000")
            auto_discover: If True, automatically discover workers on init
        """
        self.router_url = router_url.rstrip("/")
        self.list_workers_url = f"{self.router_url}/list_workers"
        self.worker_collectors: Dict[str, WorkerMetricsCollector] = {}
        self.last_snapshot_time: Optional[datetime] = None

        if auto_discover:
            self.discover_workers()

    def discover_workers(self) -> List[str]:
        """
        Discover all workers registered with the router.

        Returns:
            List of worker URLs

        Raises:
            requests.RequestException: If the request fails
        """
        response = requests.get(self.list_workers_url, timeout=5)
        response.raise_for_status()

        data = response.json()
        worker_urls = data.get("urls", [])

        # Create collectors for new workers
        for url in worker_urls:
            if url not in self.worker_collectors:
                self.worker_collectors[url] = WorkerMetricsCollector(url)

        # Remove collectors for workers that no longer exist
        current_urls = set(worker_urls)
        removed_urls = [url for url in self.worker_collectors if url not in current_urls]
        for url in removed_urls:
            del self.worker_collectors[url]

        return worker_urls

    def record_snapshot(self, rediscover: bool = False):
        """
        Record a snapshot for all workers.

        Args:
            rediscover: If True, rediscover workers before recording
        """
        if rediscover:
            self.discover_workers()

        self.last_snapshot_time = datetime.now()

        for collector in self.worker_collectors.values():
            try:
                collector.record_snapshot()
            except Exception as e:
                print(f"[Warning] Failed to record snapshot for {collector.worker_url}: {e}")

    def get_single_worker_stats(self, worker_url: str) -> Optional[WorkerStats]:
        """
        Get statistics for a single worker (for debugging).

        Args:
            worker_url: URL of the worker

        Returns:
            WorkerStats for the worker, or None if not found or failed
        """
        collector = self.worker_collectors.get(worker_url)
        if collector is None:
            return None

        try:
            # Record a new snapshot first
            collector.record_snapshot()

            if len(collector.snapshots) < 2:
                return WorkerStats(
                    worker_url=worker_url,
                    time_window_seconds=0.0,
                    snapshot_available=False,
                    error_message="Not enough snapshots (need at least 2)",
                )

            # Calculate stats between last two snapshots
            return collector.calculate_stats_since(collector.snapshots[-2], collector.snapshots[-1])
        except Exception as e:
            return WorkerStats(
                worker_url=worker_url, time_window_seconds=0.0, snapshot_available=False, error_message=str(e)
            )

    def get_aggregated_stats(self) -> AggregatedStats:
        """
        Get aggregated statistics across all workers.

        This method collects stats from all workers and computes:
        - Sum of cumulative metrics (tokens, requests)
        - Mean/min/max of gauge metrics (throughput, usage, etc.)

        Returns:
            AggregatedStats containing aggregated metrics
        """
        if not self.worker_collectors:
            raise ValueError("No workers available. Call discover_workers() first.")

        # First, record new snapshots for all workers to ensure consistent timing
        for collector in self.worker_collectors.values():
            try:
                collector.record_snapshot()
            except Exception as e:
                print(f"[Warning] Failed to record new snapshot for {collector.worker_url}: {e}")

        timestamp = datetime.now()
        worker_stats_list = []
        failed_workers = []

        # Then calculate stats from last snapshot to the new one
        for url, collector in self.worker_collectors.items():
            try:
                if collector.last_snapshot is None or len(collector.snapshots) < 2:
                    print(f"[Warning] Not enough snapshots for {url}, skipping")
                    failed_workers.append(url)
                    continue

                # Calculate stats between the last two snapshots
                stats = collector.calculate_stats_since(
                    collector.snapshots[-2],  # Second to last (old snapshot)
                    collector.snapshots[-1],  # Last (new snapshot)
                )
                worker_stats_list.append(stats)
            except Exception as e:
                print(f"[Warning] Failed to get stats for {url}: {e}")
                failed_workers.append(url)

        if not worker_stats_list:
            raise ValueError("No worker stats available")

        # Initialize aggregated stats
        agg = AggregatedStats(
            timestamp=timestamp,
            time_window_seconds=worker_stats_list[0].time_window_seconds,
            num_workers=len(self.worker_collectors),
            num_successful_workers=len(worker_stats_list),
            worker_stats=worker_stats_list,
            failed_workers=failed_workers,
        )

        # Sum cumulative metrics
        agg.total_generated_tokens = sum(s.generated_tokens for s in worker_stats_list)
        agg.total_generated_tokens_rate = sum(s.generated_tokens_rate for s in worker_stats_list)
        agg.total_prompt_tokens = sum(s.prompt_tokens for s in worker_stats_list)
        agg.total_requests = sum(s.total_requests for s in worker_stats_list)
        agg.total_requests_rate = sum(s.requests_rate for s in worker_stats_list)

        # Compute mean/min/max for gauge metrics
        def compute_stats(values: List[float]) -> tuple:
            """Helper to compute mean, min, max"""
            if not values:
                return 0.0, 0.0, 0.0
            return sum(values) / len(values), min(values), max(values)

        # Throughput
        throughputs = [s.current_throughput for s in worker_stats_list]
        agg.throughput_mean, agg.throughput_min, agg.throughput_max = compute_stats(throughputs)

        # Running requests
        running_reqs = [s.current_running_requests for s in worker_stats_list]
        agg.running_requests_mean, agg.running_requests_min, agg.running_requests_max = compute_stats(running_reqs)

        # Token usage
        token_usage = [s.token_usage_ratio for s in worker_stats_list]
        agg.token_usage_mean, agg.token_usage_min, agg.token_usage_max = compute_stats(token_usage)

        # Cache hit rate
        cache_rates = [s.cache_hit_rate for s in worker_stats_list]
        agg.cache_hit_rate_mean, agg.cache_hit_rate_min, agg.cache_hit_rate_max = compute_stats(cache_rates)

        # Latency metrics (only if available from all workers)
        ttft_values = [s.avg_time_to_first_token for s in worker_stats_list if s.avg_time_to_first_token is not None]
        if ttft_values:
            agg.ttft_mean, agg.ttft_min, agg.ttft_max = compute_stats(ttft_values)

        e2e_values = [s.avg_e2e_latency for s in worker_stats_list if s.avg_e2e_latency is not None]
        if e2e_values:
            agg.e2e_latency_mean, agg.e2e_latency_min, agg.e2e_latency_max = compute_stats(e2e_values)

        tpt_values = [
            s.avg_time_per_output_token for s in worker_stats_list if s.avg_time_per_output_token is not None
        ]
        if tpt_values:
            agg.time_per_token_mean, agg.time_per_token_min, agg.time_per_token_max = compute_stats(tpt_values)

        return agg

    def print_aggregated_stats(self, stats: AggregatedStats):
        """Pretty print aggregated statistics"""
        print(f"\n{'='*80}")
        print(f"📊 Slime Router Metrics Summary")
        print(f"{'='*80}")
        print(f"Timestamp: {stats.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Time Window: {stats.time_window_seconds:.1f} seconds")
        print(f"Workers: {stats.num_successful_workers}/{stats.num_workers} successful")

        if stats.failed_workers:
            print(f"⚠️  Failed Workers: {', '.join(stats.failed_workers)}")

        print(f"\n{'─'*80}")
        print(f"🔢 Total Token Statistics (Sum Across All Workers)")
        print(f"{'─'*80}")
        print(
            f"  Generated Tokens:    {stats.total_generated_tokens:>12,.0f}  ({stats.total_generated_tokens_rate:>8,.1f} tokens/sec)"
        )
        print(f"  Prompt Tokens:       {stats.total_prompt_tokens:>12,.0f}")
        print(f"  Total Requests:      {stats.total_requests:>12,.0f}  ({stats.total_requests_rate:>8,.1f} req/sec)")

        print(f"\n{'─'*80}")
        print(f"⚡ Per-Worker Statistics (Mean / Min / Max)")
        print(f"{'─'*80}")
        print(
            f"  Current Throughput:  {stats.throughput_mean:>8.1f} / {stats.throughput_min:>8.1f} / {stats.throughput_max:>8.1f}  tokens/sec"
        )
        print(
            f"  Running Requests:    {stats.running_requests_mean:>8.1f} / {stats.running_requests_min:>8.1f} / {stats.running_requests_max:>8.1f}"
        )
        print(
            f"  Token Usage:         {stats.token_usage_mean:>7.1%} / {stats.token_usage_min:>7.1%} / {stats.token_usage_max:>7.1%}"
        )
        print(
            f"  Cache Hit Rate:      {stats.cache_hit_rate_mean:>7.1%} / {stats.cache_hit_rate_min:>7.1%} / {stats.cache_hit_rate_max:>7.1%}"
        )

        if stats.ttft_mean is not None:
            print(f"\n{'─'*80}")
            print(f"⏱️  Latency Statistics (Mean / Min / Max)")
            print(f"{'─'*80}")
            print(
                f"  Time to First Token: {stats.ttft_mean:>8.3f} / {stats.ttft_min:>8.3f} / {stats.ttft_max:>8.3f}  seconds"
            )

            if stats.e2e_latency_mean is not None:
                print(
                    f"  End-to-End Latency:  {stats.e2e_latency_mean:>8.3f} / {stats.e2e_latency_min:>8.3f} / {stats.e2e_latency_max:>8.3f}  seconds"
                )

            if stats.time_per_token_mean is not None:
                print(
                    f"  Time per Token:      {stats.time_per_token_mean:>8.3f} / {stats.time_per_token_min:>8.3f} / {stats.time_per_token_max:>8.3f}  seconds"
                )

        print(f"\n{'='*80}\n")

    def print_worker_details(self):
        """Print detailed stats for each individual worker (for debugging)"""
        print(f"\n{'='*80}")
        print(f"🔍 Individual Worker Details")
        print(f"{'='*80}\n")

        for i, (url, collector) in enumerate(self.worker_collectors.items(), 1):
            print(f"Worker {i}: {url}")
            print(f"{'─'*80}")

            try:
                if len(collector.snapshots) < 2:
                    print(f"  ⚠️  Not enough snapshots (need at least 2)")
                    print()
                    continue

                # Use the already recorded snapshots (from get_aggregated_stats)
                stats = collector.calculate_stats_since(collector.snapshots[-2], collector.snapshots[-1])

                print(
                    f"  Generated Tokens:    {stats.generated_tokens:>10,.0f}  ({stats.generated_tokens_rate:>8,.1f} tokens/sec)"
                )
                print(f"  Prompt Tokens:       {stats.prompt_tokens:>10,.0f}")
                print(f"  Requests:            {stats.total_requests:>10,.0f}  ({stats.requests_rate:>8,.1f} req/sec)")
                print(f"  Current Throughput:  {stats.current_throughput:>10.1f} tokens/sec")
                print(f"  Running Requests:    {stats.current_running_requests:>10.0f}")
                print(f"  Token Usage:         {stats.token_usage_ratio:>9.1%}")
                print(f"  Cache Hit Rate:      {stats.cache_hit_rate:>9.1%}")

                if stats.avg_time_to_first_token is not None:
                    print(f"  Avg TTFT:            {stats.avg_time_to_first_token:>10.3f} seconds")
                if stats.avg_e2e_latency is not None:
                    print(f"  Avg E2E Latency:     {stats.avg_e2e_latency:>10.3f} seconds")

            except Exception as e:
                print(f"  Error: {e}")

            print()


def main():
    """
    Example usage for debugging.

    This demonstrates both single-worker collection and multi-worker aggregation.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Collect SGLang metrics from Slime Router")
    parser.add_argument("--router-url", type=str, default="http://localhost:8000", help="Router URL")
    parser.add_argument("--worker-url", type=str, default=None, help="Single worker URL (for debugging single worker)")
    parser.add_argument("--interval", type=int, default=30, help="Collection interval in seconds")
    parser.add_argument("--iterations", type=int, default=10, help="Number of collection iterations")
    parser.add_argument("--show-workers", action="store_true", help="Show individual worker details")

    args = parser.parse_args()

    if args.worker_url:
        # Single worker mode (for debugging)
        print(f"🔍 Single Worker Mode: {args.worker_url}")
        print(f"Collecting metrics every {args.interval} seconds for {args.iterations} iterations\n")

        collector = WorkerMetricsCollector(args.worker_url)

        try:
            collector.record_snapshot()
            print("Initial snapshot recorded. Waiting for next collection...")

            for i in range(args.iterations):
                time.sleep(args.interval)

                stats = collector.get_stats_since_last_snapshot()

                print(f"\n{'='*60}")
                print(f"Iteration {i+1}/{args.iterations}")
                print(f"{'='*60}")
                print(f"Time Window: {stats.time_window_seconds:.1f} seconds")
                print(f"Generated Tokens: {stats.generated_tokens:.0f} ({stats.generated_tokens_rate:.1f} tokens/sec)")
                print(f"Current Throughput: {stats.current_throughput:.1f} tokens/sec")
                print(f"Running Requests: {stats.current_running_requests:.0f}")
                print(f"Cache Hit Rate: {stats.cache_hit_rate:.1%}")

        except KeyboardInterrupt:
            print("\n\nStopped by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()

    else:
        # Router aggregation mode (production)
        print(f"🌐 Router Aggregation Mode: {args.router_url}")
        print(f"Collecting metrics every {args.interval} seconds for {args.iterations} iterations\n")

        aggregator = RouterMetricsAggregator(args.router_url)

        try:
            print(f"Discovered {len(aggregator.worker_collectors)} workers:")
            for url in aggregator.worker_collectors.keys():
                print(f"  - {url}")
            print()

            aggregator.record_snapshot()
            print("Initial snapshot recorded. Waiting for next collection...\n")

            for i in range(args.iterations):
                time.sleep(args.interval)

                print(f"\n{'#'*80}")
                print(f"# Iteration {i+1}/{args.iterations}")
                print(f"{'#'*80}")

                stats = aggregator.get_aggregated_stats()
                aggregator.print_aggregated_stats(stats)

                if args.show_workers:
                    aggregator.print_worker_details()

        except KeyboardInterrupt:
            print("\n\nStopped by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
