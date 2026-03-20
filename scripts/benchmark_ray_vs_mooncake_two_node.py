#!/usr/bin/env python3
"""
Two-node benchmark: Ray vs Mooncake rollout transfer using Ray actors.

Timing (full dict->put->get->dict path):
- Put: wall time from dict to handle (prepare + pack + transfer to store)
- Get: wall time from handle to dict (fetch + unpack)

Uses Ray actors for true cross-node placement:
- DataGenerator on put-node: creates rollout dict, runs put
- DataConsumer on get-node: runs get natively (handle passed via Ray, not SSH/JSON)

Prerequisites:
- Ray cluster with at least 2 nodes
- Mooncake: master + clients, MOONCAKE_MASTER, MOONCAKE_PROTOCOL=rdma

Usage:
  export MOONCAKE_MASTER=192.168.22.70:50051 MOONCAKE_PROTOCOL=rdma
  python scripts/benchmark_ray_vs_mooncake_two_node.py --put-node 192.168.22.70 --get-node 192.168.22.72 --data-size-mb 100 --num-rounds 20
  python scripts/benchmark_ray_vs_mooncake_two_node.py --backends ray mooncake ...
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import ray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slime.utils.mock_rollout import make_mock_rollout_data, get_serialized_size
from slime.utils.data_transfer import MooncakeStoreConfig
from slime.utils.rollout_hybrid_transfer import MooncakeHybridRolloutTransfer


@ray.remote(num_cpus=1, num_gpus=0)
class DataGenerator:
    def __init__(
        self,
        data_size_mb: float,
        mount_segment_size: int | None,
        mooncake_master: str = "",
    ):
        os.environ["MOONCAKE_PROTOCOL"] = os.environ.get("MOONCAKE_PROTOCOL", "rdma")
        os.environ["MOONCAKE_MASTER"] = mooncake_master or os.environ.get("MOONCAKE_MASTER", "")
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = ray.util.get_node_ip_address()
        os.environ.setdefault("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")
        os.environ.setdefault("MC_STORE_MEMCPY", "0")
        if mount_segment_size is not None:
            os.environ["MOONCAKE_MOUNT_SEGMENT_SIZE"] = str(mount_segment_size)
        self.data, self.actual_mb, self.batch, self.seq = self._make_rollout(data_size_mb)
        print(f"Generator initialized on node {ray.util.get_node_ip_address()}")
        print(f"Actual Data Size: {self.actual_mb:.2f} MB (batch={self.batch}, seq={self.seq})")

    def _make_rollout(self, target_mb: float):
        seq = 2048
        base = make_mock_rollout_data(batch_size=16, seq_len=seq, use_routing_replay=True)
        base_mb = get_serialized_size(base) / (1024 * 1024)
        batch = max(16, int(16 * (target_mb / max(1e-6, base_mb))))
        data = make_mock_rollout_data(batch_size=batch, seq_len=seq, use_routing_replay=True)
        actual_mb = get_serialized_size(data) / (1024 * 1024)
        return data, actual_mb, batch, seq

    def get_data(self):
        return self.data

    def get_info(self):
        return {"mb": self.actual_mb, "batch": self.batch, "seq": self.seq}

    def generate_ray_put(self, rounds: int):
        handles = []
        put_times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            h = ray.put(self.data)
            put_times.append((time.perf_counter() - t0) * 1000)
            handles.append(h)
        return put_times, handles

    def generate_hybrid_put(
        self,
        rounds: int,
        tensor_min_bytes: int,
        mount_segment_size: int | None,
    ):
        if not hasattr(self, "_hybrid_backend"):
            if "SLIME_PUT_FROM_SINGLE_BUFFER" in os.environ:
                del os.environ["SLIME_PUT_FROM_SINGLE_BUFFER"]
            if "SLIME_PUT_SINGLE_BUFFER_AS_MULTI" in os.environ:
                del os.environ["SLIME_PUT_SINGLE_BUFFER_AS_MULTI"]
            self._hybrid_backend = MooncakeHybridRolloutTransfer(
                tensor_min_bytes=tensor_min_bytes,
                enable_auto_cleanup=False,
                mount_segment_size=mount_segment_size,
            )
        backend = self._hybrid_backend
        handles = []
        put_times = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            h = backend.put_rollout(self.data)
            put_times.append((time.perf_counter() - t0) * 1000)
            handles.append(h)
        return handles, put_times


@ray.remote(num_cpus=1, num_gpus=0)
class DataConsumer:
    def __init__(
        self,
        mount_segment_size: int | None,
        mooncake_master: str = "",
    ):
        os.environ["MOONCAKE_PROTOCOL"] = os.environ.get("MOONCAKE_PROTOCOL", "rdma")
        os.environ["MOONCAKE_MASTER"] = mooncake_master or os.environ.get("MOONCAKE_MASTER", "")
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = ray.util.get_node_ip_address()
        os.environ.setdefault("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")
        os.environ.setdefault("MC_STORE_MEMCPY", "0")
        if mount_segment_size is not None:
            os.environ["MOONCAKE_MOUNT_SEGMENT_SIZE"] = str(mount_segment_size)
        print(f"Consumer initialized on node {ray.util.get_node_ip_address()}")

    def warmup_ray(self, handles):
        if isinstance(handles, list):
            _ = ray.get(handles[0])
        else:
            _ = ray.get(handles)

    def consume_ray_get(self, handles):
        get_times = []
        for h in handles:
            t0 = time.perf_counter()
            _ = ray.get(h)
            get_times.append((time.perf_counter() - t0) * 1000)
        return get_times

    def warmup_hybrid(
        self, handle, tensor_min_bytes: int, mount_segment_size: int | None
    ):
        if not hasattr(self, "_hybrid_backend"):
            self._hybrid_backend = MooncakeHybridRolloutTransfer(
                tensor_min_bytes=tensor_min_bytes,
                enable_auto_cleanup=False,
                mount_segment_size=mount_segment_size,
            )
        _ = self._hybrid_backend.get_rollout(handle)
        self._hybrid_backend.cleanup(handle)

    def consume_hybrid_get(
        self, handles, tensor_min_bytes: int, mount_segment_size: int | None
    ):
        if not hasattr(self, "_hybrid_backend"):
            self._hybrid_backend = MooncakeHybridRolloutTransfer(
                tensor_min_bytes=tensor_min_bytes,
                enable_auto_cleanup=False,
                mount_segment_size=mount_segment_size,
            )
        get_times = []
        for h in handles:
            t0 = time.perf_counter()
            _ = self._hybrid_backend.get_rollout(h)
            get_times.append((time.perf_counter() - t0) * 1000)
            self._hybrid_backend.cleanup(h)
        return get_times


def _resolve_node_id(ip: str) -> str | None:
    for n in ray.nodes():
        if not n.get("Alive"):
            continue
        if n.get("NodeManagerAddress") == ip:
            return n["NodeID"]
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Two-node benchmark: Ray vs Mooncake (Ray actors, handle via Ray)"
    )
    ap.add_argument("--put-node", type=str, default=None)
    ap.add_argument("--get-node", type=str, default=None)
    ap.add_argument("--data-size-mb", type=float, default=100.0)
    ap.add_argument("--num-rounds", type=int, default=30)
    ap.add_argument(
        "--backends",
        nargs="+",
        default=["ray", "mooncake"],
        choices=["ray", "mooncake"],
    )
    ap.add_argument("--tensor-min-mb", type=float, default=1.0)
    ap.add_argument(
        "--warm-up-rounds",
        type=int,
        default=24,
        help="Warmup rounds before timed runs (higher reduces Ray variance)",
    )
    ap.add_argument(
        "--discard-first",
        type=int,
        default=5,
        help="Discard first N samples after warmup (reduces cold-start effect)",
    )
    ap.add_argument(
        "--isolate-backends",
        action="store_true",
        default=True,
        help="Run each backend in separate process to avoid memory/interference (default: True)",
    )
    ap.add_argument(
        "--no-isolate-backends",
        action="store_false",
        dest="isolate_backends",
        help="Run both backends in same process (may increase Ray variance)",
    )
    ap.add_argument("--mooncake-segment-size-gb", type=float, default=None)
    ap.add_argument(
        "--trim-fraction",
        type=float,
        default=0.15,
        help="Trim top/bottom fraction for stats (0=no trim)",
    )
    ap.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format (json for machine-readable)",
    )
    args = ap.parse_args()

    # Isolate backends: run each in separate process to avoid memory/GC interference
    if (
        args.isolate_backends
        and len(args.backends) > 1
        and args.output_format != "json"
    ):
        script = Path(__file__).resolve()
        base_cmd = [
            sys.executable,
            str(script),
            "--data-size-mb",
            str(args.data_size_mb),
            "--num-rounds",
            str(args.num_rounds),
            "--warm-up-rounds",
            str(args.warm_up_rounds),
            "--discard-first",
            str(args.discard_first),
            "--no-isolate-backends",
            "--trim-fraction",
            str(args.trim_fraction),
            "--output-format",
            "json",
        ]
        if args.put_node:
            base_cmd += ["--put-node", args.put_node]
        if args.get_node:
            base_cmd += ["--get-node", args.get_node]
        if args.mooncake_segment_size_gb is not None:
            base_cmd += ["--mooncake-segment-size-gb", str(args.mooncake_segment_size_gb)]

        all_results = {}
        for backend in args.backends:
            cmd = base_cmd + ["--backends", backend]
            env = os.environ.copy()
            env.setdefault("SLIME_UNSAFE_PICKLE", "1")
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=script.parent.parent,
            )
            if proc.returncode != 0:
                print(f"Backend {backend} failed:\n{proc.stderr}", file=sys.stderr)
                sys.exit(1)
            for line in proc.stdout.strip().splitlines():
                if line.strip().startswith("{"):
                    data = json.loads(line)
                    all_results[data["backend"]] = data
                    break

        # Print merged table
        print("\n" + "=" * 80)
        print("Two-node benchmark (isolated processes for fair comparison)")
        print("=" * 80)
        print(
            f"{'Backend':<12} {'Put (ms)':<20} {'Get (ms)':<20} {'End2End (ms)':<20}"
        )
        print("-" * 80)
        for name in args.backends:
            d = all_results.get(name, {})
            pm, ps = d.get("put_mean", 0), d.get("put_std", 0)
            gm, gs = d.get("get_mean", 0), d.get("get_std", 0)
            tm, ts = d.get("e2e_mean", 0), d.get("e2e_std", 0)
            print(
                f"{name:<12} "
                f"{pm:>7.2f} ± {ps:<6.2f}   "
                f"{gm:>7.2f} ± {gs:<6.2f}   "
                f"{tm:>7.2f} ± {ts:<6.2f}"
            )
        print("-" * 80)
        return

    os.environ.setdefault("SLIME_UNSAFE_PICKLE", "1")
    os.environ.setdefault("MOONCAKE_PROTOCOL", "rdma")
    os.environ.setdefault("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")
    os.environ.setdefault("MC_STORE_MEMCPY", "0")

    overrides = {}
    if args.mooncake_segment_size_gb is not None:
        overrides["mount_segment_size"] = int(
            args.mooncake_segment_size_gb * 1024 * 1024 * 1024
        )
    cfg = MooncakeStoreConfig.load_from_env(
        overrides=overrides if overrides else None
    )
    mount_segment_size = cfg.mount_segment_size

    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)

    quiet = args.output_format == "json"
    log = (lambda *a, **kw: None) if quiet else print

    nodes = [n["NodeManagerAddress"] for n in ray.nodes() if n.get("Alive")]
    log(f"Active Ray nodes: {nodes}")
    if len(nodes) < 2:
        raise RuntimeError("Need at least 2 Ray nodes for cross-machine benchmark")

    put_node = args.put_node or nodes[0]
    get_node = args.get_node or nodes[1]
    put_node_id = _resolve_node_id(put_node)
    get_node_id = _resolve_node_id(get_node)
    if put_node_id is None:
        raise RuntimeError(f"Put node {put_node} not found in ray.nodes()")
    if get_node_id is None:
        raise RuntimeError(f"Get node {get_node} not found in ray.nodes()")
    log(f"Generator -> {put_node}, Consumer -> {get_node}")

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    mooncake_master = os.environ.get("MOONCAKE_MASTER", f"{put_node}:50051")
    gen = DataGenerator.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=put_node_id, soft=False
        ),
    ).remote(args.data_size_mb, mount_segment_size, mooncake_master)
    con = DataConsumer.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=get_node_id, soft=False
        ),
    ).remote(mount_segment_size, mooncake_master)

    info = ray.get(gen.get_info.remote())
    log(
        f"Target {args.data_size_mb} MB -> actual {info['mb']:.2f} MB "
        f"(batch={info['batch']}, seq={info['seq']}), rounds={args.num_rounds}"
    )

    tensor_min_bytes = int(args.tensor_min_mb * 1024 * 1024)
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if "ray" in args.backends:
        # Warmup: full-size batches to reduce object store variance (cold start, eviction)
        warmup_batch = min(args.warm_up_rounds, args.num_rounds)
        for _ in range(max(2, args.warm_up_rounds // max(1, warmup_batch))):
            put_times, handles = ray.get(gen.generate_ray_put.remote(warmup_batch))
            ray.get(con.consume_ray_get.remote(handles))
        time.sleep(1.0)  # Let object store settle

        # Timed run: disable GC during measurement to reduce variance
        total_rounds = args.num_rounds + args.discard_first
        put_times, handles = ray.get(gen.generate_ray_put.remote(total_rounds))
        gc.disable()
        try:
            get_times = ray.get(con.consume_ray_get.remote(handles))
        finally:
            gc.enable()
        put_arr = np.array(put_times)
        get_arr = np.array(get_times)
        discard = args.discard_first
        if discard > 0 and len(put_arr) > discard:
            put_arr = put_arr[discard:]
            get_arr = get_arr[discard:]
        results["ray"] = (put_arr, get_arr)

    if "mooncake" in args.backends:
        # Warmup
        handles, _ = ray.get(
            gen.generate_hybrid_put.remote(
                1, tensor_min_bytes, mount_segment_size
            )
        )
        ray.get(
            con.warmup_hybrid.remote(
                handles[0], tensor_min_bytes, mount_segment_size
            )
        )
        for _ in range(args.warm_up_rounds - 1):
            handles, _ = ray.get(
                gen.generate_hybrid_put.remote(
                    1, tensor_min_bytes, mount_segment_size
                )
            )
            ray.get(
                con.consume_hybrid_get.remote(
                    handles, tensor_min_bytes, mount_segment_size
                )
            )

        total_rounds = args.num_rounds + args.discard_first
        handles, put_times = ray.get(
            gen.generate_hybrid_put.remote(
                total_rounds, tensor_min_bytes, mount_segment_size
            )
        )
        get_times = ray.get(
            con.consume_hybrid_get.remote(
                handles, tensor_min_bytes, mount_segment_size
            )
        )
        put_arr = np.array(put_times)
        get_arr = np.array(get_times)
        discard = args.discard_first
        if discard > 0 and len(put_arr) > discard:
            put_arr = put_arr[discard:]
            get_arr = get_arr[discard:]
        results["mooncake"] = (put_arr, get_arr)

    def _trimmed_stats(arr: np.ndarray, frac: float):
        if frac <= 0 or len(arr) < 4:
            return arr.mean(), arr.std()
        k = max(1, int(len(arr) * frac))
        s = np.sort(arr)
        trimmed = s[k:-k] if k > 0 else s
        return float(trimmed.mean()), float(trimmed.std())

    trim = args.trim_fraction
    stats_per_backend = {}
    for name, (p, g) in results.items():
        total = p + g
        if trim > 0:
            pm, ps = _trimmed_stats(p, trim)
            gm, gs = _trimmed_stats(g, trim)
            tm, ts = _trimmed_stats(total, trim)
        else:
            pm, ps = p.mean(), p.std()
            gm, gs = g.mean(), g.std()
            tm, ts = total.mean(), total.std()
        stats_per_backend[name] = {
            "put_mean": pm,
            "put_std": ps,
            "get_mean": gm,
            "get_std": gs,
            "e2e_mean": tm,
            "e2e_std": ts,
        }

    if args.output_format == "json":
        for name, s in stats_per_backend.items():
            print(json.dumps({"backend": name, **s}))
        return

    print("\n" + "=" * 80)
    print("Two-node benchmark (Ray actors, handle via Ray)")
    print("=" * 80)
    print(
        f"{'Backend':<12} {'Put (ms)':<20} {'Get (ms)':<20} {'End2End (ms)':<20}"
    )
    print("-" * 80)
    for name, s in stats_per_backend.items():
        pm, ps = s["put_mean"], s["put_std"]
        gm, gs = s["get_mean"], s["get_std"]
        tm, ts = s["e2e_mean"], s["e2e_std"]
        print(
            f"{name:<12} "
            f"{pm:>7.2f} ± {ps:<6.2f}   "
            f"{gm:>7.2f} ± {gs:<6.2f}   "
            f"{tm:>7.2f} ± {ts:<6.2f}"
        )
    print("-" * 80)


if __name__ == "__main__":
    main()
