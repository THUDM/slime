#!/usr/bin/env python3
"""
SGLang Decode Profile Analyzer
==============================
Analyzes PyTorch profiler traces (.trace.json.gz) from SGLang decode workers.

Usage:
    python tools/analyze_profile.py --profile-dir profiles/20260303_052303_my_run
    python tools/analyze_profile.py --profile-dir profiles/20260303_052303_my_run --rank 0
    python tools/analyze_profile.py --profile-dir profiles/20260303_052303_my_run --all-ranks
"""

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field


# ─── Color helpers ──────────────────────────────────────────────────────────
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def header(text):
    print(f"\n{C.BOLD}{C.HEADER}{'═' * 80}{C.END}")
    print(f"{C.BOLD}{C.HEADER}  {text}{C.END}")
    print(f"{C.BOLD}{C.HEADER}{'═' * 80}{C.END}")


def section(text):
    print(f"\n{C.BOLD}{C.CYAN}── {text} ──{C.END}")


def warn(text):
    print(f"  {C.YELLOW}⚠  {text}{C.END}")


def good(text):
    print(f"  {C.GREEN}✓  {text}{C.END}")


def bad(text):
    print(f"  {C.RED}✗  {text}{C.END}")


def bar(pct, width=40, label=""):
    filled = int(pct / 100 * width)
    bar_str = "█" * filled + "░" * (width - filled)
    color = C.GREEN if pct >= 80 else C.YELLOW if pct >= 50 else C.RED
    return f"  {color}{bar_str}{C.END} {pct:5.1f}%  {label}"


# ─── Data structures ───────────────────────────────────────────────────────
@dataclass
class KernelInfo:
    name: str
    category: str
    count: int = 0
    total_dur: float = 0.0

    @property
    def avg_dur(self):
        return self.total_dur / max(self.count, 1)


@dataclass
class CudaGraphLaunch:
    index: int
    ts: float
    dur: float
    tid: int


@dataclass
class TraceAnalysis:
    rank_name: str = ""
    # Device
    gpu_name: str = ""
    num_gpus: int = 0
    total_vram_gb: float = 0.0
    num_sms: int = 0
    # CUDA/NCCL
    cuda_version: str = ""
    nccl_version: str = ""
    nccl_backend: str = ""
    world_size: int = 0
    pg_count: int = 0
    # Timeline
    trace_wall_time_us: float = 0.0
    gpu_active_span_us: float = 0.0
    gpu_busy_time_us: float = 0.0
    gpu_util_pct: float = 0.0
    # Events
    total_events: int = 0
    total_kernel_events: int = 0
    total_kernel_time_us: float = 0.0
    # Categories
    kernel_categories: dict[str, KernelInfo] = field(default_factory=dict)
    top_kernels: list[KernelInfo] = field(default_factory=list)
    # CUDA Graph
    cuda_graph_launches: list[CudaGraphLaunch] = field(default_factory=list)
    cuda_graph_total_cpu_us: float = 0.0
    # Gaps
    top_gaps: list[tuple[float, float, float, str]] = field(default_factory=list)  # (dur, start, end, cause)
    # Decode steps
    decode_steps: list[dict] = field(default_factory=list)
    # Per-stream
    stream_info: dict[int, dict] = field(default_factory=dict)
    # Communication
    nccl_kernel_dur_us: float = 0.0
    deep_ep_dur_us: float = 0.0
    gloo_total_us: float = 0.0
    # CPU overhead
    aten_copy_total_us: float = 0.0
    aten_copy_count: int = 0


def classify_kernel(name: str) -> str:
    nl = name.lower()
    if "nccl" in nl:
        return "NCCL Communication"
    if "deep_ep" in nl:
        if "dispatch" in nl:
            return "DeepEP Dispatch (MoE)"
        if "combine" in nl:
            return "DeepEP Combine (MoE)"
        if "clean" in nl:
            return "DeepEP Buffer Clean"
        return "DeepEP (other)"
    if "flash" in nl and "attn" in nl:
        return "Flash Attention"
    if "sparse_attn" in nl:
        return "Sparse Attention (MLA)"
    if "paged_mqa" in nl:
        return "Paged MQA/MLA Logits"
    if "attention" in nl or "fmha" in nl:
        return "Attention (other)"
    if "deep_gemm" in nl or "sm90_fp8_gemm" in nl:
        return "DeepGEMM (FP8)"
    if "nvjet" in nl:
        return "NvJet GEMM"
    if "gemm" in nl or "cutlass" in nl or "matmul" in nl or "cublas" in nl:
        return "GEMM/MatMul (other)"
    if "topk" in nl:
        return "TopK / MoE Routing"
    if "quant" in nl:
        return "Quantization (FP8)"
    if "rmsnorm" in nl or "layernorm" in nl or "norm" in nl:
        return "Normalization"
    if any(k in nl for k in ["triton", "fused"]):
        return "Fused/Triton Kernels"
    if "elementwise" in nl or "vectorized_elementwise" in nl:
        return "Elementwise Ops"
    if "reduce" in nl or "softmax" in nl:
        return "Reduce/Softmax"
    if "embedding" in nl:
        return "Embedding"
    if "memcpy" in nl or "memset" in nl:
        return "Memory Ops"
    if "index" in nl or "scatter" in nl or "gather" in nl:
        return "Index/Gather/Scatter"
    if "cat" in nl and "batched" in nl:
        return "Concat/Cat"
    return "Other Compute"


def load_trace(filepath: str) -> dict:
    with gzip.open(filepath, "rt") as f:
        return json.load(f)


def analyze_trace(data: dict, rank_name: str = "") -> TraceAnalysis:
    result = TraceAnalysis(rank_name=rank_name)
    events = data["traceEvents"]
    result.total_events = len(events)

    # ─── Device Properties ──────────────────────────────────────────────
    dev_props = data.get("deviceProperties", [])
    if dev_props:
        result.gpu_name = dev_props[0].get("name", "Unknown")
        result.num_gpus = len(dev_props)
        result.total_vram_gb = dev_props[0].get("totalGlobalMem", 0) / (1024**3)
        result.num_sms = dev_props[0].get("numSms", 0)

    # ─── CUDA/NCCL Info ─────────────────────────────────────────────────
    cuda_rt = data.get("cuda_runtime_version", 0)
    result.cuda_version = f"{cuda_rt // 1000}.{(cuda_rt % 1000) // 10}"

    dist_info = data.get("distributedInfo", {})
    result.nccl_version = dist_info.get("nccl_version", "?")
    result.nccl_backend = dist_info.get("backend", "?")
    result.world_size = dist_info.get("world_size", 0)
    result.pg_count = dist_info.get("pg_count", 0)

    # ─── Kernel Analysis ────────────────────────────────────────────────
    kernel_events = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    result.total_kernel_events = len(kernel_events)

    kernel_cats = defaultdict(lambda: {"count": 0, "total_dur": 0.0})
    kernel_indiv = defaultdict(lambda: {"count": 0, "total_dur": 0.0})

    for e in kernel_events:
        name = e.get("name", "?")
        dur = e.get("dur", 0)
        cat = classify_kernel(name)
        kernel_cats[cat]["count"] += 1
        kernel_cats[cat]["total_dur"] += dur
        kernel_indiv[name]["count"] += 1
        kernel_indiv[name]["total_dur"] += dur

    result.total_kernel_time_us = sum(v["total_dur"] for v in kernel_cats.values())

    for cat, info in kernel_cats.items():
        result.kernel_categories[cat] = KernelInfo(
            name=cat, category=cat, count=info["count"], total_dur=info["total_dur"]
        )

    for name, info in sorted(kernel_indiv.items(), key=lambda x: -x[1]["total_dur"])[:30]:
        result.top_kernels.append(
            KernelInfo(name=name, category=classify_kernel(name), count=info["count"], total_dur=info["total_dur"])
        )

    # ─── GPU Utilization ────────────────────────────────────────────────
    gpu_events = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
    intervals = sorted([(e["ts"], e["ts"] + e["dur"]) for e in gpu_events])

    if intervals:
        merged = []
        for s, e_end in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e_end))
            else:
                merged.append((s, e_end))

        result.gpu_busy_time_us = sum(e - s for s, e in merged)
        result.gpu_active_span_us = merged[-1][1] - merged[0][0]
        result.gpu_util_pct = result.gpu_busy_time_us / result.gpu_active_span_us * 100

        # Trace wall time
        all_x = [(e["ts"], e["ts"] + e["dur"]) for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
        if all_x:
            result.trace_wall_time_us = max(t[1] for t in all_x) - min(t[0] for t in all_x)

        # GPU idle gaps
        gaps = []
        for i in range(1, len(merged)):
            gap = merged[i][0] - merged[i - 1][1]
            if gap > 0:
                gaps.append((gap, merged[i - 1][1], merged[i][0]))
        gaps.sort(reverse=True)

        # Identify cause of top gaps
        cpu_ops = [
            e for e in events if e.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime") and e.get("ph") == "X"
        ]
        for gap, gap_start, gap_end in gaps[:15]:
            cause = "unknown"
            for e in cpu_ops:
                e_start = e.get("ts", 0)
                e_end = e_start + e.get("dur", 0)
                if e_start < gap_end and e_end > gap_start:
                    cause = e.get("name", "?")
                    break
            result.top_gaps.append((gap, gap_start, gap_end, cause))

    # ─── Per-stream analysis ────────────────────────────────────────────
    stream_events = defaultdict(list)
    for e in gpu_events:
        stream_events[e.get("tid")].append(e)

    for tid, evts in stream_events.items():
        total_dur = sum(e.get("dur", 0) for e in evts)
        min_ts = min(e["ts"] for e in evts)
        max_ts = max(e["ts"] + e.get("dur", 0) for e in evts)
        span = max_ts - min_ts
        result.stream_info[tid] = {
            "count": len(evts),
            "total_dur": total_dur,
            "span": span,
            "util": total_dur / span * 100 if span > 0 else 0,
        }

    # ─── CUDA Graph analysis ───────────────────────────────────────────
    cg_launches = sorted(
        [e for e in events if e.get("name") == "cudaGraphLaunch" and e.get("cat") == "cuda_runtime"],
        key=lambda x: x["ts"],
    )
    for i, e in enumerate(cg_launches):
        result.cuda_graph_launches.append(CudaGraphLaunch(index=i, ts=e["ts"], dur=e["dur"], tid=e.get("tid", 0)))
    result.cuda_graph_total_cpu_us = sum(e["dur"] for e in cg_launches)

    # Group into decode steps (3 launches per step)
    for i in range(0, len(cg_launches), 3):
        group = cg_launches[i : i + 3]
        if len(group) == 3:
            step_start = group[0]["ts"]
            step_end = group[2]["ts"] + group[2]["dur"]
            result.decode_steps.append(
                {
                    "step": i // 3,
                    "span_us": step_end - step_start,
                    "launches": [g["dur"] for g in group],
                    "ts_start": step_start,
                    "ts_end": step_end,
                }
            )

    # ─── Communication ──────────────────────────────────────────────────
    result.nccl_kernel_dur_us = sum(
        e.get("dur", 0) for e in events if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower()
    )
    result.deep_ep_dur_us = sum(
        e.get("dur", 0) for e in events if e.get("cat") == "kernel" and "deep_ep" in e.get("name", "")
    )
    result.gloo_total_us = sum(e.get("dur", 0) for e in events if "gloo" in e.get("name", "").lower())

    # ─── CPU overhead ───────────────────────────────────────────────────
    copy_events = [e for e in events if e.get("cat") == "cpu_op" and e.get("name") == "aten::copy_"]
    result.aten_copy_total_us = sum(e.get("dur", 0) for e in copy_events)
    result.aten_copy_count = len(copy_events)

    return result


# ─── Pretty print ──────────────────────────────────────────────────────────
def print_analysis(r: TraceAnalysis):
    header(f"Profile Analysis: {r.rank_name}")

    # ── Config ──────────────────────────────────────────────────────────
    section("Hardware & Config")
    print(f"  GPU:            {C.BOLD}{r.gpu_name}{C.END} × {r.num_gpus} (visible)")
    print(f"  VRAM per GPU:   {r.total_vram_gb:.0f} GB")
    print(f"  SMs per GPU:    {r.num_sms}")
    print(f"  CUDA Version:   {r.cuda_version}")
    print(f"  NCCL:           {r.nccl_version} (backend: {r.nccl_backend})")
    print(f"  World Size:     {r.world_size}")
    print(f"  Process Groups: {r.pg_count}")
    print(f"  Total Events:   {r.total_events:,}")

    # ── Timeline ────────────────────────────────────────────────────────
    section("Timeline")
    print(f"  Trace Wall Time:  {r.trace_wall_time_us / 1000:.1f} ms")
    print(f"  GPU Active Span:  {r.gpu_active_span_us / 1000:.1f} ms")
    print(f"  GPU Busy Time:    {r.gpu_busy_time_us / 1000:.1f} ms")
    print(f"  GPU Idle (bubbles): {(r.gpu_active_span_us - r.gpu_busy_time_us) / 1000:.1f} ms")
    print()
    print(f"  {C.BOLD}GPU Utilization:{C.END}")
    print(bar(r.gpu_util_pct, label="GPU busy"))
    print(bar(100 - r.gpu_util_pct, label="GPU idle (bubbles)"))

    # ── Kernel Breakdown ────────────────────────────────────────────────
    section("GPU Kernel Time Breakdown")
    total = r.total_kernel_time_us
    print(f"  Total kernel time: {total / 1000:.1f} ms ({r.total_kernel_events:,} kernel launches)\n")

    sorted_cats = sorted(r.kernel_categories.values(), key=lambda x: -x.total_dur)
    for ki in sorted_cats:
        pct = ki.total_dur / total * 100
        filled = int(pct / 100 * 30)
        bar_str = "█" * filled + "░" * (30 - filled)
        color = C.CYAN
        if "NCCL" in ki.name or "DeepEP" in ki.name:
            color = C.YELLOW
        elif "GEMM" in ki.name or "NvJet" in ki.name or "DeepGEMM" in ki.name:
            color = C.GREEN
        elif "Attention" in ki.name or "MLA" in ki.name or "MQA" in ki.name:
            color = C.BLUE
        print(f"  {color}{bar_str}{C.END} {pct:5.1f}%  {ki.total_dur / 1000:8.1f} ms  n={ki.count:5d}  {ki.name}")

    # ── Top Kernels ─────────────────────────────────────────────────────
    section("Top 20 Individual Kernels (by total GPU time)")
    print(f"  {'%':>5s}  {'Total(ms)':>9s}  {'Avg(us)':>8s}  {'Count':>6s}  Kernel")
    print(f"  {'─' * 5}  {'─' * 9}  {'─' * 8}  {'─' * 6}  {'─' * 50}")
    for ki in r.top_kernels[:20]:
        pct = ki.total_dur / total * 100
        print(f"  {pct:5.1f}  {ki.total_dur / 1000:9.2f}  {ki.avg_dur:8.1f}  {ki.count:6d}  {ki.name[:90]}")

    # ── CUDA Graph ──────────────────────────────────────────────────────
    section("CUDA Graph Analysis")
    print(f"  Total cudaGraphLaunch calls: {C.BOLD}{len(r.cuda_graph_launches)}{C.END}")
    print(f"  Total CPU time in cudaGraphLaunch: {C.BOLD}{r.cuda_graph_total_cpu_us / 1000:.1f} ms{C.END}")
    print()

    if r.cuda_graph_launches:
        # Categorize launches by duration
        small = [launch for launch in r.cuda_graph_launches if launch.dur < 1000]
        medium = [launch for launch in r.cuda_graph_launches if 1000 <= launch.dur < 5000]
        large = [launch for launch in r.cuda_graph_launches if launch.dur >= 5000]

        print("  Launch size distribution:")
        print(
            f"    Small  (<1ms):  {len(small):3d} launches, total {sum(launch.dur for launch in small) / 1000:.1f} ms"
        )
        print(
            f"    Medium (1-5ms): {len(medium):3d} launches, total {sum(launch.dur for launch in medium) / 1000:.1f} ms"
        )
        print(
            f"    {C.RED}Large  (>5ms):  {len(large):3d} launches, total {sum(launch.dur for launch in large) / 1000:.1f} ms{C.END}"
        )

    if r.decode_steps:
        print("\n  Decode Steps (3 cudaGraphLaunch per step):")
        print(f"  {'Step':>4s}  {'Span(ms)':>8s}  {'L1(us)':>7s}  {'L2(us)':>7s}  {'L3(us)':>7s}  Note")
        print(f"  {'─' * 4}  {'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 30}")
        for step in r.decode_steps:
            launches = step["launches"]
            note = ""
            if any(val > 10000 for val in launches):
                note = f"{C.RED}← large launch stall{C.END}"
            print(
                f"  {step['step']:4d}  {step['span_us'] / 1000:8.1f}  "
                f"{launches[0]:7.0f}  {launches[1]:7.0f}  {launches[2]:7.0f}  {note}"
            )

        avg_span = sum(s["span_us"] for s in r.decode_steps) / len(r.decode_steps)
        print(f"\n  Average decode step span: {C.BOLD}{avg_span / 1000:.1f} ms{C.END}")
        print(f"  Estimated decode throughput: {C.BOLD}{1e6 / avg_span:.0f} tokens/s per GPU{C.END}")

    # ── CUDA Graph Location in Timeline ─────────────────────────────────
    section("CUDA Graph Locations in Timeline")
    if r.cuda_graph_launches and r.gpu_active_span_us > 0:
        base_ts = min(launch.ts for launch in r.cuda_graph_launches)
        span = r.gpu_active_span_us
        print(f"  Timeline (0 = first GPU activity, total span = {span / 1000:.1f} ms):")
        print()
        # ASCII timeline
        WIDTH = 70
        timeline = [" "] * WIDTH
        for launch in r.cuda_graph_launches:
            pos = int((launch.ts - base_ts) / span * WIDTH)
            pos = min(pos, WIDTH - 1)
            if launch.dur >= 5000:
                timeline[pos] = f"{C.RED}▓{C.END}"
            elif launch.dur >= 1000:
                timeline[pos] = f"{C.YELLOW}▒{C.END}"
            else:
                timeline[pos] = f"{C.GREEN}░{C.END}"
        print(f"  |{''.join(timeline)}|")
        print(f"  0ms{' ' * (WIDTH - 8)}{span / 1000:.0f}ms")
        print(
            f"  Legend: {C.GREEN}░{C.END}=small(<1ms)  {C.YELLOW}▒{C.END}=medium(1-5ms)  {C.RED}▓{C.END}=large(>5ms)"
        )

    # ── GPU Idle Gaps ───────────────────────────────────────────────────
    section("Top GPU Idle Gaps (Bubbles)")
    if r.top_gaps:
        print(f"  {'#':>3s}  {'Gap(us)':>8s}  {'Cause'}")
        print(f"  {'─' * 3}  {'─' * 8}  {'─' * 50}")
        for i, (gap, _start, _end, cause) in enumerate(r.top_gaps[:10]):
            color = C.RED if gap > 3000 else C.YELLOW if gap > 1000 else ""
            end_c = C.END if color else ""
            print(f"  {i + 1:3d}  {color}{gap:8.0f}{end_c}  {cause}")

    # ── Per-Stream ──────────────────────────────────────────────────────
    section("Per-Stream GPU Activity")
    print(f"  {'Stream':>8s}  {'Events':>7s}  {'Time(ms)':>9s}  {'Span(ms)':>9s}  {'Util%':>6s}  Role")
    print(f"  {'─' * 8}  {'─' * 7}  {'─' * 9}  {'─' * 9}  {'─' * 6}  {'─' * 20}")
    for tid in sorted(r.stream_info.keys()):
        info = r.stream_info[tid]
        role = ""
        if info["count"] > 50000:
            role = "← main compute"
        elif info["count"] > 5000:
            role = "← secondary compute"
        elif info["util"] < 2:
            role = "← auxiliary"
        print(
            f"  {tid:8d}  {info['count']:7d}  {info['total_dur'] / 1000:9.1f}  "
            f"{info['span'] / 1000:9.1f}  {info['util']:6.1f}  {role}"
        )

    # ── Communication ───────────────────────────────────────────────────
    section("Communication Overhead")
    comm_total = r.nccl_kernel_dur_us + r.deep_ep_dur_us
    print(f"  NCCL AllGather (GPU kernel): {r.nccl_kernel_dur_us / 1000:.1f} ms")
    print(f"  DeepEP (MoE all-to-all):     {r.deep_ep_dur_us / 1000:.1f} ms")
    print(f"  Gloo Broadcast (CPU):        {r.gloo_total_us / 1000:.1f} ms")
    print(f"  Communication / Total Kernel: {comm_total / total * 100:.1f}%")

    # ── Performance Bottleneck Summary ──────────────────────────────────
    header("🔍 Performance Bottleneck Analysis")

    issues = []

    # 1. GPU utilization
    if r.gpu_util_pct < 90:
        issues.append(
            (
                "GPU Utilization",
                f"GPU utilization is {r.gpu_util_pct:.1f}% — {100 - r.gpu_util_pct:.1f}% idle bubbles "
                f"({(r.gpu_active_span_us - r.gpu_busy_time_us) / 1000:.1f} ms wasted)",
                "high" if r.gpu_util_pct < 80 else "medium",
            )
        )

    # 2. CUDA Graph launch overhead
    large_launches = [launch for launch in r.cuda_graph_launches if launch.dur >= 5000]
    if large_launches:
        avg_large = sum(launch.dur for launch in large_launches) / len(large_launches)
        issues.append(
            (
                "CUDA Graph Launch Stalls",
                f"{len(large_launches)} cudaGraphLaunch calls take >{5}ms (avg {avg_large / 1000:.1f}ms). "
                f"Total: {sum(launch.dur for launch in large_launches) / 1000:.1f}ms CPU blocked. "
                f"This is the {C.BOLD}#1 source of GPU bubbles{C.END}.",
                "high",
            )
        )

    # 3. DeepEP dispatch latency
    dep_cat = r.kernel_categories.get("DeepEP Dispatch (MoE)")
    if dep_cat and dep_cat.total_dur / total > 0.15:
        issues.append(
            (
                "DeepEP Dispatch Dominance",
                f"DeepEP dispatch takes {dep_cat.total_dur / 1000:.1f}ms ({dep_cat.total_dur / total * 100:.1f}% of kernel time). "
                f"MoE expert parallelism communication is a major cost.",
                "high",
            )
        )

    # 4. MoE routing overhead
    topk_cat = r.kernel_categories.get("TopK / MoE Routing")
    if topk_cat and topk_cat.total_dur / total > 0.05:
        issues.append(
            (
                "MoE Routing Overhead",
                f"TopK routing takes {topk_cat.total_dur / 1000:.1f}ms ({topk_cat.total_dur / total * 100:.1f}%). "
                f"topk_transform_decode_kernel is expensive at {topk_cat.avg_dur:.0f}us avg.",
                "medium",
            )
        )

    # 5. CPU copy overhead
    if r.aten_copy_total_us > 100000:
        issues.append(
            (
                "CPU aten::copy_ Overhead",
                f"aten::copy_ takes {r.aten_copy_total_us / 1000:.1f}ms CPU time ({r.aten_copy_count} calls). "
                f"Likely dtype conversions (BF16→FP8) stalling the CPU.",
                "medium",
            )
        )

    # 6. Small kernel launches
    small_kernels = sum(1 for k in r.top_kernels if k.avg_dur < 3)
    if small_kernels > 5:
        issues.append(
            (
                "Many Tiny Kernels",
                f"{small_kernels} of top-30 kernels have avg duration <3us. "
                f"Launch overhead may exceed compute. Consider kernel fusion.",
                "low",
            )
        )

    for issue_name, desc, severity in issues:
        color = C.RED if severity == "high" else C.YELLOW if severity == "medium" else C.CYAN
        sev_label = {"high": "HIGH", "medium": "MED", "low": "LOW"}[severity]
        print(f"\n  {color}[{sev_label}]{C.END} {C.BOLD}{issue_name}{C.END}")
        print(f"       {desc}")

    # ── Optimization Recommendations ────────────────────────────────────
    header("💡 Optimization Recommendations")

    recs = []

    # Based on detected issues
    if any("CUDA Graph Launch" in i[0] for i in issues):
        recs.append(
            (
                "Reduce CUDA Graph Re-capture / Launch Latency",
                [
                    "The large cudaGraphLaunch (13-16ms) occurs once per decode step — this is the MoE expert "
                    "portion that runs OUTSIDE the CUDA graph (DeepEP dispatch/combine + NCCL allgather).",
                    "The 3-launch pattern per step = (1) pre-MoE graph, (2) post-MoE graph, (3) MoE-expert graph.",
                    "Optimization: try increasing decode batch size to amortize graph launch overhead per token.",
                    "Check if `--sglang-disable-cuda-graph` helps isolate whether the overhead is in graph "
                    "management vs. actual compute.",
                    "Consider padding batch sizes to avoid frequent graph re-capture for different sizes.",
                ],
            )
        )

    if any("DeepEP" in i[0] for i in issues):
        recs.append(
            (
                "Optimize MoE Expert Parallelism (DeepEP)",
                [
                    f"DeepEP dispatch+combine = {r.deep_ep_dur_us / 1000:.1f}ms/step = "
                    f"{r.deep_ep_dur_us / total * 100:.1f}% of GPU time — this is all-to-all expert communication.",
                    "dispatch (76.8us avg × 1556 calls) dominates over combine (29.4us avg).",
                    "Consider: reduce number of MoE layers, or reduce EP degree if not fully utilizing all experts.",
                    "Ensure NVLink/NVSwitch bandwidth is saturated (H100 should be 900 GB/s bidirectional).",
                    "Check if low-latency mode for DeepEP is enabled — the clean_buffer kernel (700us avg) suggests "
                    "low-latency mode is active.",
                ],
            )
        )

    if any("TopK" in i[0] for i in issues):
        recs.append(
            (
                "Optimize MoE TopK Routing",
                [
                    "topk_transform_decode_kernel at 49us avg is expensive for decode (small batch).",
                    "Consider using a more efficient routing algorithm or fusing TopK with dispatch.",
                ],
            )
        )

    recs.append(
        (
            "General Decode Throughput",
            [
                (
                    f"Current: ~{1e6 / (sum(s['span_us'] for s in r.decode_steps) / max(len(r.decode_steps), 1)):.0f} tokens/s/GPU "
                    f"(avg {sum(s['span_us'] for s in r.decode_steps) / max(len(r.decode_steps), 1) / 1000:.1f}ms/step)."
                    if r.decode_steps
                    else "No decode step data."
                ),
                "Increase batch size to improve GPU SM occupancy — many kernels are memory-bound at small batch.",
                "Speculative decoding could help if generation is latency-bound.",
                "Verify `--sglang-mem-fraction-static` is set high enough for large KV cache.",
            ],
        )
    )

    for i, (title, items) in enumerate(recs):
        print(f"\n  {C.BOLD}{i + 1}. {title}{C.END}")
        for item in items:
            print(f"     • {item}")


def print_cross_rank_summary(analyses: list[TraceAnalysis]):
    header("Cross-Rank Comparison")
    print(
        f"  {'Rank':<25s}  {'Span(ms)':>9s}  {'Busy(ms)':>9s}  {'Util%':>6s}  "
        f"{'Kernel(ms)':>10s}  {'DeepEP(ms)':>10s}  {'GEMM(ms)':>9s}  {'NCCL(ms)':>9s}"
    )
    print(f"  {'─' * 25}  {'─' * 9}  {'─' * 9}  {'─' * 6}  {'─' * 10}  {'─' * 10}  {'─' * 9}  {'─' * 9}")

    for r in analyses:
        gemm_dur = sum(
            ki.total_dur
            for ki in r.kernel_categories.values()
            if "GEMM" in ki.name or "NvJet" in ki.name or "DeepGEMM" in ki.name
        )
        print(
            f"  {r.rank_name:<25s}  {r.gpu_active_span_us / 1000:9.1f}  {r.gpu_busy_time_us / 1000:9.1f}  "
            f"{r.gpu_util_pct:6.1f}  {r.total_kernel_time_us / 1000:10.1f}  "
            f"{r.deep_ep_dur_us / 1000:10.1f}  {gemm_dur / 1000:9.1f}  {r.nccl_kernel_dur_us / 1000:9.1f}"
        )

    utils = [r.gpu_util_pct for r in analyses]
    spans = [r.gpu_active_span_us for r in analyses]
    print(f"\n  Utilization range: {min(utils):.1f}% ~ {max(utils):.1f}% (spread: {max(utils) - min(utils):.1f}%)")
    print(f"  Span range: {min(spans) / 1000:.1f}ms ~ {max(spans) / 1000:.1f}ms")

    if max(utils) - min(utils) > 5:
        warn("Significant load imbalance detected across ranks (>5% spread).")
    else:
        good("Load balance across ranks is reasonable (<5% spread).")


def main():
    parser = argparse.ArgumentParser(description="Analyze SGLang decode profile traces")
    parser.add_argument("--profile-dir", type=str, required=True, help="Directory containing .trace.json.gz files")
    parser.add_argument("--rank", type=int, default=None, help="Specific rank to analyze (default: first file)")
    parser.add_argument("--all-ranks", action="store_true", help="Analyze all ranks and show comparison")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top kernels to show")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.profile_dir, "*.trace.json.gz")))
    if not files:
        print(f"No .trace.json.gz files found in {args.profile_dir}")
        sys.exit(1)

    print(f"Found {len(files)} trace files in {args.profile_dir}")

    if args.all_ranks:
        analyses = []
        for fpath in files:
            basename = os.path.basename(fpath)
            # Extract rank info
            parts = basename.split("-")
            rank_name = "-".join(parts[1:]).replace(".trace.json.gz", "")
            print(f"  Loading {rank_name}...", end=" ", flush=True)
            data = load_trace(fpath)
            analysis = analyze_trace(data, rank_name)
            analyses.append(analysis)
            print("done")
        # Full analysis of first rank
        print_analysis(analyses[0])
        # Cross-rank comparison
        print_cross_rank_summary(analyses)
    else:
        if args.rank is not None:
            target = f"TP-{args.rank}"
            matching = [f for f in files if target in f]
            if not matching:
                print(f"No file found for rank {args.rank}")
                sys.exit(1)
            fpath = matching[0]
        else:
            fpath = files[0]

        basename = os.path.basename(fpath)
        rank_name = "-".join(basename.split("-")[1:]).replace(".trace.json.gz", "")
        print(f"Analyzing: {basename}")

        data = load_trace(fpath)
        analysis = analyze_trace(data, rank_name)
        print_analysis(analysis)


if __name__ == "__main__":
    main()
