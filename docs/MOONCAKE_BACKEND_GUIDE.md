# Mooncake Backend Guide

This document covers configuration, data flow, storage modes, and troubleshooting for using Mooncake as Slime's rollout transfer backend.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Configuration and Environment Variables](#2-configuration-and-environment-variables)
3. [Usage and Best Practices](#3-usage-and-best-practices)
4. [Data Flow and Single-Key Storage](#4-data-flow-and-single-key-storage)
5. [Memory Optimization](#5-memory-optimization)
6. [Benchmark Results](#6-benchmark-results)
7. [Troubleshooting](#7-troubleshooting)
8. [References](#8-references)

---

## 1. Overview

### 1.1 What It Is

**Mooncake Hybrid** is Slime's rollout data transfer backend that uses Mooncake distributed store instead of Ray Object Store for moving data from rollout nodes to training nodes. Core logic lives in `slime/utils/rollout_hybrid_transfer.py`.

| Backend | Description | Best For |
|---------|-------------|----------|
| **ray** (default) | Ray Object Store | Single-node, colocated training and inference |
| **mooncake** | Mooncake Hybrid | Multi-node, disaggregated setups with RDMA |

### 1.2 Storage Modes

| Mode | Keys | Environment Variable | Description |
|------|------|----------------------|-------------|
| **Legacy** | 1 + N | `SLIME_USE_LEGACY_TRANSFER=1` | One meta key, one key per tensor |
| **Single-Key** | 1 | Default (non-Legacy) | All data in one contiguous value, one key |
| **Two-Key** | 2 | `SLIME_META_TENSOR_SPLIT_KEYS=1` | Meta in one key, tensors concatenated in another |

---

## 2. Configuration and Environment Variables

### 2.1 Required

| Variable | Description | Example |
|----------|-------------|---------|
| `MOONCAKE_MASTER` | Master address (host:port) | `192.168.22.70:50051` |

### 2.2 Optional (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `MOONCAKE_PROTOCOL` | `tcp` | `tcp` or `rdma`; use `rdma` for multi-node |
| `MOONCAKE_DEVICE` | `""` | RDMA device name (e.g. `erdma_0`) |
| `MOONCAKE_LOCAL_HOSTNAME` | Auto-detected | Local hostname/IP |
| `MOONCAKE_TE_META_DATA_SERVER` | `P2PHANDSHAKE` | Transfer engine metadata server |
| `MOONCAKE_MOUNT_SEGMENT_SIZE` | 4 GiB | Segment size to mount |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | 2 GiB | Local buffer size |

### 2.3 Slime-Specific

| Variable | Default | Description |
|----------|---------|-------------|
| `SLIME_USE_LEGACY_TRANSFER` | `0` | `1` to use Legacy multi-key mode |
| `SLIME_RING_BUFFER_SIZE_MB` | `2048` | Ring buffer size per slot (MB) for Get |
| `SLIME_UNSAFE_PICKLE` | `0` | `1` to disable restricted unpickler (trusted env only) |
| `MC_STORE_MEMCPY` | `0` | **Must be 0 for cross-node**; can be 1 for same-node |
| `SLIME_PACK_DIRECT_TO_BUFFER` | `0` | `1` to enable Direct Pack optimization |
| `SLIME_REGISTER_PICKLE_BUFFERS` | `0` | `1` to try zero-copy Put via register |
| `SLIME_META_TENSOR_SPLIT_KEYS` | `0` | `1` to enable Two-Key mode |
| `SLIME_USE_NUMPY_META` | `0` | `1` to use numpy-format meta |

### 2.4 Example Configurations

**Single-node (TCP):**
```bash
export MOONCAKE_MASTER=127.0.0.1:50051
export MOONCAKE_PROTOCOL=tcp
export MC_STORE_MEMCPY=1
```

**Two-node (RDMA):**
```bash
export MOONCAKE_MASTER=192.168.22.70:50051
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=erdma_0
export MC_STORE_MEMCPY=0
```

---

## 3. Usage and Best Practices

### 3.1 Command Line

```bash
python -m slime.train --transfer-backend mooncake ...
```

### 3.2 Prerequisites

1. Mooncake installation and cluster (master + clients on each node)
2. For RDMA multi-node: InfiniBand or RoCE configured correctly
3. For large payloads (500MB+): consider increasing `MOONCAKE_MOUNT_SEGMENT_SIZE`

### 3.3 Best Practices

- Use RDMA for multi-node: `MOONCAKE_PROTOCOL=rdma`
- `MC_STORE_MEMCPY=1` only when Put and Get are on the same node
- Segment size: `segment_size ≥ num_rounds × data_size` when batching (benchmark puts all rounds before get)
- `SLIME_UNSAFE_PICKLE=1` only in trusted environments

---

## 4. Data Flow and Single-Key Storage

### 4.1 Data Flow Overview

```
rollout dict (scattered memory)
    → Prepare (pack ragged + pickle OOB)
    → meta_bytes + tensors
    → Put (write to _meta_buf/_header_buf, prepare tensors)
    → batch_put_from_multi_buffers (Mooncake RDMA reads ptrs)
    → Mooncake store (concatenates into value by layout)
    → Get: batch_get_into (read entire value into buf)
    → Parse layout (meta + tensor views)
    → pickle.loads(meta_bytes, buffers=[...])
    → rollout dict
```

### 4.2 Single-Key Value Layout

The value is a contiguous byte stream: `Hdr0(40B) | Meta(pad0) | Hdr1(40B) | T0(pad1) | Hdr2(40B) | T1(pad2) | ...`

- **Hdr**: 40B struct with `dtype_id | ndim | shape0..3`
- **Meta**: Pickle meta (OOB references tensors)
- **T0, T1...**: Raw tensor bytes, 64B-aligned

### 4.3 Put: Scattered Memory → Contiguous Value

Put-side data lives in separate buffers: `_header_buf`, `_meta_buf`, and per-tensor buffers. `batch_put_from_multi_buffers(key, ptrs, sizes)` receives interleaved ptr/size pairs; Mooncake RDMA-reads them in order and concatenates into the stored value.

```
ptrs = [hdr0_ptr, meta_ptr, hdr1_ptr, t0_ptr, hdr2_ptr, t1_ptr, ...]
sizes = [40, pad0, 40, pad1, 40, pad2, ...]
```

### 4.4 Get: Single Read + Parse

1. `batch_get_into([key], [buf_ptr], [total])` reads the entire value
2. `meta_bytes = buf[40:40+meta_size]`
3. Loop over `padded_sizes` to extract tensor views
4. `pickle.loads(meta_bytes, buffers=[...])` reconstructs dict with zero-copy

---

## 5. Memory Optimization

### 5.1 register_buffer / alloc_from_mem_pool

- `register_buffer(ptr, size)`: Registers user memory with Mooncake for RDMA
- `alloc_from_mem_pool(size)`: Allocates from Mooncake memory pool; often returns 0 under standard setup

### 5.2 SLIME_REGISTER_PICKLE_BUFFERS

When enabled, tries `register_buffer(pickle_tensor_ptr)`; on success, Put is zero-copy; on failure, copies to _put_buffers.

### 5.3 Direct Pack (SLIME_PACK_DIRECT_TO_BUFFER=1)

- Packs directly into buffer without intermediate allocations
- Tries `alloc_from_mem_pool` first, falls back to `torch.empty`+`register_buffer`
- Returns `registered_range`; Put skips copy when tensors fall within that range

---

## 6. Benchmark Results

### 6.1 Two-Node RDMA (Put node 70 → Get node 72)

**运行时间**: 2026-03-11  
**环境**: Put 192.168.22.70, Get 192.168.22.72, RDMA, warmup=24, discard-first=5, trim=15%, isolate-backends

| Data Size | Backend | Put (ms) | Get (ms) | E2E (ms) |
|-----------|---------|----------|----------|----------|
| **100 MB** (~99 MB) | Ray | 22.65 ± 3.06 | 41.53 ± 0.31 | 64.39 ± 2.89 |
| | **Mooncake** | 34.15 ± 0.36 | **14.71 ± 0.38** | **48.89 ± 0.26** |
| **200 MB** (~203 MB) | Ray | 42.11 ± 3.94 | 82.83 ± 2.62 | 126.13 ± 4.46 |
| | **Mooncake** | 75.93 ± 3.93 | **29.50 ± 0.53** | **105.51 ± 3.15** |
| **500 MB** (~507 MB) | Ray | 104.18 ± 9.42 | 208.60 ± 6.15 | 315.39 ± 10.89 |
| | **Mooncake** | 170.58 ± 1.12 | **83.82 ± 2.31** | **254.92 ± 2.13** |
| **1000 MB** (~1000 MB) | Ray | 206.07 ± 23.12 | 419.21 ± 10.52 | 627.89 ± 21.03 |
| | **Mooncake** | 374.37 ± 1.69 | **179.50 ± 1.20** | **553.78 ± 3.91** |

**结论**:
- 100 MB: Mooncake E2E **49 ms** vs Ray **64 ms**，约快 **24%**；Get 端 Mooncake **14.7 ms** vs Ray **41.5 ms**，约快 **2.8×**
- 200 MB: Mooncake E2E **106 ms** vs Ray **126 ms**，约快 **16%**
- 500 MB: Mooncake E2E **255 ms** vs Ray **315 ms**，约快 **19%**；需 `--mooncake-segment-size-gb 16`
- 1000 MB: Mooncake E2E **554 ms** vs Ray **628 ms**，约快 **12%**；需 `--mooncake-segment-size-gb 16`（segment ≥ num_rounds × data_size）

### 6.2 Running Benchmarks

```bash
export MOONCAKE_MASTER=192.168.22.70:50051 MOONCAKE_PROTOCOL=rdma SLIME_UNSAFE_PICKLE=1
python scripts/benchmark_ray_vs_mooncake_two_node.py \
  --put-node 192.168.22.70 --get-node 192.168.22.72 \
  --data-size-mb 100 --num-rounds 30 --warm-up-rounds 24 \
  --discard-first 5 --trim-fraction 0.15 \
  --backends ray mooncake
```

- 默认 `--isolate-backends`：Ray 与 Mooncake 分别在独立进程中运行，避免内存/GC 干扰，使 Ray 方差更稳定
- 运行前确保 Mooncake 处于干净状态（无冲突进程）；必要时手动 `pkill -f mooncake_master` 并重启 master
- `--mooncake-segment-size-gb N`: segment 需 ≥ num_rounds × data_size（benchmark 先 put 完再 get）
- 详细结果见 `scripts/TWO_NODE_BENCHMARK_RESULTS.md`

---

## 7. Troubleshooting

### 7.1 Mooncake Slower Than Ray

- Confirm Put/Get nodes: should be cross-node (PUT≠GET)
- Use `benchmark_ray_vs_mooncake_two_node.py`; avoid cross_node (SSH) script
- Run Mooncake alone to rule out Ray interference
- Ensure `MC_STORE_MEMCPY=0` for cross-node

### 7.2 "Overlapped memory region" Warning

With Direct Pack, this indicates overlapping memory regions and may trigger a slower path. Try disabling `SLIME_PACK_DIRECT_TO_BUFFER`.

### 7.3 alloc_from_mem_pool Returns 0

Common under standard setup; falls back to `torch.empty`+`register_buffer`. Functionality is unaffected.

---

## 8. References

- [Mooncake Documentation](https://kvcache-ai.github.io/Mooncake/)
- [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- Slime code: `slime/utils/data_transfer.py`, `slime/utils/rollout_hybrid_transfer.py`
