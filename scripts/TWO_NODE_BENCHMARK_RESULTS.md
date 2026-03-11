# 双机 Two-Node Benchmark 结果（Ray vs Mooncake）

**运行时间**: 2026-03-11  
**环境**: Put 192.168.22.70, Get 192.168.22.72, RDMA, warmup=24, discard-first=5, trim=15%, isolate-backends

---

## 1. 测试方法

- **脚本**: `scripts/benchmark_ray_vs_mooncake_two_node.py`
- **拓扑**: DataGenerator 在 70（Put），DataConsumer 在 72（Get），handle 经 Ray 传递
- **时序**: Put = dict→handle 全流程；Get = handle→dict 全流程；E2E = Put + Get
- **稳定化**: `--isolate-backends` 各 backend 独立进程；warmup=24、discard-first=5、trim=15% 降低 Ray 方差

---

## 2. 最新结果（稳定版）

### 2.1 100 MB

| Backend | Put (ms) | Get (ms) | E2E (ms) |
|---------|----------|----------|----------|
| **ray** | 22.65 ± 3.06 | 41.53 ± 0.31 | 64.39 ± 2.89 |
| **mooncake** | 34.15 ± 0.36 | **14.71 ± 0.38** | **48.89 ± 0.26** |

- 实际数据量: ~99 MB，30 轮
- Mooncake E2E 约快 24%，Get 约快 2.8×

### 2.2 200 MB

| Backend | Put (ms) | Get (ms) | E2E (ms) |
|---------|----------|----------|----------|
| **ray** | 42.11 ± 3.94 | 82.83 ± 2.62 | 126.13 ± 4.46 |
| **mooncake** | 75.93 ± 3.93 | **29.50 ± 0.53** | **105.51 ± 3.15** |

- 实际数据量: ~203 MB，30 轮，`--mooncake-segment-size-gb 8`
- Mooncake E2E 约快 16%

### 2.3 500 MB

| Backend | Put (ms) | Get (ms) | E2E (ms) |
|---------|----------|----------|----------|
| **ray** | 104.18 ± 9.42 | 208.60 ± 6.15 | 315.39 ± 10.89 |
| **mooncake** | 170.58 ± 1.12 | **83.82 ± 2.31** | **254.92 ± 2.13** |

- 实际数据量: ~507 MB，30 轮，`--mooncake-segment-size-gb 16`
- Mooncake E2E 约快 19%，Get 约快 2.5×

### 2.4 1000 MB

| Backend | Put (ms) | Get (ms) | E2E (ms) |
|---------|----------|----------|----------|
| **ray** | 206.07 ± 23.12 | 419.21 ± 10.52 | 627.89 ± 21.03 |
| **mooncake** | 374.37 ± 1.69 | **179.50 ± 1.20** | **553.78 ± 3.91** |

- 实际数据量: ~1000 MB，8 轮，`--mooncake-segment-size-gb 16`
- Mooncake E2E 约快 12%
- **注意**: segment 需 ≥ (num_rounds + discard_first) × data_size（先 put 完再 get）

---

## 3. 复现命令

```bash
export PUT_NODE=192.168.22.70 GET_NODE=192.168.22.72
export MOONCAKE_MASTER=192.168.22.70:50051 MOONCAKE_PROTOCOL=rdma
export SLIME_UNSAFE_PICKLE=1

# 100 MB
python scripts/benchmark_ray_vs_mooncake_two_node.py \
  --put-node $PUT_NODE --get-node $GET_NODE \
  --data-size-mb 100 --num-rounds 30 --warm-up-rounds 24 \
  --backends ray mooncake

# 200 MB
python scripts/benchmark_ray_vs_mooncake_two_node.py \
  --put-node $PUT_NODE --get-node $GET_NODE \
  --data-size-mb 200 --num-rounds 30 --warm-up-rounds 24 \
  --backends ray mooncake --mooncake-segment-size-gb 8

# 500 MB
python scripts/benchmark_ray_vs_mooncake_two_node.py \
  --put-node $PUT_NODE --get-node $GET_NODE \
  --data-size-mb 500 --num-rounds 30 --warm-up-rounds 24 \
  --backends ray mooncake --mooncake-segment-size-gb 16

# 1000 MB（segment ≥ 16GB，因 8+5 rounds × 1GB）
python scripts/benchmark_ray_vs_mooncake_two_node.py \
  --put-node $PUT_NODE --get-node $GET_NODE \
  --data-size-mb 1000 --num-rounds 8 --warm-up-rounds 8 --discard-first 3 \
  --backends ray mooncake --mooncake-segment-size-gb 16
```

---

## 4. 小结

- **100–1000 MB**: Mooncake 双机 E2E 均优于 Ray（约 12%–24%）
- **Ray 方差**: 通过 isolate-backends、warmup=24、discard-first=5、trim=15% 已显著降低
- **Segment 配置**: segment_size ≥ (num_rounds + discard_first) × data_size（benchmark 先 put 完所有轮次再 get）
