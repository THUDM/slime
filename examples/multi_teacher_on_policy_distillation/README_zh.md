# 多教师在线策略蒸馏 (MOPD) 算法说明

本文档详细说明 MOPD 的三种蒸馏模式 (`token_level`、`top_k`、`full_vocab`) 的算法原理、计算流程和使用建议。

---

## 概述

MOPD (Multi-Teacher On-Policy Distillation) 通过 `--mopd-distill-type` 参数支持三种蒸馏模式，核心区别在于 **反向 KL 散度** `D_KL(π_θ ∥ π_d)` 的计算方式：

| | `token_level` | `top_k` | `full_vocab` |
|---|---|---|---|
| **KL 精度** | 单点估计（仅采样 token） | 近似（top-k + 尾部校正） | 精确 |
| **每 token 教师数据** | 1 个标量 (`log π_d(y_t)`) | k×2 个值 (logit + index) | V 个值 (完整 logits) |
| **每 GPU 教师内存\*** | ≈0 | `B × R × k × 2 × 4B / TP` | `B × R × V × 4B / TP` |
| **教师模式** | SGLang 或 Megatron | 仅 Megatron | 仅 Megatron |
| **TP 感知** | 不需要 | 需要（局部索引 + all-reduce） | 需要（词表分片） |
| **梯度** | 仅通过策略损失 | 通过完整学生 softmax | 通过完整学生 softmax |
| **适用场景** | 快速迭代、SGLang 教师 | 大多数生产场景 | 最大精度、内存充裕 |

*\*B=batch, R=平均响应长度, V=词表大小, k=topk-k, TP=张量并行度。三种模式训练时还需 `B × R × V × 4B / TP` 的学生 logits 内存（不可避免）。*

```
内存:      token_level  ◄────────────────────────────────────►  full_vocab
                (≈0)              top_k (O(k) vs O(V))         (O(V))

精度:      token_level  ◄────────────────────────────────────►  full_vocab
           (低: 单token             top_k (高: 捕获            (精确)
            近似)                    ~99%+ KL)
```

---

## 1. Token-Level 模式 (`--mopd-distill-type token_level`，默认)

### 核心思想

仅使用**采样到的 token** 的对数概率差作为反向 KL 散度的点估计。这是最轻量、最省内存的模式，但只在采样的 token 位置提供 KL 信息。

### 公式

对每个采样 token `y_t`，每个教师的反向 KL 优势近似为：

```
reverse_kl_d(y_t) = sg[log π_d(y_t) - log π_θ(y_t)]
```

其中 `sg[·]` 表示 stop-gradient（不向教师回传梯度）。这是 `D_KL(π_θ ∥ π_d)` 的单 token 估计器——它只在采样位置等于完整 KL，对词表其余部分无信息。

**训练损失：**

```
w_t = sg[π_θ(y_t) / μ_θ(y_t)]  截断至 [ε_low, ε_high]      # 重要性采样权重
Â_MOPD,t = (1/D) Σ_d (reverse_kl_d + α · Â_ORM)              # 跨 D 个教师平均
L = -E[1/|y| Σ_t w_t · Â_MOPD,t · log π_θ(y_t)]              # 代理策略损失
```

### 特点

- **所需数据**：仅教师在每个采样 token 上的 log 概率——每个 token 每个教师一个标量。
- **内存**：可忽略（仅存储 `log π_d(y_t)`）。
- **教师模式**：同时支持 SGLang 和 Megatron 教师。
- **精度问题**：会**低估**真实 KL。因为采样 token `y_t` 来自学生策略，倾向落入高 `π_θ` 区域，遗漏了高 `π_d` 但低 `π_θ` 的 token 的贡献。当学生和教师分布差异显著时，偏差更大。

---

## 2. Full-Vocabulary 模式 (`--mopd-distill-type full_vocab`)

### 核心思想

在**完整词表**上计算精确的反向 KL 散度：

```
D_KL(π_θ ∥ π_d) = Σ_y π_θ(y) [log π_θ(y) - log π_d(y)]
```

需要访问学生和教师模型在每个响应位置的完整 logit 向量 `[R, V]`。

**训练损失：**

```
w_t = sg[π_θ(y_t) / μ_θ(y_t)]  截断至 [ε_low, ε_high]       # 重要性采样权重
L_fv_kl = (1/D) Σ_d (1/|y| Σ_t w_t · D_KL(π_θ ∥ π_d))        # 经 IS 校正的 KL 损失
L = L_fv_kl + α · L_pg                                         # 与策略损失组合
```

- 当 `α = 0`：`L = L_fv_kl`（纯蒸馏，无需 ORM）。
- 当 `α > 0`：`L = L_fv_kl + α · L_pg`（蒸馏 + ORM 策略梯度）。

### TP 并行计算

当使用张量并行（TP > 1）时，词表在 TP ranks 间分片，每个 rank 本地持有 `V / tp_size` 个 logits。KL 在数值稳定的 TP 感知方式下计算：

1. **局部 softmax**：`s_max` 和 `s_sum_exp` 跨 TP ranks 做 all-reduce。
2. 使用全局归一化因子在本地计算完整的 softmax 概率/对数概率。
3. `vocab_parallel_reverse_kl` 累加局部 KL 贡献：`KL_local = Σ_{y_local} π_s(y)[log π_s(y) - log π_t(y)]`，由于每个 token 恰好出现在一个 TP rank 上，累加结果等于完整 KL。

### 特点

- **所需数据**：每个样本每个教师的完整 logits `[R_i, V/TP]`（在 rollout 前向传播时计算）。
- **内存**：非常高——每 GPU 每个教师的 rollout 存储为 `B × R × (V/TP) × 4B`（fp32）。示例：V=152K, TP=2, B=4, R=4096 → 4×4096×76K×4 ≈ 4.6 GB；V=248K, TP=8, B=4, R=4096 → 4×4096×31K×4 ≈ 1.9 GB。
- **教师模式**：仅 Megatron 模式（`--mopd-teacher-loads`），因为 SGLang 无法高效返回完整 logit 向量。
- **精度**：精确 KL——蒸馏质量的金标准。

---

## 3. Top-K 模式 (`--mopd-distill-type top_k`)

### 核心思想

full_vocab 的**内存高效近似**。不存储教师完整词表的 logits，仅保留 top-k 的 logits 和索引，加上对剩余词表的解析尾部校正。

### 公式推导

KL 散度分解为两部分：

```
D_KL(π_θ ∥ π_d) ≈ KL_topk + KL_tail
```

#### Top-K 部分 — 在教师的 top-k token 上精确计算

```
KL_topk = Σ_{y ∈ top-k} π_s(y) [log π_s(y) - log π_t(y)]
```

对每个位置，教师提供其 top-k logit 值和对应的 token 索引。学生使用索引 gather 对应位置的概率，并在 top-k 支撑集上精确计算 KL。

**教师 log-prob 的计算**：由于只有 top-k logits，无法得到完整词表的归一化因子 `Z_t`。因此先用 top-k 内部的归一化计算近似 log-prob：

```
log π_t_approx(y) = (logit_t(y) - max_topk) - log(Σ_{y'∈top-k} exp(logit_t(y') - max_topk)),   y ∈ top-k
```

然后通过尾部校正补偿缺失的概率质量。

#### 尾部校正 — 近似非 top-k token 的 KL 贡献

```
KL_tail ≈ π_s_tail · log(π_s_tail / π_t_tail)
```

其中：

- `π_s_tail = 1 - Σ_{y ∈ top-k} π_s(y)` — 学生的精确尾部概率质量（通过跨 TP ranks 的 all-reduce 计算）。
- `π_t_tail ≈ (V - k × tp_size) / V` — 教师的估计尾部概率质量，假设非 top-k token 上服从均匀分布。

**为什么这是一个保守上界？** 均匀分布假设通常会**高估**教师的尾部概率质量（实际上教师 top-k 的 logits 主导了概率分布，真实尾部质量更小），因此近似 KL 会**略微高估**真实 KL。这意味着蒸馏时会略微过度正则化（更偏向教师），这对蒸馏来说是**安全**的。

### 完整损失

```
w_t = sg[π_θ(y_t) / μ_θ(y_t)]  截断至 [ε_low, ε_high]      # 重要性采样权重
L_topk_kl = (1/D) Σ_d (1/|y| Σ_t w_t · KL_topk+d(π_θ ∥ π_d))  # 经 IS 校正的近似 KL
L = L_topk_kl + α · L_pg                                        # 与策略损失组合
```

### TP 并行计算

当使用张量并行（TP > 1）时：

- 教师的 top-k 索引是每个 TP 分片的**局部索引**（因为教师 logits 是词表分片的，`topk` 在每个分片内选择 top-k）。
- 学生直接在这些局部索引位置 gather 概率——**无需跨分片索引转换**。
- 需要的 all-reduce 操作：全局 `s_max`、全局 `s_sum_exp`、全局 `t_max`、全局 `t_sum_exp`、全局 `student_topk_mass`、全局 `local_kl_topk`。

### 特点

- **所需数据**：每个样本每个教师的 top-k logits + 索引 `[R_i, k]`（k 由 `--mopd-topk-k` 控制，默认 1024）。
- **内存**：非常低——每 GPU 每个教师的 rollout 存储为 `B × R × k × 2 × 4B / TP`（fp32 logits + int32/int64 indices）。与 full_vocab 的比值约为 `2k/V`，内存减少约 `1 - 2k/V`。示例：k=1024, V=152K, TP=2, B=4, R=4096 → ≈ 128 MB vs full_vocab 的 ≈ 4.6 GB（**~97% 减少**）；k=1024, V=248K, TP=8, B=4, R=4096 → ≈ 16 MB vs ≈ 1.9 GB。
- **教师模式**：仅 Megatron 模式（`--mopd-teacher-loads`）。
- **精度**：非常接近 full_vocab——top-k token 捕获了绝大部分概率质量，尾部校正提供了对剩余贡献的有界估计。

---

## 选择建议

- **使用 `token_level`**：如果你使用 SGLang 教师，或需要最快的迭代速度和最小内存开销。
- **使用 `top_k`**（Megatron 教师推荐默认值）：精度和效率的最佳平衡。从 `--mopd-topk-k 1024` 开始；如果词表非常大或需要更高精度，可增加到 2048 或 4096。
- **使用 `full_vocab`**：仅当你需要精确 KL 且 GPU 内存充足时。通常仅用于研究验证或小规模实验。

---

## 关键参数

| 参数 | 说明 |
|------|------|
| `--use-mopd` | 启用多教师在线策略蒸馏。与 `--use-opd` 互斥。 |
| `--mopd-teachers` | 教师配置的 JSON 列表，每项含 `name` 和 `domain`（必填）。示例：`'[{"name":"math_t","domain":"math"},{"name":"code_t","domain":"code"}]'` |
| `--mopd-teacher-loads` | Megatron 模式教师的检查点路径，空格分隔。数量须与 `--mopd-teachers` 中的教师数一致。 |
| `--mopd-teacher-ckpt-steps` | 每个教师模型的可选检查点步数。数量须与教师数一致。 |
| `--mopd-alpha` | MOPD 优势与 ORM 优势的组合系数（默认 0.0）。0 为纯蒸馏，>0 为蒸馏+ORM 组合。 |
| `--mopd-eps-low` | IS 权重截断下界（默认 0.2）。低于此值的权重置零。 |
| `--mopd-eps-high` | IS 权重截断上界（默认 5.0）。高于此值的权重置零。 |
| `--mopd-sampling-logprobs-key` | rollout_data 中用于 IS 权重计算的采样 log-probs 键名（默认 `rollout_log_probs`）。 |
| `--mopd-distill-type` | 蒸馏类型：`token_level`（默认）使用采样 token 的 log-prob 差作为反向 KL 近似；`full_vocab` 计算精确的全词表反向 KL 散度；`top_k` 使用教师 top-k logits + 尾部校正计算近似 KL。`full_vocab` 和 `top_k` 均需要 `--mopd-teacher-loads`（Megatron 模式）。详见[算法](#算法)部分。 |
| `--mopd-topk-k` | `top_k` 蒸馏时每个位置保留的 top-k token 数（默认 1024）。k 越大 KL 近似越精确但内存越多。仅在 `--mopd-distill-type=top_k` 时生效。 |

---

## SGLang 模式 vs Megatron 模式

| 模式 | 教师位置 | 何时使用 |
|------|----------|----------|
| `sglang` | 外部 SGLang 服务器（每个教师一个） | 教师架构不同，或太大无法放入训练 GPU 内存 |
| `megatron` | 加载到 Megatron 训练进程 | 教师与策略/参考模型架构相同 |

### SGLang 模式

- 每个教师作为独立 SGLang 服务器运行。
- 教师网址通过 `MOPD_TEACHER_URLS` 环境变量（JSON 字典：`domain -> URL`）或 `--mopd-teachers` 中每个教师配置的 `rm_url` 字段配置。
- 需配置 `--custom-rm-path slime.rollout.mopd.reward_func` 和 `--custom-reward-post-process-path slime.rollout.mopd.post_process_rewards`。
- `--rm-url` 作为未配置单教师 URL 时的回退。

### Megatron 模式

- 教师模型通过 `TensorBackuper` 加载到 CPU 内存，训练时切换到 GPU 进行前向传播。
- 需 `--enable-weights-backuper`（默认开启）用于权重备份/恢复。
- 每个教师必须与策略模型**架构相同**。
- 内存注意：每个教师模型额外占用 CPU 内存用于权重备份。

---

## 核心代码组件

- `slime/rollout/mopd.py`：SGLang 模式 MOPD 的实现。
  - `reward_func`：并发查询所有教师 SGLang 服务器，返回按域名分组的响应。
  - `post_process_rewards`：从响应中提取 token 级教师 log-probs，存入 `sample.mopd_teacher_log_probs`。
- `slime/backends/megatron_utils/actor.py`：Rollout 阶段教师前向传播。
  - `token_level` 模式：调用 `compute_log_prob` 获取教师 log-probs。
  - `full_vocab` / `top_k` 模式：调用 `compute_log_prob(return_logits=True)` 获取完整 logits，`top_k` 额外执行 `topk()` 截取。
- `slime/backends/megatron_utils/loss.py`：训练阶段的损失计算。
  - `apply_mopd_to_advantages`：计算每个教师的反向 KL、IS 权重和聚合的 MOPD 优势（token_level 模式）。
  - `apply_mopd_full_vocab_to_loss`：计算精确全词表 KL 损失（full_vocab 模式）。
  - `apply_mopd_topk_to_loss`：计算 top-k 近似 KL 损失（top_k 模式）。
  - `policy_loss_function`：根据 `mopd_distill_type` 应用相应的损失组合。
- `slime/utils/ppo_utils.py`：
  - `vocab_parallel_reverse_kl`：TP 感知的全词表反向 KL 计算。
  - `vocab_parallel_topk_reverse_kl`：TP 感知的 top-k 近似反向 KL 计算（含尾部校正）。
- `slime/utils/arguments.py`：`--mopd-distill-type` 和 `--mopd-topk-k` 参数定义与验证。

---

## 数据流

### Token-Level 模式

```
Rollout 阶段:
  教师 → compute_log_prob() → log π_d(y_t) (每 token 1 标量) → rollout_data["mopd_teacher_log_probs"]

训练阶段:
  ① apply_mopd_to_advantages():
     reverse_kl_d = log π_d(y_t) - log π_θ(y_t)
     Â_MOPD = avg_d(reverse_kl_d + α·Â_ORM)
     IS权重 w_t = clip(π_θ/μ_θ, [ε_low, ε_high])
  ② 替换优势: pg_loss 用 Â_MOPD 替代 Â_ORM
  ③ 乘以 IS 权重: loss = pg_loss × w_t
```

### Full-Vocab / Top-K 模式

```
Rollout 阶段:
  教师 → compute_log_prob(return_logits=True) → 完整 logits [R_i, V_local]
    ├─ full_vocab: 直接存储所有 logits → rollout_data["mopd_teacher_{domain}_fv_logits"]
    └─ top_k:      topk(k, dim=-1) 截取 → rollout_data["mopd_teacher_{domain}_topk_logits"]
                                              rollout_data["mopd_teacher_{domain}_topk_indices"]

训练阶段:
  ① 学生前向传播 → get_logits(apply_temperature=False) → student_logits [R_i, V_local]
  ② full_vocab: vocab_parallel_reverse_kl(student_logits, teacher_logits) → D_KL 精确值
     top_k:      vocab_parallel_topk_reverse_kl(student_logits, teacher_topk_logits, teacher_topk_indices)
                 → KL_topk + KL_tail → 近似 D_KL
  ③ IS 权重: w_t = clip(π_θ/μ_θ, [ε_low, ε_high])
  ④ 加权 KL: L_kl = (1/D) Σ_d (1/|y| Σ_t w_t · KL_d)
  ⑤ 损失组合: L = L_kl + α · L_pg  (α=0 时纯蒸馏)
```

### 内存对比（单个教师）

每个 GPU 上每种模式每样本的内存公式（fp32）：

| | Rollout 教师存储 (per GPU) | top_k / full_vocab 比值 |
|---|---|---|
| `token_level` | `B × R × 4B` (1 scalar/token) | — |
| `top_k` | `B × R × k × 2 × 4B / TP` | `2k / V` |
| `full_vocab` | `B × R × V × 4B / TP` | 1 |

训练时三种模式均需额外 `B × R × V × 4B / TP` 的学生 logits 内存（`top_k` 和 `full_vocab` 模式需要完整 logits 计算 softmax）。

参数说明：B=batch, R=平均响应长度, V=词表大小, k=`--mopd-topk-k`, TP=张量并行度。

**参考数值：**

| 配置 | `token_level` | `top_k` (k=1024) | `full_vocab` |
|------|---------------|-------------------|--------------|
| V=152K, TP=2, B=4, R=4096 | ~64 KB | ~128 MB | ~4.6 GB |
| V=248K, TP=8, B=4, R=4096 | ~64 KB | ~16 MB | ~1.9 GB |
| V=152K, TP=2, B=8, R=2048 | ~32 KB | ~64 MB | ~2.3 GB |

注意：训练时学生 logits 的内存开销在三种模式下相同（都需要完整 logits 来计算 KL），差异主要在教师的 rollout 存储。

---

## FAQ

1. **MOPD 可以和 OPD 同时使用吗？**
   不可以。`--use-mopd` 和 `--use-opd` 互斥。需要多教师时使用 MOPD。

2. **所有教师需要架构相同吗？**
   - Megatron 模式：是的，所有教师必须与策略模型架构相同。
   - SGLang 模式：不需要，每个教师可以是不同架构，因为它们运行在独立的服务器上。

3. **Megatron 模式下 MOPD 需要多少额外内存？**
   每个教师模型需要 CPU 内存用于权重备份（通过 `TensorBackuper`）。教师权重仅在训练前向传播时临时加载到 GPU，然后恢复到 CPU。按 `N × model_size` 规划额外 CPU 内存，N 为教师数量。此外 `full_vocab` 模式还需要大量 GPU 内存存储教师 logits（见上文内存对比表）。

4. **SGLang 模式下教师服务器故障怎么办？**
   `reward_func` 会记录警告并跳过该教师。训练会继续使用剩余教师，但优势会有偏。请密切监控教师服务器健康状态。

5. **为什么 `--group-rm` 不支持 MOPD？**
   MOPD 的 `reward_func` 返回按域名的字典（非标量奖励），与批量 `group_rm` 奖励路径不兼容。使用默认的逐样本奖励路径（不加 `--group-rm`）。

6. **`token_level`、`top_k` 和 `full_vocab` 三种蒸馏类型有什么区别？**
   - `token_level`（默认）：使用采样 token 的 log-prob 差近似反向 KL。高效，支持 SGLang 和 Megatron，但仅在采样位置捕获 KL 信息。当学生和教师分布差异大时会低估真实 KL。
   - `top_k`：使用教师 top-k logits + 解析尾部校正计算近似反向 KL。内存高效（比 `full_vocab` 少 ~97%），仅 Megatron 模式。k=1024+ 时精度非常接近 `full_vocab`。
   - `full_vocab`：计算精确的全词表反向 KL 散度。最精确但内存密集（存储完整 `[R, V]` logits）。仅 Megatron 模式。
   - 详见[算法](#算法)部分和[对比表](#概述)。

7. **什么时候用 `top_k`，什么时候用 `full_vocab`？**
   大多数生产场景用 `top_k`——以 `full_vocab` 约 3% 的内存捕获 >99% 的 KL 信号。只有在需要精确 KL 做研究验证、或 GPU 内存十分充裕时才用 `full_vocab`。从 `--mopd-topk-k 1024` 开始；如果词表非常大（如 V > 200K）或发现尾部校正过于激进，可增大到 2048 或 4096。

8. **`top_k` 的尾部校正原理是什么？**
   top-k 分解将 KL 分为 `KL_topk`（在教师 top-k token 上精确计算）和 `KL_tail`（近似剩余 V−k 个 token 的贡献）。尾部假设非 top-k token 上教师服从均匀分布：`π_t_tail ≈ (V − k·tp_size) / V`。这是一个保守上界——真实的教师尾部质量通常更小（教师 top-k 占据了主要概率），因此近似 KL 会略微高估真实 KL，意味着略微过度正则化（更偏向教师），对蒸馏是安全的。

9. **三种蒸馏模式的内存用量？**
   每种模式的 GPU 内存与词表大小 V、张量并行度 TP、批量 B、响应长度 R 成正比：
   - `token_level`：可忽略（`B × R × 4B`）。
   - `top_k`（k=1024）：`B × R × k × 2 × 4B / TP`。比例约为 full_vocab 的 `2k/V`（约 1~2%）。
   - `full_vocab`：`B × R × V × 4B / TP`。
   具体数值因模型而异，例如 V=152K/TP=2/B=4/R=4096 时 top_k ≈ 128 MB、full_vocab ≈ 4.6 GB；而 V=248K/TP=8/B=4/R=4096 时 top_k ≈ 16 MB、full_vocab ≈ 1.9 GB。
   如果 `full_vocab` OOM，切换到 `top_k` 或减小 `--rollout-batch-size` / `--rollout-max-response-len`。

10. **`top_k` 模式的 `k` 值怎么选？**
    - `k=1024`（默认）：适用于大多数场景，平衡精度和内存。
    - `k=2048`：词表较大（V > 200K）时推荐，进一步减少尾部校正的近似误差。
    - `k=4096`：需要更高精度时使用，内存仍远小于 `full_vocab`。
    - 经验法则：k/V > 0.5% 即可捕获 >99% 的 KL 信号，因为教师分布通常高度集中。