# Multi-Teacher On-Policy Distillation (MOPD) Example

This example shows how to run **multi-teacher on-policy distillation (MOPD)** using slime. MOPD extends OPD to support multiple domain-specific teachers, enabling a single student model to simultaneously learn from several experts (e.g., a math teacher and a code teacher) while using importance sampling (IS) for stable off-policy training.

## Key Features

- **Multi-teacher distillation**: Aggregate knowledge from multiple domain-specific teachers into a single student, with per-teacher reverse KL advantages averaged across domains.
- **Importance sampling (IS) correction**: Clipped IS weights `w_t = sg[π_θ/μ_θ]` ensure stable training when the student diverges from the sampling policy.
- **ORM combination**: Optional coefficient `α` blends reverse KL advantages with standard ORM advantages: `Â_MOPD,t = sg[log(π_domain/π_θ)] + α · Â_ORM`.
- **Two teacher modes** (same as OPD):
  - **sglang**: Teachers run on external SGLang servers, teacher log-probs are obtained during rollout.
  - **megatron**: Teachers are loaded directly into Megatron via `--mopd-teacher-loads`, teacher log-probs are computed during the training forward pass.

## Algorithm

MOPD supports three distillation types, controlled by `--mopd-distill-type`:

### Token-Level Mode (`--mopd-distill-type token_level`, default)

Uses the sampled token's log-prob difference as a **point estimate** of the reverse KL divergence. This is the cheapest and most memory-efficient mode, but only captures KL information at the positions of the actually sampled tokens.

**Core formula:**

For each sampled token `y_t`, the per-teacher reverse KL advantage is approximated as:

```
reverse_kl_d(y_t) = sg[log π_d(y_t) - log π_θ(y_t)]
```

where `sg[·]` denotes stop-gradient (no gradient flows to the teacher). This is a single-token estimator of `D_KL(π_θ ∥ π_d)`: it equals the full KL only at the sampled position and provides no information about the rest of the vocabulary.

**Training loss:**

```
w_t = sg[π_θ(y_t) / μ_θ(y_t)]  clipped to [ε_low, ε_high]   # IS weight
Â_MOPD,t = (1/D) Σ_d (reverse_kl_d + α · Â_ORM)              # avg across D teachers
L = -E[1/|y| Σ_t w_t · Â_MOPD,t · log π_θ(y_t)]              # proxy policy loss
```

**Characteristics:**
- **Data needed**: Only teacher log-probs at sampled tokens — a scalar per token per teacher.
- **Memory**: Negligible (storing `log π_d(y_t)` only).
- **Teacher modes**: Works with both SGLang and Megatron teachers.
- **Accuracy**: Underestimates the true KL because it only evaluates at sampled positions. When the student and teacher distributions differ significantly, the sampled token `y_t` (from the student's policy) tends to be in high-`π_θ` regions, missing contributions from high-`π_d` but low-`π_θ` tokens.

### Full-Vocabulary Mode (`--mopd-distill-type full_vocab`)

Computes the **exact** reverse KL divergence over the entire vocabulary:

```
D_KL(π_θ ∥ π_d) = Σ_y π_θ(y) [log π_θ(y) - log π_d(y)]
```

This requires accessing the full logit vectors `[R, V]` from both the student and teacher models at every response position.

**Training loss:**

```
w_t = sg[π_θ(y_t) / μ_θ(y_t)]  clipped to [ε_low, ε_high]   # IS weight
L_fv_kl = (1/D) Σ_d (1/|y| Σ_t w_t · D_KL(π_θ ∥ π_d))        # IS-corrected KL loss
L = L_fv_kl + α · L_pg                                         # combined with PG loss
```

- When `α = 0`: `L = L_fv_kl` (pure distillation, no ORM needed).
- When `α > 0`: `L = L_fv_kl + α · L_pg` (distillation + ORM policy gradient).

**TP-aware computation:** When using tensor parallelism (TP > 1), the vocabulary is sharded across TP ranks. Each rank holds `V / tp_size` logits locally. The KL is computed in a numerically stable, TP-aware manner:
1. Local softmax: `s_max` and `s_sum_exp` are all-reduced across TP ranks.
2. Full-softmax prob/log-prob are computed locally using the global normalizer.
3. `vocab_parallel_reverse_kl` sums the local KL contributions: `KL_local = Σ_{y_local} π_s(y)[log π_s(y) - log π_t(y)]`, which yields the full KL because each token appears on exactly one TP rank.

**Characteristics:**
- **Data needed**: Full teacher logits `[R_i, V/TP]` per sample per teacher (computed during rollout forward pass).
- **Memory**: Very high — per-GPU memory per teacher is `B × R × (V/TP) × 4B` (fp32). Example: V=152K, TP=2, B=4, R=4096 → ~4.6 GB; V=248K, TP=8, B=4, R=4096 → ~1.9 GB.
- **Teacher modes**: Only Megatron mode (`--mopd-teacher-loads`), because SGLang cannot efficiently return full logit vectors.
- **Accuracy**: Exact KL — the gold standard for distillation quality.

### Top-K Mode (`--mopd-distill-type top_k`)

A **memory-efficient approximation** of the full-vocab KL. Instead of storing the entire vocabulary of teacher logits, only the top-k logits and their indices are kept, plus an analytical tail correction to account for the remaining vocabulary.

**Core formula:**

The KL divergence is decomposed into two parts:

```
D_KL(π_θ ∥ π_d) ≈ KL_topk + KL_tail
```

**Top-K part** — computed exactly over the teacher's top-k tokens:

```
KL_topk = Σ_{y ∈ top-k} π_s(y) [log π_s(y) - log π_t(y)]
```

For each position, the teacher provides its top-k logit values and the corresponding token indices. The student's probabilities at those positions are gathered using the indices, and the exact KL over the top-k support is computed.

**Tail correction** — approximates the KL contribution from non-top-k tokens:

```
KL_tail ≈ π_s_tail · log(π_s_tail / π_t_tail)
```

where:
- `π_s_tail = 1 - Σ_{y ∈ top-k} π_s(y)` — the student's exact tail mass (computed via all-reduce across TP ranks).
- Teacher tail mass estimation differs by mode:
  - **Megatron mode**: `π_t_tail ≈ (V - V_eff) / V` — uniform distribution assumption over non-top-k tokens, where `V_eff = k × tp_size` is the total number of valid top-k entries. Since Megatron mode typically uses larger k (e.g., 1024+), the top-k entries capture most of the probability mass and this approximation is reasonable.
  - **SGLang mode**: `π_t_tail = 1 - Σ_{y ∈ top-k} exp(log_prob_t(y))` — **exact** computation from the teacher's full-vocabulary log-probs returned by SGLang. Since SGLang returns `log(π_t(y))` (already softmax-normalized over the full vocabulary), summing `exp(log_prob)` gives the true probability mass in the top-k partition, and the tail is simply `1 - mass_topk`. This is a key advantage of SGLang mode: the tail mass is computed exactly, not estimated.

**Important**: In SGLang mode, the teacher returns log-probs (not raw logits). The `vocab_parallel_topk_reverse_kl` function detects this via the `is_log_probs=True` flag and skips the log_softmax step, using the log-probs directly and computing tail mass from their exp-sum. This avoids the double-softmax bug and the inaccurate uniform tail estimate that would otherwise make KL values unrealistically large (e.g., ~7.7 nats with k=128 and V=152K).

**Full loss:**

```
w_t = sg[π_θ(y_t) / μ_θ(y_t)]  clipped to [ε_low, ε_high]    # IS weight
L_topk_kl = (1/D) Σ_d (1/|y| Σ_t w_t · KL_topk+d(π_θ ∥ π_d))  # IS-corrected approx KL
L = L_topk_kl + α · L_pg                                         # combined with PG loss
```

**TP-aware computation:** When using tensor parallelism (TP > 1), the handling differs by teacher mode:

- **Megatron mode**: The teacher's top-k indices are **local** per TP shard (since the teacher logits are vocab-sharded, `topk` selects the top-k within each shard). The student gathers its probabilities at these local indices directly — no cross-shard index translation needed.
- **SGLang mode**: The SGLang server returns **global** token IDs. During the actor's data preparation step, each TP rank converts global indices to local indices within its vocab range `[vocab_offset, vocab_offset + vocab_local_size)`. Entries whose global token ID falls outside the shard's range are replaced with `-inf` logit and `0` index (padding). The `valid_topk_mask` (computed as `~torch.isinf(teacher_topk_logits)`) automatically identifies valid vs. padding entries.

For both modes, all-reduce operations are needed for: global `s_max`, global `s_sum_exp`, global `t_max`, global `t_sum_exp`, global `student_topk_mass`, and global `local_kl_topk`.

**Characteristics:**
- **Data needed**: Teacher top-k logits + indices `[R_i, k]` per sample per teacher (k controlled by `--mopd-topk-k`, default 1024).
- **Memory**: Very low — per-GPU memory per teacher is `B × R × k × 2 × 4B / TP` (fp32 logits + int64 indices). The ratio vs full_vocab is approximately `2k/V`, saving `1 - 2k/V`. Example: k=1024, V=152K, TP=2, B=4, R=4096 → ~128 MB vs full_vocab's ~4.6 GB (~97% reduction); k=1024, V=248K, TP=8, B=4, R=4096 → ~16 MB vs ~1.9 GB.
- **Teacher modes**: SGLang or Megatron. SGLang mode uses the `top_logprobs_num` parameter to request top-k logprobs from the remote server; Megatron mode computes top-k during the teacher forward pass.
- **Accuracy**: Very close to full_vocab — the top-k tokens capture the vast majority of the probability mass. The tail correction provides a bounded estimate of the remaining contribution.

### Comparison of Distillation Types

| | `token_level` | `top_k` | `full_vocab` |
|---|---|---|---|
| **KL accuracy** | Point estimate (sampled token only) | Approximate (top-k + tail correction) | Exact |
| **Teacher data per token** | 1 scalar (`log π_d(y_t)`) | k×2 values (logit + index) | V values (full logits) |
| **Teacher mem per GPU\*** | ≈0 | `B × R × k × 2 × 4B / TP` | `B × R × V × 4B / TP` |
| **Teacher mode** | SGLang or Megatron | SGLang or Megatron | Megatron only |
| **TP aware** | Not needed | Yes (local indices, all-reduce) | Yes (vocab-sharded) |
| **Gradient** | Through policy loss only | Through full student softmax | Through full student softmax |
| **When to use** | Quick iteration, SGLang teachers | Best balance of accuracy & efficiency | Max accuracy, sufficient memory |

*\*B=batch, R=avg response length, V=vocab size, k=topk-k, TP=tensor parallelism degree. All three modes also require `B × R × V × 4B / TP` student logits memory during training (unavoidable for `top_k` and `full_vocab`).*

The following diagram illustrates the trade-off:

```
Memory:    token_level  ◄────────────────────────────────────►  full_vocab
                (≈0)              top_k (O(k) vs O(V))         (O(V))

Accuracy:  token_level  ◄────────────────────────────────────►  full_vocab
           (low: 1-token          top_k (high: ~99%+          (exact)
            approximation)         of KL captured)
```

### How to Choose

- **Use `token_level`** if you need the fastest iteration with minimal memory overhead, or if your teacher only supports sampled-token logprobs (no top-k API).
- **Use `top_k`** (recommended default) for the best balance of accuracy and efficiency. Works with both SGLang and Megatron teachers. Start with `--mopd-topk-k 1024`; increase to 2048 or 4096 if the vocabulary is very large or you want more precision.
- **Use `full_vocab`** only when you need the exact KL and have sufficient GPU memory. Only available with Megatron teachers. Typically only needed for research validation or very small-scale experiments.

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-mopd` | Enable multi-teacher on-policy distillation. Mutually exclusive with `--use-opd`. |
| `--mopd-teachers` | JSON list of teacher configs, each with `name` and `domain` (required). Example: `'[{"name":"math_t","domain":"math"},{"name":"code_t","domain":"code"}]'` |
| `--mopd-teacher-loads` | Space-separated checkpoint paths for megatron-mode teachers. Must match the number of teachers in `--mopd-teachers`. |
| `--mopd-teacher-ckpt-steps` | Optional checkpoint steps for each teacher model. Must match the number of teachers. |
| `--mopd-alpha` | Coefficient for combining MOPD advantage with ORM advantage (default: 0.0). Set to 0 for pure distillation, >0 for ORM combination. |
| `--mopd-eps-low` | IS weight lower bound for clipping (default: 0.2). Weights below this are zeroed. |
| `--mopd-eps-high` | IS weight upper bound for clipping (default: 5.0). Weights above this are zeroed. |
| `--mopd-sampling-logprobs-key` | Key in rollout_data for sampling log-probs used in IS weight computation (default: `rollout_log_probs`). |
| `--mopd-distill-type` | Distillation type: `token_level` (default) uses sampled token log-prob difference as a reverse KL approximation; `full_vocab` computes the exact full-vocabulary reverse KL divergence; `top_k` computes approximate KL using teacher top-k logits + tail correction. `full_vocab` requires `--mopd-teacher-loads` (Megatron mode); `top_k` and `token_level` work with both SGLang and Megatron teachers. See [Algorithm](#algorithm) for details. |
| `--mopd-topk-k` | Number of top-k tokens to keep per position for `top_k` distillation (default: 1024). Higher k gives more accurate KL approximation at the cost of more memory. Only used when `--mopd-distill-type=top_k`. |

## SGLang vs Megatron Mode

| Mode | Teacher Location | When to use |
|------|------------------|-------------|
| `sglang` | External SGLang servers (one per teacher) | Teachers have different architecture or are too large for training GPU memory |
| `megatron` | Loaded into Megatron training process | Teachers have the same architecture as the policy/ref model |

### SGLang Mode

- Each teacher runs as an independent SGLang server.
- Teacher URLs are configured via the `MOPD_TEACHER_URLS` environment variable (JSON dict: `domain -> URL`) or via the `rm_url` field in each teacher config in `--mopd-teachers`.
- `--custom-rm-path` and `--custom-reward-post-process-path` are auto-configured when not explicitly set (you typically don't need to specify them manually).
- `--rm-url` serves as a fallback URL if no per-teacher URL is configured.
- **Supported distill types**: `token_level` and `top_k`. `full_vocab` is not supported (SGLang cannot efficiently return full-vocabulary logits).
- **`top_k` specifics**: The SGLang server is queried with `top_logprobs_num=k` to return per-position top-k logprobs. During training, global token IDs from SGLang are converted to per-TP-shard local indices with `-inf` padding for out-of-shard entries.

### Megatron Mode

- Teacher models are loaded into CPU memory via `TensorBackuper` and switched to GPU for forward passes during training.
- Requires `--enable-weights-backuper` (default) for weight backup/restore.
- Each teacher must have the **same architecture** as the policy model.
- Memory note: each teacher model occupies additional CPU memory for weight backup.

## Components

- `slime/rollout/mopd.py` implements SGLang-mode MOPD:
  - `reward_func`: queries all teacher SGLang servers concurrently, returns per-domain responses. For `top_k` mode, the SGLang request includes `top_logprobs_num=k`.
  - `post_process_rewards`: extracts teacher data from SGLang responses — token-level log-probs, and (for `top_k` mode) top-k logit values and global token indices. Stores them in `sample.mopd_teacher_log_probs`, `sample.mopd_teacher_topk_logits`, and `sample.mopd_teacher_topk_indices`.
- `slime/ray/rollout.py`: collects per-sample MOPD data from rollouts and splits by data parallelism.
- `slime/backends/megatron_utils/actor.py`: for SGLang `top_k` mode, converts global token IDs to per-TP-shard local indices with `-inf` padding for out-of-shard entries.
- `slime/backends/megatron_utils/loss.py`:
  - `apply_mopd_topk_to_loss`: computes IS-weighted top-k approximate reverse KL loss.
  - `policy_loss_function`: integrates MOPD KL loss with the policy gradient loss.
- `slime/utils/ppo_utils.py`:
  - `vocab_parallel_topk_reverse_kl`: TP-aware top-k KL computation with tail correction and `valid_topk_mask` support.
- `run-qwen3-8B-mopd-sglang.sh`: launches SGLang teacher servers, then submits a Ray job.
- `run-qwen3-8B-mopd-megatron.sh`: uses Megatron-loaded teacher models (no external server needed).

## Running the Example

### Using SGLang Teachers (External Servers)

1. Download or prepare the required checkpoints and data:
```bash
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-Coder-32B --local-dir /root/Qwen3-Coder-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
```

2. Convert student model to Megatron format:
```bash
cd /root/slime
source scripts/models/qwen3-8B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist
```

3. Run MOPD with SGLang teachers:
```bash
bash examples/multi_teacher_on_policy_distillation/run-qwen3-8B-mopd-sglang.sh
```

The script will:
- Launch math and code teacher SGLang servers automatically
- Set `MOPD_TEACHER_URLS` environment variable
- Submit the training job via Ray

### Using Megatron Teachers (No External Server)

1. Prepare student checkpoint (same as above).

2. Convert teacher models to Megatron format:
```bash
cd /root/slime
source scripts/models/qwen3-8B.sh  # Or your teacher model config

# Math teacher
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B-Math \
    --save /root/Qwen3-8B-Math_torch_dist

# Code teacher
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B-Code \
    --save /root/Qwen3-8B-Code_torch_dist
```

> **Note**: This example uses the same model architecture for student and teachers. In practice, use **stronger** models as teachers.

3. Edit `run-qwen3-8B-mopd-megatron.sh` to update paths:
   - Change `--mopd-teacher-loads` to your teacher model paths
   - Adjust `--mopd-alpha`, `--mopd-eps-low`, `--mopd-eps-high` for your task

4. Run:
```bash
bash examples/multi_teacher_on_policy_distillation/run-qwen3-8B-mopd-megatron.sh
```

## Customization

### Adding More Teachers

Add entries to `--mopd-teachers` JSON and corresponding paths:

```bash
--mopd-teachers '[{"name":"math_t","domain":"math"},{"name":"code_t","domain":"code"},{"name":"reason_t","domain":"reasoning"}]'
--mopd-teacher-loads /path/to/math_ckpt /path/to/code_ckpt /path/to/reasoning_ckpt
```

For SGLang mode, add the URL to `MOPD_TEACHER_URLS`:
```bash
export MOPD_TEACHER_URLS='{"math":"http://...","code":"http://...","reasoning":"http://..."}'
```

### Mixing Distillation with Task Rewards

Set `--mopd-alpha > 0` to blend reverse KL advantages with standard ORM advantages:
```bash
--mopd-alpha 0.5   # Equal weight between distillation and task reward
--rm-type math     # Required when alpha > 0: provides ORM reward signal
```

**Reward model requirements:**
- `--mopd-alpha 0.0` (pure distillation): No reward model needed. If `--rm-type` and `--custom-rm-path` are both unset, it defaults to `zero` (always returns 0.0). The learning signal comes entirely from the distillation KL advantages.
- `--mopd-alpha > 0` (distillation + ORM): You **must** set `--rm-type` or `--custom-rm-path`, otherwise an error is raised, because ORM advantages require a reward signal.

### Tuning IS Weight Clipping

The IS weight clipping bounds control the trade-off between bias and variance:
- Tighter bounds (e.g., `[0.5, 2.0]`): Lower variance but more bias
- Looser bounds (e.g., `[0.1, 10.0]`): Less bias but higher variance

### Per-Sample Domain Routing

By default, every sample is distilled from **all** configured teachers. For datasets where different samples belong to different domains (e.g., math problems should only learn from the math teacher, code problems from the code teacher), you can specify per-sample routing via the `mopd_domains` field in sample metadata.

#### Data Format

Add a `mopd_domains` field in the `metadata` of your JSONL data:

```jsonl
{"prompt": "Solve: x^2 - 5x + 6 = 0", "metadata": {"mopd_domains": ["math"]}}
{"prompt": "Write a Python quicksort", "metadata": {"mopd_domains": ["code"]}}
{"prompt": "Explain quantum mechanics", "metadata": {"mopd_domains": ["math", "code"]}}
{"prompt": "General question"} 
```

- `"mopd_domains": ["math"]` — only distill from the math teacher
- `"mopd_domains": ["code"]` — only distill from the code teacher
- `"mopd_domains": ["math", "code"]` — distill from both (equivalent to default)
- No `mopd_domains` field — distill from **all** teachers (backward compatible)

For string convenience, you can also use a single string instead of a list:
```jsonl
{"prompt": "Solve: x^2 - 5x + 6 = 0", "metadata": {"mopd_domains": "math"}}
```

#### How It Works

- **SGLang mode**: `reward_func` only queries the specified teacher servers, saving compute on unnecessary inference.
- **Megatron mode**: All teachers still run forward passes (no way to skip per-sample), but `apply_mopd_to_advantages` uses zero advantages for non-matching domains, effectively excluding them from the loss.

## Differences from OPD

| Feature | OPD | MOPD |
|---------|-----|------|
| Number of teachers | 1 | Multiple (configurable) |
| Advantage computation | `KL = log(π_T/π_θ)`, added to loss | Per-teacher reverse KL, averaged across domains |
| IS weight correction | Not included | Clipped IS weight `w_t ∈ [ε_low, ε_high]` |
| ORM combination | Via `--opd-kl-coef` | Via `--mopd-alpha` |
| Mutual exclusivity | `--use-opd` | `--use-mopd` (cannot use both) |

## FAQ

1. **Can I use MOPD with OPD at the same time?**
   No. `--use-mopd` and `--use-opd` are mutually exclusive. Use MOPD if you need multiple teachers.

2. **Do all teachers need to have the same architecture?**
   - Megatron mode: Yes, all teachers must share the same architecture as the policy model.
   - SGLang mode: No, each teacher can have a different architecture since they run on separate servers.

3. **How much extra memory does MOPD need in megatron mode?**
   Each teacher model requires CPU memory for weight backup (via `TensorBackuper`). The teacher weights are only loaded to GPU temporarily during the forward pass, then restored to CPU. Plan for `N × model_size` additional CPU memory where `N` is the number of teachers.

4. **What happens if a teacher server fails in SGLang mode?**
   The `reward_func` will log a warning and skip the failed teacher for that sample. The training will continue with remaining teachers, but the advantages will be biased. Monitor teacher server health carefully.

5. **Why is `--group-rm` not supported with MOPD?**
   MOPD's `reward_func` returns per-domain dicts (not scalar rewards), which is incompatible with the batch `group_rm` reward path. Use the default per-sample reward path (no `--group-rm`).

6. **What is the difference between `token_level`, `top_k`, and `full_vocab` distillation types?**
    - `token_level` (default): Approximates reverse KL using the sampled token log-prob difference `sg[log π_d(y_t) - log π_θ(y_t)]`. Efficient, works with SGLang and Megatron, but only captures KL at the sampled token position. Underestimates the true KL when student and teacher distributions diverge.
    - `top_k`: Computes approximate reverse KL using the teacher's top-k logits plus an analytical tail correction. Memory-efficient (~97% less than `full_vocab`). Works with both SGLang and Megatron teachers. Accuracy is very close to `full_vocab` for typical k values (1024+).
    - `full_vocab`: Computes the exact full-vocabulary reverse KL divergence `D_KL(π_θ ∥ π_d)`. Most accurate but memory-intensive (stores full `[R, V]` logits). Megatron-only (SGLang cannot efficiently return full vocab logits).
    - See the [Algorithm](#algorithm) section and [Comparison table](#comparison-of-distillation-types) for detailed formulas and memory analysis.

7. **When should I use `top_k` vs `full_vocab`?**
    Use `top_k` in most production scenarios — it captures >99% of the KL signal with ~3% of the memory of `full_vocab`. Works with both SGLang and Megatron teachers. Use `full_vocab` only when you need the exact KL for research validation or have ample GPU memory (Megatron teachers only). Start with `--mopd-topk-k 1024`; increase to 2048 or 4096 if the vocabulary is very large (e.g., V > 200K) and you observe the tail correction is too aggressive.

8. **How does the `top_k` tail correction work?**
    The top-k decomposition splits the KL into `KL_topk` (exact over the teacher's top-k tokens) and `KL_tail` (approximate over the remaining tokens). The tail correction method differs by teacher mode:
    - **Megatron mode**: `π_t_tail ≈ (V − V_eff) / V` — assumes uniform distribution over non-top-k tokens, where `V_eff = k × tp_size`. This is a conservative upper bound; the actual teacher tail mass is typically smaller, so the approximate KL slightly over-estimates the true KL. This approximation works well when k is large (e.g., k ≥ 1024) since the top-k entries capture most of the probability mass.
    - **SGLang mode**: `π_t_tail = 1 − Σ exp(log_prob)` — **exact** computation. SGLang returns full-vocabulary log-probs for the top-k tokens, so summing their probabilities directly gives the true top-k mass and the tail is computed exactly. No uniform assumption is needed. This makes SGLang mode's tail correction accurate even with small k (e.g., k=128).

9. **What is the memory usage of each distillation type?**
    Memory scales with vocab size V, tensor parallelism TP, batch B, and response length R:
    - `token_level`: Negligible (`B × R × 4B`).
    - `top_k` (k=1024): `B × R × k × 2 × 4B / TP`. Ratio vs full_vocab is approximately `2k/V` (~1–2%).
    - `full_vocab`: `B × R × V × 4B / TP`.
    Concrete examples vary by model: V=152K/TP=2/B=4/R=4096 gives top_k ~128 MB, full_vocab ~4.6 GB; V=248K/TP=8/B=4/R=4096 gives top_k ~16 MB, full_vocab ~1.9 GB.
    If OOM occurs with `full_vocab`, switch to `top_k` or reduce `--rollout-batch-size` / `--rollout-max-response-len`.

10. **What are the requirements for SGLang `top_k` mode?**
    - The SGLang teacher server must support the `top_logprobs_num` parameter (available in recent SGLang versions).
    - The teacher's **vocabulary size must exactly match** the student's `padded_vocab_size`. This is because global token IDs from the teacher are converted to per-TP-shard local indices during training. A vocab size mismatch would produce incorrect index mappings and silently corrupt the KL computation.
    - The `MOPD_TEACHER_URLS` environment variable must be set (JSON dict mapping domain names to SGLang `/generate` endpoints), or `--rm-url` must be provided as a fallback.
    - `--custom-rm-path` and `--custom-reward-post-process-path` are auto-configured when both are unset — you typically don't need to set them manually.