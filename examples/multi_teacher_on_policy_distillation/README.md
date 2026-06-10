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

For each teacher domain *d*, MOPD computes:

```
reverse_kl_d = sg[log π_d(y_t) - log π_θ(y_t)]          # per-teacher reverse KL
w_t = sg[π_θ(y_t) / μ_θ(y_t)]  clipped to [ε_low, ε_high] # IS weight
Â_MOPD,t = (1/D) Σ_d (reverse_kl_d + α · Â_ORM)          # averaged across D teachers
L = -E[1/|y| Σ_t w_t · Â_MOPD,t · log π_θ(y_t)]          # proxy policy loss
```

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

## SGLang vs Megatron Mode

| Mode | Teacher Location | When to use |
|------|------------------|-------------|
| `sglang` | External SGLang servers (one per teacher) | Teachers have different architecture or are too large for training GPU memory |
| `megatron` | Loaded into Megatron training process | Teachers have the same architecture as the policy/ref model |

### SGLang Mode

- Each teacher runs as an independent SGLang server.
- Teacher URLs are configured via the `MOPD_TEACHER_URLS` environment variable (JSON dict: `domain -> URL`) or via the `rm_url` field in each teacher config in `--mopd-teachers`.
- `--custom-rm-path slime.rollout.mopd.reward_func` and `--custom-reward-post-process-path slime.rollout.mopd.post_process_rewards` are required.
- `--rm-url` serves as a fallback URL if no per-teacher URL is configured.

### Megatron Mode

- Teacher models are loaded into CPU memory via `TensorBackuper` and switched to GPU for forward passes during training.
- Requires `--enable-weights-backuper` (default) for weight backup/restore.
- Each teacher must have the **same architecture** as the policy model.
- Memory note: each teacher model occupies additional CPU memory for weight backup.

## Components

- `slime/rollout/mopd.py` implements SGLang-mode MOPD:
  - `reward_func`: queries all teacher SGLang servers concurrently, returns per-domain responses.
  - `post_process_rewards`: extracts token-level teacher log-probs from responses and stores them in `sample.mopd_teacher_log_probs`.
- `slime/backends/megatron_utils/loss.py`:
  - `apply_mopd_to_advantages`: computes per-teacher reverse KL, IS weights, and aggregated MOPD advantages.
  - `policy_loss_function`: applies `mopd_advantages` and IS weights to the policy gradient loss.
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