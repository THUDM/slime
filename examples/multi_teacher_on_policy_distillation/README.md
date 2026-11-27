# Multi-Teacher On-Policy Distillation Example

This example shows how to run **multi-teacher on-policy distillation (MOPD)** using slime. One student (Qwen3.5-9B) is aligned to several domain teachers (math / code / ifeval), and every rollout sample is scored by the teacher named by its `domain` label, so the teacher count follows the data instead of the command line.

## Key Features

- **Data-driven teacher routing**: one prompt file with a `domain` column. The data source stamps the label on each sample and the router maps it to exactly one teacher; there is no per-file binding table.
- **Plain MOPD**: slime's `--use-opd` already folds `student_logp - teacher_logp` into the advantage per sample. Routing only decides which teacher scores each sample, so no custom loss or advantage function is involved.
- **Two teacher backends**: `megatron` keeps every teacher loaded in the actor, `sglang` queries external teacher servers during rollout.
- **Reward is optional**: the launcher uses a zero reward because the learning signal is the OPD KL penalty; `mopd_deepscaler_reward` / `mopd_humaneval_reward` can mix in task rewards.

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-opd` | Enable on-policy distillation. Required flag to use OPD. |
| `--opd-type` | Teacher backend: `megatron` or `sglang`. Must match the `backend` of every teacher in the config. |
| `--opd-kl-coef` | OPD KL penalty coefficient (default: 1.0). |
| `--opd-teacher-load` | Validated by slime even in Megatron mode. The launcher fills it with the first configured teacher; all Megatron teachers are loaded from the config. |
| `--custom-config-path` | MOPD YAML with the `mopd` block (teacher registry + routing). |
| `--data-source-path` | `slime_mopd.data_source.MopdRolloutDataSource`, which preserves the `domain` label. |
| `--custom-rm-path` | `slime_mopd.rm.mopd_zero_reward` for the Megatron backend, `slime_mopd.rm.mopd_reward_func` for the SGLang backend. |
| `--custom-reward-post-process-path` | `slime_mopd.post_process.mopd_post_process`. SGLang backend only. |
| `--custom-convert-samples-to-train-data-path` | `slime_mopd.train_data.mopd_convert_samples_to_train_data`. Megatron backend only. |

## Mode Comparison

| Mode | Teacher Location | When to use |
|------|------------------|-------------|
| `megatron` | Loaded into the actor via `TensorBackuper` under `teacher:<id>` tags | Teachers share the student architecture and fit in host RAM |
| `sglang` | External SGLang server per teacher (`rm_url`) | Teachers are larger, have a different architecture, or are shared between jobs |

The shipped launcher and configs use `megatron`.

## Data Format

One prompt file with one label per row:

| Field | Use |
|-------|-----|
| `prompt` | chat messages, or a plain string (wrapped as one user message) |
| `domain` | routing label, for example `math` / `code` / `ifeval`; also accepted as `teacher_id` or `metadata.domain` / `metadata.mopd_domain` |
| `label`, `metadata` | optional; kept on the sample, unused by the OPD loss |

`.jsonl` and `.parquet` are supported.

## Configuration

```yaml
mopd:
  teachers:
    - {id: math,   backend: megatron, load: ${MOPD_TEACHER_MATH}}
    - {id: code,   backend: megatron, load: ${MOPD_TEACHER_CODE}}
    - {id: ifeval, backend: megatron, load: ${MOPD_TEACHER_IFEVAL}}
  default_teacher: math   # optional fallback
  max_concurrency: 64     # SGLang teacher requests
```

| Field | Description |
|-------|-------------|
| `teachers[].id` | Teacher id, matched against the row's `domain` unless `domains` remaps it. |
| `teachers[].backend` | `megatron` or `sglang`. |
| `teachers[].load` | Checkpoint path for `megatron`. `${VAR}` is expanded. |
| `teachers[].rm_url` | SGLang server URL for `sglang`. |
| `teachers[].ckpt_step` | Optional checkpoint step. |
| `domains` | Optional `domain -> teacher id` map. |
| `default_teacher` | Fallback for rows with no matching teacher; unset makes them a hard error. |
| `active_teachers` | Optional subset of teacher ids to load. |
| `max_concurrency`, `request_timeout_s`, `max_retries` | SGLang request limits. |
| `logprob_dtype` | Stored teacher log-prob dtype: `auto` (model dtype), `float32`, `float16`, `bfloat16`. |

Routing precedence: `metadata.teacher` → `domain` → `default_teacher`.

The config is read from `--custom-config-path`, then from `SLIME_MOPD_CONFIG`. The launcher resolves `${VAR}` into a plain YAML before submitting, so the same file works on every host.

## Components

- `slime_mopd/data_source.py` implements `MopdRolloutDataSource`: reads the prompt file, applies the chat template, and stamps the domain on each sample's `source`.
- `slime_mopd/config.py`, `slime_mopd/routing.py`: teacher registry, domain mapping, and per-sample teacher resolution.
- `slime_mopd/rm.py` provides the SGLang reward hooks (`mopd_reward_func`, `mopd_batched_reward_func`) and `mopd_zero_reward`; `slime_mopd/post_process.py` trims teacher log-probs to the response span and writes `sample.teacher_log_probs`.
- `slime_mopd/megatron_actor.py` implements `MopdMegatronTrainRayActor`: loads every teacher once under `teacher:<id>` and forwards only each sample's routed teacher.
- `slime_mopd/train_data.py` ships per-sample teacher ids into `train_data`; `slime_mopd/train_mopd.py` loads slime's `train.py` unchanged and injects the MOPD actor.
- `run-qwen3.5-9B-mopd.sh` starts or reuses Ray and submits the job.
- `config/qwen3.5.three-teachers.yaml` is the math + code + ifeval registry; `config/qwen3.5.single-teacher.yaml` routes every domain to the math teacher for ablation.

## Running the Example

### Using Megatron Teachers (No External Server)

1. Prepare the student and teacher checkpoints in HF format (the recipe keeps `--kl-loss-coef 0`, so no reference model is loaded), then export the paths:
```bash
export MODEL_DIR=/root/models/Qwen3.5-9B
export PROMPT_DATA=/root/datasets/mopd/train_3domain.jsonl
export MOPD_TEACHER_MATH=/root/models/Qwen3.5-9B-teacher-math
export MOPD_TEACHER_CODE=/root/models/Qwen3.5-9B-teacher-code
export MOPD_TEACHER_IFEVAL=/root/models/Qwen3.5-9B-teacher-ifeval
```

2. Launch (8 GPUs: 4 actor with TP=2, 4 rollout as two TP=2 SGLang engines):
```bash
bash examples/multi_teacher_on_policy_distillation/run-qwen3.5-9B-mopd.sh
```

Useful overrides: `NUM_GPUS`, `ACTOR_GPUS`, `ROLLOUT_GPUS_PER_ENGINE`, `NUM_ROLLOUT`, `ROLLOUT_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`, `GLOBAL_BATCH_SIZE`, `MAX_RESPONSE_LEN`, `TP_SIZE`, `CP_SIZE`, `LR`, `SAVE_PATH`, `MOPD_CONFIG` (point it at `config/qwen3.5.single-teacher.yaml` for the single-teacher ablation).

### Using SGLang Teachers (External Server)

1. Start one SGLang server per domain teacher and record its `/generate` URL.

2. Set `backend: sglang` and `rm_url` for each teacher in the config (for example `http://<TEACHER_IP>:<TEACHER_PORT>/generate`).

3. Replace the launcher's Megatron OPD flags and hooks with the SGLang ones (omit `--opd-teacher-load` and `--custom-convert-samples-to-train-data-path`, and run `train.py` directly):
```bash
--use-opd --opd-type sglang --opd-kl-coef 1.0 \
--custom-rm-path slime_mopd.rm.mopd_reward_func \
--custom-reward-post-process-path slime_mopd.post_process.mopd_post_process
```

## FAQ

1. **Why is the reward zero?**
   The learning signal is the OPD KL penalty, not the environment reward. slime still requires a custom rm, so `mopd_zero_reward` returns 0. Use `mopd_deepscaler_reward` or `mopd_humaneval_reward` to mix in task rewards.

2. **Why does `--opd-teacher-load` still need a value in Megatron mode?**
   slime validates the flag whenever `--use-opd` is set. The launcher fills it with the first configured teacher; the MOPD actor loads every teacher from the config and suppresses slime's single-teacher path.

3. **What happens when a sample has no matching teacher?**
   A domain that maps to no teacher is rejected at startup unless `default_teacher` is set. Unlabelled rows also fall back to `default_teacher`; without it, routing raises at rollout. Fallbacks are counted and reported per batch.

4. **How expensive is Megatron teacher scoring?**
   Each teacher is a pinned-CPU weight copy scored by the actor, so three 9B teachers need host RAM for three copies and three weight swaps per step.
