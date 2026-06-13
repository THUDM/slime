# On-Policy Distillation

On-policy distillation (OPD) enables a student model to learn from a larger teacher model by training on its own rollouts while matching the teacher's token-level log-probabilities. OPD is orthogonal to advantage estimators — it works as an additive KL penalty on top of any estimator (GRPO, PPO, REINFORCE++, etc.).

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-opd` | Enable on-policy distillation. Required flag to use OPD. |
| `--opd-type` | Type of OPD: `sglang` or `megatron`. Required when `--use-opd` is set. |
| `--opd-kl-coef` | OPD KL penalty coefficient (default: 1.0). Controls the weight of the distillation signal relative to the RL advantage. |
| `--opd-teacher-load` | Path to teacher Megatron checkpoint. **Required** when `--opd-type=megatron`, **must not be set** when `--opd-type=sglang`. |
| `--opd-teacher-ckpt-step` | Optional checkpoint step for teacher model. |

## How It Works

OPD modifies the advantage computation by subtracting a KL penalty term that encourages the student to match the teacher's output distribution:

$$
\hat{A}_t = A_t - \lambda_{\text{opd}} \cdot D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})_t
$$

Where $A_t$ is the original advantage from the base estimator (e.g., GRPO), $\lambda_{\text{opd}}$ is `--opd-kl-coef`, and $D_{\text{KL}}$ is the token-level reverse KL divergence.

This means OPD can be combined with any advantage estimator, including GRPO, PPO, REINFORCE++, and GSPO.

## Two Teacher Modes

### SGLang Mode (`--opd-type sglang`)

The teacher runs on an external SGLang server. Teacher log-probs are obtained during the rollout phase.

**When to use**: The teacher has a different architecture from the student, or the teacher is too large to load alongside the training model.

**How it works**:
1. An external SGLang server runs the teacher model.
2. During rollout, the custom reward function (`slime.rollout.on_policy_distillation.reward_func`) sends each sample to the teacher server to obtain token-level log-probs.
3. The custom post-processing function (`slime.rollout.on_policy_distillation.post_process_rewards`) trims the teacher log-probs to the response span and stores them in `sample.teacher_log_probs`.
4. During training, the KL penalty is computed from the stored teacher log-probs and applied to advantages.

**Configuration**:
```bash
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
--custom-rm-path slime.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
--rm-url http://<TEACHER_IP>:<TEACHER_PORT>/generate
```

### Megatron Mode (`--opd-type megatron`)

The teacher model is loaded directly into Megatron via `--opd-teacher-load`. Teacher log-probs are computed during the training forward pass.

**When to use**: The teacher has the same architecture as the student/reference model and fits in GPU memory.

**How it works**:
1. The teacher model is loaded as an additional Megatron model during initialization.
2. During the training forward pass, the teacher model computes log-probs for each sample.
3. The KL penalty is computed inline and applied to advantages.

**Configuration**:
```bash
--use-opd
--opd-type megatron
--opd-kl-coef 1.0
--opd-teacher-load /path/to/teacher_torch_dist
```

> **Note**: The teacher checkpoint must be in Megatron format (`torch_dist` or `torch`). You can convert from HuggingFace format using `tools/convert_hf_to_torch_dist.py`.

### Self Mode (`--opd-type self`) — On-Policy Self-Distillation (OPSD)

This mode implements **On-Policy Self-Distillation** ("Self-Distilled Reasoner"). A single model acts as both student and teacher, differing only by *context*:

- **Student**: conditioned on the problem only; generates the on-policy rollout (the trainable current policy).
- **Teacher**: the **same model, frozen at the initial-policy checkpoint** (`--opd-teacher-load`), conditioned on the problem **plus privileged information** (the ground-truth solution). The teacher does not generate — it scores the student's response tokens in a single forward pass.

Unlike the other modes (which fold a sampled-token reverse-KL into the advantage), OPSD uses a **direct, full-vocab token-level Jensen-Shannon divergence (JSD)** as the loss, with **no task reward** (pure distillation):

$$
\mathcal{L}_{\text{OPSD}} = \mathbb{E}_{t}\Big[\min\big(\text{JSD}_\beta\big(P_{\text{teacher}}(\cdot|x, s, y_{<t})\,\|\,P_{\text{student}}(\cdot|x, y_{<t})\big),\ c\big)\Big]
$$

where $x$ is the problem, $s$ the privileged solution, $\beta$ the JSD interpolation weight (`--opsd-beta`: 0 → forward KL, 1 → reverse KL, 0.5 → symmetric), and $c$ the per-token clip (`--opsd-jsd-clip`, default 0.05) that prevents high-divergence style tokens (e.g. "wait", "think") from dominating the gradient.

**Key arguments**:

| Argument | Description |
|----------|-------------|
| `--opsd-beta` | JSD interpolation weight (default 0.5). |
| `--opsd-jsd-clip` | Per-token JSD clamp (default 0.05). |
| `--opsd-privileged-info-key` | Dataset field with the privileged information (ground-truth solution) shown only to the teacher. **Required**. |

**How it works**:
1. The teacher (frozen init checkpoint) is loaded as an additional Megatron model.
2. During rollout conversion, the privileged teacher sequence `[prompt + privileged_info + response]` is built (the response is a verbatim copy of the student's rollout).
3. Before training, the teacher is forwarded on that privileged sequence to obtain response-position logits.
4. During the student training forward, the per-token full-vocab JSD between student and teacher logits is computed, clamped, and reduced into the loss.

**Configuration**:
```bash
--use-opd
--opd-type self
--opd-teacher-load /path/to/init_torch_dist   # frozen teacher = initial policy
--opsd-beta 0.5
--opsd-jsd-clip 0.05
--opsd-privileged-info-key solution           # dataset field with the ground-truth solution
```

> **Limitations (current MVP)**: OPSD requires `--context-parallel-size 1`. The teacher response logits are held over the full vocabulary between the teacher and student forwards, so memory scales with `response_length × vocab_size`; prefer it for smaller models / shorter responses for now. The example script is `examples/on_policy_distillation/run-qwen3-8B-opsd.sh`.

## Running the Examples

Complete example scripts are provided in `examples/on_policy_distillation/`:

### SGLang Teacher

```bash
# 1. Download models and data
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# 2. Convert student model
cd /root/slime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist

# 3. Run
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```

### Megatron Teacher

```bash
# 1. Convert both student and teacher models to Megatron format
# 2. Run
bash examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh
```

## Preliminary Results

Using Qwen3-8B-Base model SFT-ed on part of the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset, on-policy distillation with a Qwen3-32B teacher on the remaining data yields:

|                                  | Pass@1 |
|-----------------------------------------------|--------|
| Qwen3-8B-Base + SFT                           | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation  | 94%    |
