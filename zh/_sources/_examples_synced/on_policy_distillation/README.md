# On-Policy Distillation Example

This example shows how to run **on-policy distillation (OPD)** using slime. A small student (Qwen3-8B) is aligned to imitate a larger teacher (Qwen3-32B) by training only on the student's own rollouts and matching the teacher's token-level log-probabilities.

## Key Features

- **OPD is orthogonal to advantage estimators**: OPD works as an additive KL penalty on top of any advantage estimator (GRPO, PPO, REINFORCE++, etc.), not as a separate estimator.
- **Two teacher modes**:
  - **sglang**: Teacher runs on an external SGLang server, teacher log-probs are obtained during rollout.
  - **megatron**: Teacher is loaded directly into Megatron via `--opd-teacher-load`, teacher log-probs are computed during training forward pass.

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-opd` | Enable on-policy distillation. Required flag to use OPD. |
| `--opd-type` | Type of OPD: `sglang` or `megatron`. Required when `--use-opd` is set. |
| `--opd-kl-coef` | OPD KL penalty coefficient (default: 1.0). |
| `--opd-teacher-load` | Path to teacher checkpoint. **Required** when `--opd-type=megatron`, **must not be set** when `--opd-type=sglang`. |
| `--opd-teacher-ckpt-step` | Optional checkpoint step for teacher model. |

## Mode Comparison

| Mode | Teacher Location | When to use |
|------|------------------|-------------|
| `sglang` | External SGLang server | Teacher has different architecture or larger than GPU memory |
| `megatron` | Loaded into Megatron training | Teacher has same architecture as policy/ref model |

## Components

- `on_policy_distillation.py` implements (for SGLang mode):
  - `reward_func` calls the teacher server (via `args.rm_url`) with every sample to obtain token-level logprobs.
  - `post_process_rewards` trims the teacher logprobs to the generated response span and writes the tensors back to each `Sample` to compute advantages.
- `run-qwen3-8B-opd.sh` launches an SGLang teacher server, then submits a Ray job that runs `train.py`.
- `run-qwen3-8B-opd-megatron.sh` uses Megatron-loaded teacher model (no external server needed).

## Running the example

### Using SGLang Teacher (External Server)

1. Download or prepare the required checkpoints and data.
```bash
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
```

2. Run the hf to mcore for student model conversion:
```bash
cd /root/slime
source scripts/models/qwen3-8B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist
```

3. Run on-policy distillation:
```bash
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```

### Using Megatron Teacher (No External Server)

1. Prepare student checkpoint (same as above).

2. **IMPORTANT**: Convert your teacher model to Megatron format (change the path to your actual teacher):
```bash
# This example uses the same model as both student and teacher (for demonstration only)
# In practice, use a different (stronger) model as the teacher!
cd /root/slime
source scripts/models/qwen3-8B.sh  # Or your teacher model config

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/YourTeacherModel \
    --save /root/YourTeacherModel_torch_dist
```

3. Edit `run-qwen3-8B-opd-megatron.sh` to update paths:
   - Change `--opd-teacher-load` to your teacher model path
   - Adjust `--opd-kl-coef` based on your task

4. Run:
```bash
bash examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh
```


# Preliminary Results
Using Qwen3-8B-Base model sfted on part of the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset, we performed on-policy distillation with a Qwen3-32B teacher on the remaining data. Evaluation on Math500 shows:

|                                  | Pass@1 |
|-----------------------------------------------|--------|
| Qwen3-8B-Base + SFT                           | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation  | 94%    |





# FAQ
1. **Why are there two OPD modes?**
   - `sglang` mode: The teacher runs on an independent SGLang server. This is useful when the teacher has a different architecture or is too large to load together with the policy model.
   - `megatron` mode: The teacher is loaded into Megatron using the same parameter loading mechanism as the reference model. This requires the teacher to have the same architecture as the policy model.

2. **How do I use Megatron-based teacher instead of SGLang server?**
   Replace your OPD arguments:
   ```bash
   # Instead of:
   --use-opd --opd-type sglang --opd-kl-coef 1.0
   # Use:
   --use-opd --opd-type megatron --opd-kl-coef 1.0 --opd-teacher-load /path/to/teacher_checkpoint
   ```

3. **What happens if I set wrong arguments?**
   The system will raise clear errors:
   - `--use-opd` without `--opd-type`: Error asking you to specify type
   - `--opd-type megatron` without `--opd-teacher-load`: Error asking for teacher checkpoint
   - `--opd-type sglang` with `--opd-teacher-load`: Error indicating conflict


# References
1. https://thinkingmachines.ai/blog/on-policy-distillation/
2. https://arxiv.org/abs/2306.13649
3. https://arxiv.org/abs/2306.08543