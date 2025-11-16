## Usage

All test cases use the template script `rollout_correction_template.sh` with different arguments.

**Important**: All commands should be run from the slime main directory (`/root/slime`):

```bash
cd /root/slime
bash examples/train_infer_mismatch_helper/tests/rollout_correction_template.sh [--use-rollout-correction] [--use-rollout-logprobs] [--custom-config NAME] [--wandb-prefix PREFIX]
```

## Test Cases

### Case 0: Baseline
No rollout correction, no rollout logprobs, no custom config.

```bash
cd /root/slime
bash examples/train_infer_mismatch_helper/tests/rollout_correction_template.sh --wandb-prefix 0_baseline
```

### Case 1: Pure IS
Use both rollout correction and rollout logprobs with custom config.

```bash
bash examples/train_infer_mismatch_helper/tests/rollout_correction_template.sh --use-rollout-correction --use-rollout-logprobs --custom-config 1_pure_is --wandb-prefix 1_pure_is
```

### Case 2: Use Rollout in PPO
Use rollout logprobs only, no custom config.

```bash
bash examples/train_infer_mismatch_helper/tests/rollout_correction_template.sh --use-rollout-logprobs --wandb-prefix 2_use_rollout_in_ppo
```

### Case 3: Token IS
Use rollout correction with custom config.

```bash
bash examples/train_infer_mismatch_helper/tests/rollout_correction_template.sh --use-rollout-correction --custom-config 3_token_is --wandb-prefix 3_token_is
```

### Case 4: Seq IS
Use rollout correction with custom config.

```bash
bash examples/train_infer_mismatch_helper/tests/rollout_correction_template.sh --use-rollout-correction --custom-config 4_seq_is --wandb-prefix 4_seq_is
```

### Case 5: Seq IS RS
Use rollout correction with custom config.

```bash
bash examples/train_infer_mismatch_helper/tests/rollout_correction_template.sh --use-rollout-correction --custom-config 5_seq_is_rs --wandb-prefix 5_seq_is_rs
```

### Case 6: Geo RS
Use rollout correction with custom config.

```bash
bash examples/train_infer_mismatch_helper/tests/rollout_correction_template.sh --use-rollout-correction --custom-config 6_geo_rs --wandb-prefix 6_geo_rs
```

## Quick Reference Table

| Case | Rollout Correction | Rollout Logprobs | Custom Config | WandB Prefix |
|------|-------------------|------------------|---------------|--------------|
| 0    | ❌                | ❌               | -             | 0_baseline   |
| 1    | ✅                | ✅               | 1_pure_is     | 1_pure_is    |
| 2    | ❌                | ✅               | -             | 2_use_rollout_in_ppo |
| 3    | ✅                | ❌               | 3_token_is    | 3_token_is   |
| 4    | ✅                | ❌               | 4_seq_is      | 4_seq_is     |
| 5    | ✅                | ❌               | 5_seq_is_rs   | 5_seq_is_rs  |
| 6    | ✅                | ❌               | 6_geo_rs      | 6_geo_rs     |

