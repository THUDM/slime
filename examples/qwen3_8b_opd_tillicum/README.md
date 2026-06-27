# Qwen3-8B SFT + OPD on Tillicum

This directory contains thin Tillicum wrappers for a Qwen3-8B SFT plus
on-policy distillation experiment using OpenThoughts3-1.2M math data.

The wrappers intentionally reuse slime's existing paths:

- `docs/en/get_started/quick_start.md` for the container and Megatron
  conversion workflow.
- `scripts/models/qwen3-8B.sh` for the student Megatron model args.
- `examples/on_policy_distillation/run-qwen3-8B-opd.sh` for the SGLang
  teacher OPD shape.
- `scripts/run-qwen3-4B-base-sft.sh` for the SFT rollout/training shape.
- `examples/eval_multi_task/multi_task.sh` and `multi_task.yaml` for eval.

No slime core code is modified.

## Required environment

Source `env.sh` before running commands:

```bash
cd /gpfs/scrubbed/suryadv/repos/slime
source examples/qwen3_8b_opd_tillicum/env.sh
```

Important variables:

- `ACCOUNT`: Slurm account. Default: `raivn`.
- `PARTITION`: Slurm partition. Default: `gpu-h200`.
- `QOS`: Slurm QOS. Default: `normal`.
- `SCRATCH_ROOT`: root for all generated data, checkpoints, caches, logs, and
  the Apptainer image. Default:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd`.
- `DATA_ROOT`: prepared datasets. Default: `$SCRATCH_ROOT/data`.
- `MODEL_ROOT`: HF model snapshots and Megatron torch_dist conversion.
  Default: `$SCRATCH_ROOT/models`.
- `OUTPUT_ROOT`: training/eval outputs. Default: `$SCRATCH_ROOT/outputs`.
- `HF_HOME`: Hugging Face cache under scratch. Default: `$SCRATCH_ROOT/hf_home`.
- `WANDB_MODE`: default `offline`.
- `SLIME_CONTAINER_FORMAT`: Apptainer image format. Default: `sandbox`,
  because Tillicum's Apptainer produced invalid SquashFS SIFs for this large
  Docker image during testing. Set to `sif` to force SIF output.
- `SLIME_SIF`: Apptainer image/sandbox path. Default:
  `$SCRATCH_ROOT/containers/slime_latest.sandbox`.

The scripts avoid writing caches/checkpoints/data under home.

## Dry checks

Dry checks do not pull the container, download data/models, install packages, or
submit real jobs.

```bash
cd /gpfs/scrubbed/suryadv/repos/slime
source examples/qwen3_8b_opd_tillicum/env.sh
bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh
```

The dry check runs `bash -n`, Python bytecode compilation, and
`sbatch --test-only` for the four Slurm scripts. If `$SLIME_SIF` already
exists and `RUN_CONTAINER_CHECKS=1` is set, it also verifies imports inside the
container.

## Setup and launch after approval

Run the setup steps manually, in this order:

```bash
bash examples/qwen3_8b_opd_tillicum/00_pull_or_load_container.sh
bash examples/qwen3_8b_opd_tillicum/01_prepare_env.sh
bash examples/qwen3_8b_opd_tillicum/container_exec.sh \
  python examples/qwen3_8b_opd_tillicum/02_prepare_openthoughts3_math_sample.py
```

Then launch only after explicit approval:

```bash
jid0=$(sbatch -A "$ACCOUNT" -p "$PARTITION" --qos "$QOS" \
  examples/qwen3_8b_opd_tillicum/03_convert_models_if_needed.sbatch | awk '{print $4}')
jid1=$(sbatch -A "$ACCOUNT" -p "$PARTITION" --qos "$QOS" --dependency=afterok:$jid0 \
  examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch | awk '{print $4}')
jid2=$(sbatch -A "$ACCOUNT" -p "$PARTITION" --qos "$QOS" --dependency=afterok:$jid1 \
  examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch | awk '{print $4}')
sbatch -A "$ACCOUNT" -p "$PARTITION" --qos "$QOS" --dependency=afterok:$jid2 \
  examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
```

## Outputs

- SFT split: `$SFT_PARQUET`
- OPD prompt split: `$OPD_JSONL`
- Data metadata: `$SPLIT_METADATA`
- Student HF snapshot: `$STUDENT_HF_DIR`
- Teacher HF snapshot: `$TEACHER_HF_DIR`
- Student Megatron torch_dist: `$STUDENT_TORCH_DIST_DIR`
- SFT checkpoint: `$SFT_SAVE_DIR`
- OPD checkpoint: `$OPD_SAVE_DIR`
- Eval summaries: `$EVAL_OUTPUT_DIR`

## Slurm resources

Each job requests one node with `--gres=gpu:h200:8`, `--ntasks=1`,
`--cpus-per-task=64`, and all node memory. The account, partition, and QOS are
passed at submit time from the environment variables above.

The intended wall-clock budget after model/data/container preparation is:

- SFT 100k: 2-4 hours.
- OPD 50k: 2-3 hours.
- MATH-500 greedy eval once for base, SFT, and OPD: <=1 hour.

The main runtime risk is the Qwen3-32B teacher logprob server throughput during
OPD. If dry measurements show the full OPD run will exceed the budget, the
first reduced run to try is 100k SFT plus 25k OPD.
