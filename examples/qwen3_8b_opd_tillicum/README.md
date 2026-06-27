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
bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh
```

Then launch only after explicit approval:

```bash
bash examples/qwen3_8b_opd_tillicum/submit_25k_10k_chain.sh
```

## Reproducing On Another Slurm Cluster

These wrappers are Tillicum-shaped, but the core workflow is portable to a
Slurm cluster with Apptainer or Singularity, one 8-GPU node per job, outbound
access to Hugging Face and Docker Hub, and enough scratch space for large model
checkpoints.

Clone the fork and switch to the reproduction branch:

```bash
git clone https://github.com/suryathecreator/slime.git
cd slime
git checkout opd-reproduction
```

Choose cluster-local paths and Slurm settings. Keep all generated state on
scratch or project storage, not home:

```bash
export ACCOUNT="<your-account>"
export PARTITION="<your-gpu-partition>"
export QOS="<your-qos>"
export SCRATCH_ROOT="/path/to/scratch/${USER}/slime-qwen3-8b-opd"
export CONTAINER_BIND_ROOTS="$(pwd),${SCRATCH_ROOT},/tmp"

# Match your site's GPU gres. Examples: gpu:8, gpu:a100:8, gpu:h100:8.
export GPU_GRES="gpu:8"

# Use sif if your Apptainer can build a normal SIF for slimerl/slime:latest.
export SLIME_CONTAINER_FORMAT="sandbox"

source examples/qwen3_8b_opd_tillicum/env.sh
```

Run the same preparation commands:

```bash
bash examples/qwen3_8b_opd_tillicum/00_pull_or_load_container.sh
bash examples/qwen3_8b_opd_tillicum/01_prepare_env.sh
bash examples/qwen3_8b_opd_tillicum/container_exec.sh \
  python examples/qwen3_8b_opd_tillicum/02_prepare_openthoughts3_math_sample.py
```

Submit the dependency chain, overriding the Tillicum `h200` gres embedded in
the sbatch files:

```bash
GPU_GRES=gpu:h200:8 bash examples/qwen3_8b_opd_tillicum/submit_25k_10k_chain.sh
```

If your cluster uses `--gpus-per-node` instead of `--gres`, replace the
`--gres "$GPU_GRES"` arguments above with your site's GPU request flag. If your
cluster uses Docker rather than Apptainer/Singularity, run the same Python and
Slurm entrypoints inside `slimerl/slime:latest` and bind the repo plus
`$SCRATCH_ROOT` into the container; `container_exec.sh` is the only
Apptainer-specific layer.

## Outputs

- SFT split: `$SFT_PARQUET` (JSONL by default despite the legacy variable
  name).
- OPD prompt split: `$OPD_JSONL`
- Data metadata: `$SPLIT_METADATA`
- Student HF snapshot: `$STUDENT_HF_DIR`
- Teacher HF snapshot: `$TEACHER_HF_DIR`
- Student Megatron torch_dist: `$STUDENT_TORCH_DIST_DIR`
- SFT full optimizer checkpoint: `$SFT_SAVE_DIR`
- OPD full optimizer checkpoint: `$OPD_SAVE_DIR`
- SFT model-only eval snapshots: `$SFT_HF_SNAPSHOT_DIR`
- OPD model-only eval snapshots: `$OPD_HF_SNAPSHOT_DIR`
- Eval summaries and curves: `$BASE_EVAL_OUTPUT_DIR`, `$SFT_EVAL_OUTPUT_DIR`,
  `$OPD_EVAL_OUTPUT_DIR`
- Checkpoint storage reports: `$CHECKPOINT_REPORT_DIR`

## Slurm resources

Each job requests one node with `--gres=gpu:h200:8`, `--ntasks=1`,
`--cpus-per-task=64`, and all node memory. The account, partition, and QOS are
passed at submit time from the environment variables above.

The intended wall-clock budget after model/data/container preparation is:

- SFT 25k: 3 hours.
- OPD 10k: 5 hours.
- MATH-500 greedy eval: base 1 hour, SFT curve 3 hours, OPD curve 3 hours.

The main runtime risk is the Qwen3-32B teacher logprob server throughput during
OPD. The current conservative chain evaluates base first, then SFT milestones,
then OPD milestones so SFT results are available before OPD starts.
