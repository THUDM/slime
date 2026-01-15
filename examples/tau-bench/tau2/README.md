# Tau2 Pipeline (tau2-bench)

This folder contains the tau2-bench integration used by the canonical cookbook in `examples/tau-bench/training_cookbook.md`.

Start here: `examples/tau-bench/training_cookbook.md`.

## What's in this folder

- `run_sft.sh`, `run_grpo.sh`: convenience scripts for SFT and GRPO (write under `TAU_BENCH_OUT_DIR`)
- `start_user_sim_server.sh`: starts a local user simulator server for GRPO/eval (port 30001)
- `rollout.py`: GRPO rollout entrypoint (`--custom-generate-function-path rollout.generate`)
- `tasks.py`: generates `tau2_{split}_all_tasks.jsonl` task index files under `TAU_BENCH_OUT_DIR`
- `eval.py`: unified evaluation harness (supports Pass@1 with `--num-samples=1 --temperature=0.0` for greedy, and Pass@K with `--num-samples=K` for multi-sampling; Pass@4 is the headline metric; pass@k = any success among k attempts; defaults to GPT-4.1-mini user sim)
- `prompting.py`, `actions.py`, `env.py`, `reward.py`: utilities used by rollouts/eval

## Pre-generated dataset

The cookbook uses a pinned pre-generated SFT dataset by default:
`Jarrodbarnes/tau2-sft-seed-v3` on Hugging Face.
