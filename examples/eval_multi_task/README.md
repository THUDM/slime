# Multi-Task Evaluation Example

## Configuring `multi_task.yaml`
- `eval.defaults` defines inference parameters shared by every dataset entry. Override them inside an individual dataset block if needed.
- `eval.datasets` enumerates the datasets to evaluate. Each entry should specify:
  - `name`: a short identifier that appears in logs and dashboards.
  - `path`: the path to the dataset JSONL file.
  - `rm_type`: which reward function to use for scoring.
  - `n_samples_per_eval_prompt`: how many candidate completions to generate per prompt.

## IFBench Notes
- When `ifbench` is used, `slime/rollout/rm_hub/ifbench.py` will automatically prepares the scoring environment, so no additional manual setup is required beyond providing the dataset path.
