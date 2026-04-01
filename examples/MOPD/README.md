# Multi-domain on-policy distillation (MOPD) example

This folder mirrors the experimental OPD multidomain setup under `exp/opd/`, with **generic paths** so you can copy or adapt it without site-specific directories.

## Files

| File | Purpose |
|------|---------|
| `sglang_opd_multimodel.yaml` | Declares every SGLang model server slime starts for rollout + teacher scoring. |
| `run-opd-multidomain.sh` | Starts Ray, submits `train_async.py` with OPD + multimodel SGLang flags. |

## 1. Multimodel YAML (`sglang_opd_multimodel.yaml`)

Slime reads this file via `--sglang-config` and launches one SGLang stack per top-level entry under `sglang`.

- **Entry at index 0 must be the student**  
  - Used for on-policy **rollout generation**.  
  - Must have `update_weights: true` so training can push updated weights to this server.

- **All following entries are teachers**  
  - Typically `update_weights: false`.  
  - Used when the custom RM calls the teacher (e.g. `reward_func_route_by_domain`) to obtain token log-probs for OPD.

- **Minimize GPUs on teachers when you can**  
  Teachers only need a forward pass with `max_new_tokens: 0` and `return_logprob: true` (no long generation). Their memory and throughput needs are usually much smaller than the student’s. Prefer **`num_gpus: 1`** (and `num_gpus_per_engine: 1`) per teacher unless the model is too large for a single GPU.

- **Teacher `name` must match your data**  
  With `reward_func_route_by_domain`, the client picks the teacher URL from `sample.metadata["domain"]`. That string must equal the `name` field of a teacher block in this YAML (e.g. `stem`, `tool`, `structured`).

Replace every `/path/to/teacher/hf-or-served-checkpoint` with real checkpoints. Different domains may use different paths or the same path on separate GPUs.

## 2. GPU accounting and training parallelism

- **`--rollout-num-gpus`** (passed in `run-opd-multidomain.sh` as `ROLLOUT_NUM_GPUS`) must equal the **sum of all `num_gpus`** across every `server_groups` block in the YAML (student + all teachers).

- **`--actor-num-gpus-per-node`** is separate: those GPUs run Megatron training and are **not** included in `ROLLOUT_NUM_GPUS`. Align **`--tensor-model-parallel-size`** (and related parallel flags) with `ACTOR_NUM_GPUS_PER_NODE`.

- Ensure the machine exposes enough GPUs for **rollout + actor** together (and leave headroom if Ray or the OS needs it). Example: `ROLLOUT_NUM_GPUS=5` and `ACTOR_NUM_GPUS_PER_NODE=2` uses seven GPUs for the job on an eight-GPU node.

## 3. OPD reward path

This example uses:

- `--custom-rm-path slime.rollout.on_policy_distillation.reward_func_route_by_domain`
- `--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards`

Together they route each sample to the teacher named by `metadata["domain"]`, extract teacher token log-probs for the response, and feed OPD’s KL-style signal during advantage computation. Scalar task rewards are zero unless you extend `post_process_rewards`.
