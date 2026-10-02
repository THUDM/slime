# FlashREINFORCE

FlashREINFORCE ([NVIDIA molt](https://github.com/NVIDIA-NeMo/labs-molt), [labs-molt#116](https://github.com/NVIDIA-NeMo/labs-molt/pull/116)) is critic-free RL with one rollout per prompt, built for asynchronous rollout. This directory reproduces molt's quick-start recipe in slime: DeepSeek-R1-Distill-Qwen-1.5B on the FP16 sanity test `sail/Sanity-Test-R1D-1.5B`, evaluated on AIME 2024 and 2025.

## The algorithm in slime

FlashREINFORCE is not a new loss. It combines four pieces, each of which is a slime option:

| Piece | slime option |
|---|---|
| Advantage: reward minus the mean reward of the whole rollout batch, with no prompt groups, no std normalization and no whitening | `--advantage-estimator flash_reinforce` |
| REINFORCE gradient: one optimizer step per rollout, with the training forward reused as the old log-probs (PPO ratio exactly 1) | `--num-steps-per-rollout 1`; the objective defaults to `--pg-loss-type reinforce` |
| Importance sampling against the rollout engine: every token carries its unclipped ratio `pi_train / pi_rollout`, and a sequence whose mean sampled-token binary KL exceeds δ is dropped | `--use-tis --custom-tis-function-path slime.backends.megatron_utils.loss.binary_kl_trust_region_function --tis-binary-kl-threshold 5e-3` |
| Loss: token mean within each sequence, then mean over the sequences of the step | the default (do not set `--calculate-per-token-loss`) |

The rollout side runs [fully-async](../fully_async/README.md) and forms batches the way molt's asynchronous trainer does (`--fully-async-pool-size 512 --fully-async-max-queued-batches 8 --fully-async-drain-each-epoch`):

* A pool of 512 rollouts, counting the ones still generating and the finished ones not yet in a batch.
* A batch is formed only while one of 8 slots is free, so at most 8 batches are formed ahead of the trainer. The trainer frees a slot when it takes a batch.
* While forming a batch, the worker takes finished rollouts one at a time and refills the pool with one prompt for each. When several have finished, it takes them in a random order fixed at dispatch, as molt's `ray.wait` does.
* Each pass over the 1,460 prompts stops refilling once every prompt is out, drains the pool, and ends with a 52-rollout batch. That batch is trained as one update, so 1,000 passes give 12,000 updates.
* A rollout that straddles a weight update continues from its partial response under the new weights. Its tokens then come from several policy versions; the per-token importance weights and the trust region correct for this.

Weight decay applies to every parameter (`--apply-wd-to-all-params`), as with molt's AdamW.

The FlashREINFORCE section of the usage guide (`docs/en/get_started/usage.md`) describes each option.

## Files

* `run-deepseek-r1-distill-qwen-1.5B.sh`: the recipe on one 8-GPU node (1 trainer GPU and 7 SGLang engines).
* `reward.py`: `math_reward`, molt's grader applied the way molt's math agent applies it.
* `math_grader.py`: molt's math grader (Apache-2.0), vendored with lint-only changes. The reward is 1 or 0: it takes the last `\boxed{}` answer (falling back to an "answer is X" phrase or the last number) and compares it by normalized string or sympy equality.

## Prerequisites

```bash
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /root/models/DeepSeek-R1-Distill-Qwen-1.5B
hf download sail/Sanity-Test-R1D-1.5B --repo-type dataset --local-dir /root/datasets/Sanity-Test-R1D-1.5B
```

The dataset is read as-is: `train/math_1460.parquet` for training, and `test/aime_2024.parquet` and `test/aime_2025.parquet` for evaluation. Each prompt is a chat message list that already asks for a `\boxed{}` answer, and `reward_model` holds the ground truth. The HF checkpoint is loaded directly, so no conversion to `torch_dist` is needed.

## Run

```bash
cd slime
bash examples/flash_reinforce/run-deepseek-r1-distill-qwen-1.5B.sh
```

The defaults are molt's full recipe: 12,000 updates of 128 rollouts (52 for the last update of each pass), learning rate 1e-6 with 360 warmup updates and cosine decay to 1e-7, weight decay 0.1, Adam betas (0.9, 0.95), temperature 1.0, 8k responses, no KL. `MODEL_DIR`, `DATA_DIR`, `SAVE_DIR`, `SAVE_INTERVAL`, `KEEP_CHECKPOINTS`, `NUM_ROLLOUT`, `POOL_SIZE`, `QUEUED_BATCHES` and `WANDB_PROJECT` (with `WANDB_MODE` and `WANDB_DIR`) override paths and scale. The learning-rate schedule always spans 12,000 updates and advances one step per update, so a shorter first run can be resumed from its checkpoint with the same schedule.

To reproduce molt's ablation without the trust region (importance-weighted policy gradient, which the paper reports to drift in BF16), run with `TRUST_REGION_DELTA=inf`.

## What to watch

* `train/ppo_kl` stays exactly 0: the old log-probs are the detached training forward.
* `train/tis_seq_reject_frac` (fraction of sequences outside the trust region) and `train/tis_binary_kl` stay small and stable.
* `rollout/staleness/mean`, `rollout/staleness/max` and `rollout/staleness/multi_version_frac` show how off-policy the batches are; the last one is the fraction of samples generated by more than one weight version. A rollout spends about 512 / 128 = 4 updates in the pool; when training is slower than generation, the 8 slots stay full and add about 8 more.
* `rollout/fully_async/queued_batches` (batches formed ahead when the trainer takes one), `rollout/fully_async/pool_finished` (finished rollouts waiting in the pool) and `rollout/fully_async/epoch_tail` (1 for the last batch of a pass).
* `eval/aime24` and `eval/aime25` are avg@32 (pass@1 averaged over 32 samples), molt's `eval_aime_2024_pass1` and `eval_aime_2025_pass1`; the paper reports their mean. slime logs the evaluation after update N as `eval N-1` (and the one before training as `eval 0`), molt as step N.

## Differences from molt

* Evaluation: slime stops training to evaluate the weights after update N. molt evaluates while the trainer keeps consuming queued batches, so its evaluation at N also sees the next few updates.
* Weight updates: slime aborts the requests still generating and resends each prompt with its partial response; molt pauses vLLM in keep mode. In both, the prefix is prefilled again under the new weights and the earlier tokens keep their original log-probs.
* Collection order: when several rollouts have finished, molt takes the one `ray.wait` happens to return first, which is an arbitrary order; slime uses a random order fixed at dispatch.
* Prompt length: molt left-truncates prompts to 1,024 tokens (its 9,216-token context minus 8,192 new tokens). One training prompt (1,065 tokens) is affected; slime keeps it whole.
* Engines and numerics: SGLang and Megatron here, vLLM and FSDP2 in molt, both in bf16, so the rollout-versus-training mismatch that the trust region measures differs somewhat. The prompt order and sampling seeds differ too.
* slime keeps every checkpoint unless `KEEP_CHECKPOINTS` is set; molt keeps the last three.
