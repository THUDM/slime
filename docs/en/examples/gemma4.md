# Gemma4 Dense and MoE with GSM8K

This example is a small model-support validation for the Gemma4 text models. It
uses GSM8K because the purpose is to verify the Megatron model path, SGLang
rollout load path, loss masking, backward pass, and live weight update without
adding task-specific runtime variables.

Larger task-specific recipes should be layered on after this validation passes.

## What to Run

Run the dense and MoE variants separately on one 8-GPU node:

| Model | Script | Megatron topology | SGLang topology |
| --- | --- | --- | --- |
| `google/gemma-4-31B-it` | `scripts/run-gemma4-31B-gsm8k.sh` | TP2 PP4 CP1 | TP8 |
| `google/gemma-4-26B-A4B-it` | `scripts/run-gemma4-26B-A4B-gsm8k.sh` | TP2 PP2 EP2 CP1 | TP8 |

The scripts default to two rollouts with short responses. They are intended to
prove that the model can train, not to report a meaningful GSM8K score. A small
default `--entropy-coef` keeps the optimizer path active even when the tiny
sample receives zero reward.

Use a fresh converted checkpoint directory for each model and topology. The
default paths include TP/PP/EP/CP because Megatron distributed checkpoints are
sharded by the conversion topology.

## Prepare Checkpoints and Data

```bash
cd /root
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e . --no-deps

hf download google/gemma-4-31B-it --local-dir /root/gemma-4-31B-it
hf download google/gemma-4-26B-A4B-it --local-dir /root/gemma-4-26B-A4B-it
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k
```

Convert the dense checkpoint:

```bash
cd /root/slime
source scripts/models/gemma4-31B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint /root/gemma-4-31B-it \
   --tensor-model-parallel-size 2 \
   --pipeline-model-parallel-size 4 \
   --context-parallel-size 1 \
   --save /root/gemma-4-31B-it_tp2_pp4_cp1_torch_dist
```

Convert the MoE checkpoint:

```bash
cd /root/slime
source scripts/models/gemma4-26B-A4B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint /root/gemma-4-26B-A4B-it \
   --tensor-model-parallel-size 2 \
   --pipeline-model-parallel-size 2 \
   --expert-model-parallel-size 2 \
   --context-parallel-size 1 \
   --save /root/gemma-4-26B-A4B-it_tp2_pp2_ep2_cp1_torch_dist
```

## Run Training

```bash
cd /root/slime
bash scripts/run-gemma4-31B-gsm8k.sh
bash scripts/run-gemma4-26B-A4B-gsm8k.sh
```

To log the validation runs:

```bash
USE_WANDB=1 WANDB_PROJECT=slime-gemma4-gsm8k bash scripts/run-gemma4-31B-gsm8k.sh
USE_WANDB=1 WANDB_PROJECT=slime-gemma4-gsm8k bash scripts/run-gemma4-26B-A4B-gsm8k.sh
```

## Expected Signal

A successful run should show:

- SGLang loading `Gemma4ForConditionalGeneration`.
- At least one completed rollout and train step.
- `train/loss`, `train/grad_norm`, and entropy metrics in stdout or W&B.
- Successful raw `update_weights` from Megatron to SGLang.

For quality training, increase the rollout count, batch sizes, response length,
and evaluation interval, and set `ENTROPY_COEF=0`.
