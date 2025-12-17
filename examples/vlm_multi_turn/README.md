# VLM Multi-Turn (FSDP backend, sokoban dataset)

Multi-turn VLM training on Sokoban with custom rollout, pluggable interactive env, and customized reward function.

## How to run the training script
1) Set basic env (override as needed):
```bash
export WANDB_API_KEY=...                                  
export SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-2B-Instruct       
export SLIME_SCRIPT_NUM_GPUS=4                            
export SLIME_SCRIPT_SOKOBAN_DATASET_ID=VeraIsHere/sokoban_processed
export SLIME_SCRIPT_SOKOBAN_DATA_ROOT=/root/datasets/sokoban_processed
export SLIME_SCRIPT_SOKOBAN_TRAIN_PATH=$SLIME_SCRIPT_SOKOBAN_DATA_ROOT/train.parquet
```

2) Run the script (downloads model + dataset, then launches training):
```bash
python examples/vlm_multi_turn/run_sokoban_vlm_multi_turn.py
```

What it does:
- Downloads the chosen Qwen-VL checkpoint into `/root/models/<MODEL_NAME>` if absent.
- Downloads the preprocessed Sokoban dataset from `${SLIME_SCRIPT_SOKOBAN_DATASET_ID}` into `${SLIME_SCRIPT_SOKOBAN_DATA_ROOT}`.
- Launches FSDP training with custom rollout/reward hooks:
  - `--custom-generate-function-path examples.vlm_multi_turn.rollout.generate`
  - `--custom-rm-path examples.vlm_multi_turn.reward_sokoban.async_compute_reward`
  - `--rollout-interaction-env-path examples.vlm_multi_turn.env_sokoban`
  - Multi-turn caps: `--max-turns`, token budget stop-on-max enabled.


## What each file does
- `examples/vlm_multi_turn/run_sokoban_vlm_multi_turn.py`: downloads model + preprocessed dataset, and builds the full training CLI args.
- `examples/vlm_multi_turn/data_preprocess/preprocess_sokoban.py`: optional tool to regenerate the processed dataset (VeraIsHere/sokoban_processed) from raw dataset (Xiaofeng77/sokoban) if needed.
- `examples/vlm_multi_turn/rollout.py`: custom multi-turn rollout (with pluggable interactive env) that streams tokens,builds aligned loss masks/log_probs, enforces max_turns, early-stops on max_new_tokens.
- `examples/vlm_multi_turn/env_sokoban.py`: lightweight Sokoban environment exposing reset/step/format hooks; renders grids to images, tracks metrics, and returns observations per turn.
- [Temparory]`examples/vlm_multi_turn/reward_sokoban.py`:  reward helper combining avg step rewards, final bonus, and distance shaping; used via `--custom-rm-path`.
