# Clear-Obstacles: RL on an interactive grid game

This example trains a model with GRPO to play the **clear-obstacles** grid game
from `real-time/environment/clear_obstacles.py`. It replaces the Python
code-interpreter tool environment with an interactive environment exposed through
four move tools.

The model sees a grid and must move `F` up to the GOAL row (row 0) while avoiding
obstacles (`#`). It acts by emitting `move_up` / `move_down` / `move_left` /
`move_right` tool calls. Reaching the GOAL gives reward `1.0`; crashing into an
obstacle (or running out of turns) gives `0.0`.

## How it works

1. **Seed dataset** (`obstacles_data_preprocess.py`). Each training example is a
   `(prompt, seed)` pair. The seed *is* the problem instance: it fully determines
   the obstacle layout and start position. Initializing the environment with that
   seed is all that's needed to materialize the grid.

2. **Interactive rollout** (`generate_with_obstacles.py`). At rollout time the
   seed arrives on `sample.label`. `generate` reconstructs the **exact** same grid
   with `ClearObstaclesToolEnv.reset(seed=...)` and keeps that single env instance
   live for the whole tool-calling loop, so every move acts on the real, evolving
   grid (true interactivity). The opening prompt contains the game rules (system),
   the four move tools (in the `<tools>` block), and the **first observation** —
   the rendered starting grid — as the user message. Each move's resulting grid is
   appended as an `<observation>...</observation>` block.

3. **Reward** (`generate_with_obstacles.reward_func`). The terminal reward from the
   environment (`1.0` win / `0.0` otherwise) is stashed on `sample.metadata` during
   the rollout and returned under the `score` key (selected via `--reward-key score`).

## Files

- `obstacles_data_preprocess.py`: build the seed dataset (train only).
- `generate_with_obstacles.py`: rollout generation + reward function.
- `obstacles_qwen3_4b_rl.sh`: GRPO training launcher.

The original Python-code-interpreter (`retool`) files (`generate_with_retool.py`,
`tool_sandbox.py`, `retool_*.sh`) remain in this directory for reference.

## Usage

Run all commands from the directory that contains `slime/` (the repo root, which
also contains `real-time/` and `Megatron-LM/`).

1. Install slime, then download and convert the base model
   (`Qwen/Qwen3-4B`):

```bash
pip install -e ./slime --no-deps
hf download Qwen/Qwen3-4B --local-dir $HOME/Qwen/Qwen3-4B

source ./slime/scripts/models/qwen3-4B.sh
PYTHONPATH=./Megatron-LM python ./slime/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint $HOME/Qwen/Qwen3-4B \
    --rotary-base 1000000 \
    --save $HOME/Qwen/Qwen3-4B_torch_dist
```

2. Build the seed dataset (the `environment` package must be importable, so put
   `real-time/` on `PYTHONPATH`):

```bash
PYTHONPATH=./real-time python3 ./slime/examples/realtime/obstacles_data_preprocess.py
# -> $HOME/obstacles-seeds/train.jsonl
```

3. Launch RL:

```bash
bash ./slime/examples/realtime/obstacles_qwen3_4b_rl.sh
```

The training script adds `real-time/` to the ray workers' `PYTHONPATH` so the
rollout can import `environment.clear_obstacles`.

## Tool format

The model is given four no-argument tools and calls them with the standard Qwen
tool-call format:

```
<tool_call>
{"name": "move_up", "arguments": {}}
</tool_call>
```
