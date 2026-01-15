# OSWorld VLM Training Cookbook

Train multi-turn computer-use agents on OSWorld using Slime's FSDP backend.

![OSWorld Architecture](architecture.jpeg)

## Overview

This cookbook demonstrates a stable, evergreen recipe for training multi-turn computer-use agents on OSWorld. The pipeline follows a practical VLM agent training approach:

```
SFT Alignment → On-Policy Distillation → PPO/GRPO Fine-Tuning → Evaluation
```

This sequence emphasizes format-correct tool calls, stable long-horizon behavior, and reward-aligned task success.

## Quick Start (4x H100, ~$50)

```bash
# Container setup
docker pull slimerl/slime:latest
docker run --gpus all --ipc=host --shm-size=16g \
  -v /ephemeral:/ephemeral -it slimerl/slime:latest /bin/bash

# Inside container
git clone https://github.com/THUDM/slime.git && cd /root/slime && pip install -e .

# Option A: Train SFT from scratch (downloads dataset automatically)
./examples/osworld/train_sft.sh

# Option B: Use pre-trained SFT checkpoint for GRPO
huggingface-cli download Jarrodbarnes/osworld-vlm-sft-step25 --local-dir /ephemeral/osworld-vlm-sft-step25-hf
huggingface-cli download Jarrodbarnes/osworld-train-v1 --repo-type dataset --local-dir /ephemeral/osworld_train

# Start OSWorld server on host (requires KVM)
# See "Environment Setup" section below

# Run GRPO training
export OSWORLD_SERVER_URL=http://172.17.0.1:8100
./examples/osworld/train_grpo.sh
```

## Environment Setup

OSWorld requires KVM for VM acceleration. The training container and OSWorld server run separately due to torch version conflicts (`desktop-env` pins torch 2.5.1, sglang requires 2.9+). The HTTP bridge API described below is Slime’s wrapper around OSWorld, not the native OSWorld server API.

```
Host (osworld_venv)              Container (slime_train)
────────────────────             ─────────────────────────
torch 2.5.1                      torch 2.9.1
desktop-env + KVM                sglang + PPO/GRPO
osworld_server.py :8100  <────>  HTTPRemoteDesktopEnv
```

### Host Setup

Stay in ~/OSWorld when using desktop-env to avoid VM re-downloads; run Qwen3-VL inference in the Slime container to avoid host dependency drift.

```bash
python3 -m venv ~/osworld_venv && source ~/osworld_venv/bin/activate
pip install desktop-env
git clone https://github.com/xlang-ai/OSWorld.git ~/OSWorld && cd ~/OSWorld
git clone https://github.com/THUDM/slime.git ~/slime
sudo -E ~/osworld_venv/bin/python quickstart.py --provider_name docker  # Downloads 11.4GB VM
sudo chown -R "$USER:$USER" ~/OSWorld  # Fix permissions after sudo
sudo rm -f /tmp/docker_port_allocation.lck  # Ensure port lock is writable

# Start server (run in tmux)
cd ~/OSWorld
sudo -E ~/osworld_venv/bin/python ~/slime/examples/osworld/tools/osworld_env_server.py --port 8100
```

### Parallel Rollouts

Scale with multiple servers:

```bash
# On host: start servers on different ports (run from ~/OSWorld)
for port in 8100 8101 8102 8103; do
  sudo -E ~/osworld_venv/bin/python ~/slime/examples/osworld/tools/osworld_env_server.py --port $port &
done

# In container: comma-separated URLs (match rollout_batch_size)
export OSWORLD_SERVER_URL="http://172.17.0.1:8100,http://172.17.0.1:8101,http://172.17.0.1:8102,http://172.17.0.1:8103,http://172.17.0.1:8104,http://172.17.0.1:8105,http://172.17.0.1:8106,http://172.17.0.1:8107"
```

## Training Pipeline

### 1. SFT Alignment (format + reasoning)

**Dataset**: [`Jarrodbarnes/osworld-reasoning-sft-v1`](https://huggingface.co/datasets/Jarrodbarnes/osworld-reasoning-sft-v1) (339 samples)
- 273 original ground-truth successful demonstrations
- 66 Claude Opus 4.5 reasoning trajectories with rich `<thinking>` blocks

**Pre-trained checkpoint**: `Jarrodbarnes/osworld-vlm-sft-step25`

Teaches **both** format-correct tool calls **and** reasoning patterns. The reasoning trajectories teach the model to observe, think, and act—not just produce correct syntax. Following Qwen3-VL SFT methodology, the vision encoder is frozen to preserve visual representations while the LLM learns to reason about GUI states.

```bash
# Run SFT training (4x H100, ~30 min)
./examples/osworld/train_sft.sh

# Or with custom settings
SLIME_SCRIPT_NUM_GPUS=8 SLIME_SCRIPT_OUTPUT_DIR=/path/to/output ./examples/osworld/train_sft.sh
```

**Note**: The SFT checkpoint alone achieves limited task success—it is a warmup that teaches format compliance and reasoning patterns. Task success comes from the data flywheel (Phase 2) and PPO/GRPO fine-tuning (Phase 3).

### 2. On-Policy Distillation (data flywheel)

Unlike offline SFT (which trains on fixed, pre-collected data), on-policy distillation trains on the **student's own successful rollouts**. This ensures:

- **Distribution matching**: Student learns from states it actually visits
- **No compounding errors**: No mismatch from teacher-only trajectories
- **Self-improvement**: Each iteration improves data quality

**The Data Flywheel**

The flywheel iterates: generate rollouts → filter by task success → SFT on filtered data → repeat.

| Iteration | Input | Process | Output |
|-----------|-------|---------|--------|
| Bootstrap (optional) | SFT checkpoint | Teacher-guided rollouts or curated seed set | Initial successful trajectories |
| 1 | SFT or bootstrap | Student rollouts, filter by `task_reward >= 0.5` | Filtered trajectories → SFT |
| 2+ | Iter N-1 output | Student rollouts, filter by success | Repeat until success rate plateaus |

**Bootstrap Stage (Optional)**: If the SFT checkpoint achieves 0 task success, bootstrap the flywheel with teacher-guided rollouts (e.g., Claude API demonstrations) or a small curated set of successful trajectories.

**Filtering Criterion**: Filter by **task reward** (binary 0/1), not shaped reward. Shaped rewards are for PPO/GRPO optimization, not trajectory filtering.

**Why Not Pre-Collected Teacher Data?** Pre-collected teacher trajectories (without the flywheel) are offline SFT—they suffer from distribution mismatch and compounding errors. The flywheel avoids these by always training on student-generated data (after optional bootstrap).

### 3. PPO/GRPO Fine-Tuning

After 3-5 flywheel iterations produce stable multi-turn behavior, PPO/GRPO optimizes for task success, efficiency, and shaped rewards. Partial signals (parse validity, execution, a11y grounding, screen change) provide dense feedback while task success remains the main objective.

### 4. Reward Shaping

```
shaped_reward = task_reward + 0.3 * partial_score - penalties
```

**Partial scores**: action parsing, execution, a11y grounding, efficiency, screen changes

**Penalties**: repetition, excessive waits, fallback parsing

Debugging tip:

```bash
OSWORLD_REWARD_DEBUG_LIMIT=10 bash examples/osworld/train_grpo.sh
```

## Methods

**SFT**: Grounds the model in OSWorld's action space. Desktop GUI automation requires precise coordinate clicks, keyboard input, and multi-step workflows. SFT teaches format compliance before RL exploration.

**On-policy distillation (data flywheel)**: The student generates rollouts in OSWorld, successful trajectories are filtered by task reward, and the student is trained (SFT) on its own successful behavior. An optional bootstrap with teacher demonstrations seeds the flywheel for weak starting checkpoints. This self-improvement loop (UI-TARS-2 style) avoids distribution mismatch. Typically 3-5 iterations until success rate plateaus.

**PPO/GRPO**: A reliable multi-turn RL baseline for GUI agents. Shaped rewards provide dense feedback while task success remains the primary objective.

**Terminology**: Documentation uses "PPO/GRPO" to describe the policy gradient phase. The `train_grpo.sh` script uses GSPO (Group-Sampled Policy Optimization) as the concrete implementation.

## Artifacts

### Checkpoints

- `Jarrodbarnes/osworld-vlm-sft-step25` - SFT warmup
- `Jarrodbarnes/osworld-vlm-gspo-step25` - After PPO/GRPO fine-tuning

### Datasets

**SFT Dataset**: [`Jarrodbarnes/osworld-reasoning-sft-v1`](https://huggingface.co/datasets/Jarrodbarnes/osworld-reasoning-sft-v1)
- 339 samples (273 original demos + 66 reasoning trajectories)
- Format: `<thinking>...</thinking>\nAction: ...\n<tool_call>...</tool_call>`
- Turn-aware masking applied to failed reasoning trajectories

**GRPO Artifacts** (from `Jarrodbarnes/osworld-train-v1`):
- `/ephemeral/osworld_train/osworld_tasks_train.parquet` (66 Ubuntu tasks with replay overlap + full task_config; 76 total including Windows)
- `/ephemeral/osworld_train/osworld_replay_train.jsonl` (expanded replay buffer, normalized system prompt)
- `/ephemeral/osworld_train/osworld_train_stats.json`

### Task Coverage

This cookbook uses a subset of OSWorld (66 Ubuntu tasks with replay overlap; Windows tasks are excluded to avoid environment mismatches).

### Reproducibility (Rebuild from Sources)

If you need to rebuild the union artifacts from raw datasets:

```bash
git clone https://github.com/xlang-ai/OSWorld.git /root/OSWorld
python examples/osworld/tools/build_union_datasets.py \
  --hf-root /ephemeral \
  --osworld-repo /root/OSWorld \
  --output-dir /ephemeral/osworld_train
```

This expects the HF datasets listed in `build_union_datasets.py` to be downloaded under `/ephemeral/osworld_datasets/`.

### Logs

- W&B: `jbarnes850-near-protocol/osworld-grpo`

## Code Structure

```
examples/osworld/
├── env.py              # OSWorld wrapper + HTTP client
├── rollout.py          # Multi-turn VLM rollout + replay injection
├── reward.py           # Shaped reward computation
├── replay_buffer.py    # Experience replay (optional, PPO/GRPO phase only)
├── train_sft.sh        # SFT training (Phase 1)
├── train_grpo.sh       # GRPO training (Phase 3)
├── grpo_config.yaml    # PPO/GRPO hyperparameters
└── tools/
    ├── osworld_env_server.py         # HTTP server (host)
    ├── build_union_datasets.py       # Build union task registry + replay buffer
    ├── curate_reasoning_sft.py       # Curate reasoning SFT dataset
    ├── process_reasoning_trajectories.py  # Process Claude trajectories
    ├── collect_reasoning_trajectories.py  # Collect reasoning trajectories
```

**Note on replay_buffer.py**: This is for experience replay during PPO/GRPO training, NOT for on-policy distillation. It injects successful trajectories when all online rollouts fail, preventing gradient vanishing in sparse reward settings. The data flywheel (distillation) is a separate, earlier phase.

## Evaluation

For offline evaluation, use OSWorld's native evaluation pipeline:

```bash
# See https://github.com/xlang-ai/OSWorld#evaluation
cd ~/OSWorld
python run.py --observation_type screenshot --model your_model --result_dir results/
```

The training script includes optional built-in evaluation. Set `DISABLE_EVAL=false` in `train_grpo.sh` to enable periodic evaluation during training.

For local eval with the lightweight script, use a higher turn cap to avoid truncating successful multi-step tasks:

```bash
python examples/osworld/tools/eval.py \
  --checkpoint /path/to/model \
  --tasks /path/to/tasks.parquet \
  --max-turns 12
```

## References

- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) - SOTA on OSWorld
- [UI-TARS-2](https://arxiv.org/abs/2509.02544) - On-policy distillation for GUI agents
- [OSWorld Benchmark](https://os-world.github.io/) - Desktop automation evaluation
