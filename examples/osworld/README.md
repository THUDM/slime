# OSWorld VLM Training Cookbook

Train multi-turn computer-use agents on OSWorld using Slime's FSDP backend.

![OSWorld Architecture](architecture.jpeg)

## Overview

This cookbook demonstrates GSPO training with experience replay for sparse reward environments. The pipeline follows the standard VLM agent training approach:

```
Data Generation → SFT Warmup → GSPO with Replay → Evaluation
```

**Note on methodology**: On-policy distillation (UI-TARS-2, arXiv:2509.02544) achieves stronger results (47.5% on OSWorld) than off-policy approaches. This cookbook uses off-policy replay as a reproducible baseline; production deployments should consider on-policy methods with live annotator feedback.

## Quick Start (4x H100, ~$50)

```bash
# Container setup
docker pull slimerl/slime:latest
docker run --gpus all --ipc=host --shm-size=16g \
  -v /ephemeral:/ephemeral -it slimerl/slime:latest /bin/bash

# Inside container
git clone https://github.com/THUDM/slime.git && cd /root/slime && pip install -e .

# Download checkpoints and union datasets (paths must match train_grpo.sh defaults)
huggingface-cli download Jarrodbarnes/osworld-vlm-sft-step25 --local-dir /ephemeral/osworld-vlm-sft-step25-hf
huggingface-cli download Jarrodbarnes/osworld-train-v1 --repo-type dataset --local-dir /ephemeral/osworld_train

# Start OSWorld server on host (requires KVM)
# See "Environment Setup" section below

# Run training
export OSWORLD_SERVER_URL=http://172.17.0.1:8100
./examples/osworld/train_grpo.sh
```

## Environment Setup

OSWorld requires KVM for VM acceleration. The training container and OSWorld server run separately due to torch version conflicts (`desktop-env` pins torch 2.5.1, sglang requires 2.9+). The HTTP bridge API described below is Slime’s wrapper around OSWorld, not the native OSWorld server API.

```
Host (osworld_venv)              Container (slime_train)
────────────────────             ─────────────────────────
torch 2.5.1                      torch 2.9.1
desktop-env + KVM                sglang + GSPO
osworld_server.py :8100  <────>  HTTPRemoteDesktopEnv
```

### Host Setup

Stay in ~/OSWorld when using desktop-env to avoid VM re-downloads; run Qwen3-VL inference in the Slime container to avoid host dependency drift.

```bash
python3 -m venv ~/osworld_venv && source ~/osworld_venv/bin/activate
pip install desktop-env
git clone https://github.com/xlang-ai/OSWorld.git ~/OSWorld && cd ~/OSWorld
git clone https://github.com/THUDM/slime.git ~/slime
sudo -E python quickstart.py --provider_name docker  # Downloads 11.4GB VM
sudo chown -R "$USER:$USER" ~/OSWorld  # Fix permissions after sudo
sudo rm -f /tmp/docker_port_allocation.lck  # Ensure port lock is writable

# Start server (run in tmux)
source ~/osworld_venv/bin/activate
python ~/slime/examples/osworld/tools/osworld_env_server.py --port 8100
```

### Parallel Rollouts

Scale with multiple servers:

```bash
# On host: start servers on different ports
for port in 8100 8101 8102 8103; do
  python osworld_env_server.py --port $port &
done

# In container: comma-separated URLs
export OSWORLD_SERVER_URL="http://172.17.0.1:8100,http://172.17.0.1:8101,http://172.17.0.1:8102,http://172.17.0.1:8103"
```

## Training Pipeline

### 1. SFT Warmup

Pre-trained checkpoint: `Jarrodbarnes/osworld-vlm-sft-step25`

Teaches format-correct tool calls. Reduces malformed actions from 43% to 4%.

### 2. GSPO with Experience Replay

GSPO requires within-prompt variance for advantage computation. When all rollouts fail, advantages collapse. Experience replay injects successful trajectories when online samples all fail.

Key configuration in `train_grpo.sh`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--n-samples-per-prompt` | 4 | Within-prompt variance |
| `--rollout-temperature` | 0.8 → 0.5 | Exploration → stability (two-phase) |
| `OSWORLD_REPLAY_BUFFER` | osworld_replay_train.jsonl | Replay injection |
| `--rollout-function-path` | examples.osworld.rollout.generate_rollout | Custom batch rollout |

Recommended schedule:

```bash
# Phase 1: exploration
SLIME_SCRIPT_ROLLOUT_TEMPERATURE=0.8 bash examples/osworld/train_grpo.sh

# Phase 2: stability (reduce repeated actions)
SLIME_SCRIPT_ROLLOUT_TEMPERATURE=0.5 bash examples/osworld/train_grpo.sh
```

### 3. Reward Shaping

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

**SFT**: Grounds the model in OSWorld's action space. Desktop GUI automation requires precise coordinate clicks, keyboard input, and multi-step workflows. SFT teaches format compliance (reduces malformed actions from 43% to 4%) before RL exploration.

**GSPO with Replay**: Reinforcement learning for task completion. Experience replay addresses sparse rewards by injecting successful trajectories when online rollouts fail. Cross-app and browser tasks enable diverse action paths; single-step tasks benefit from replay variance.

**On-policy distillation** (recommended for production): UI-TARS-2 achieves 47.5% using live annotator feedback with 7B+ models. This cookbook provides the OSWorld integration; scale model size and consider on-policy methods for deployment.

## Artifacts

### Checkpoints

- `Jarrodbarnes/osworld-vlm-sft-step25` - SFT warmup
- `Jarrodbarnes/osworld-vlm-gspo-step25` - After GSPO

### Datasets

Primary artifacts (from `Jarrodbarnes/osworld-train-v1`):

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
├── replay_buffer.py    # Experience replay buffer
├── train_grpo.sh       # Training launcher
├── grpo_config.yaml    # GSPO hyperparameters
└── tools/
    ├── osworld_env_server.py  # HTTP server (host)
    ├── build_union_datasets.py  # Build union task registry + replay buffer
```

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
