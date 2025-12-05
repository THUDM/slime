# RLVE Integration for slime (Starter Pack)

Train with a curated starter pack of RLVE verifiable environments (math/logic with deterministic verification). Uses a pip shim for RLVE Gym, no core slime changes.

## Scope

This integration provides **infrastructure + 7 starter environments**. The architecture supports extending to RLVE's full 400+ environments.

| Included | How to extend |
|----------|---------------|
| Integration infrastructure (generate, reward, provider) | N/A - ready to use |
| 7 starter environments via `pip install rlve-gym` | Add to YAML config |
| Curriculum learning with difficulty controllers | Set `use_controllers: true` |

**Why a starter pack?** The `rlve-gym` shim packages only RLVE's `Gym/` directory, avoiding the vendored slime dependency that would conflict with your slime installation. The 7 included environments have fast, deterministic verifiers suitable for true-on-policy training.

**To use all 400+ environments:** Clone the full [RLVE repo](https://github.com/Zhiyuan-Zeng/RLVE), add to `PYTHONPATH`, and add entries to the YAML config. The integration architecture supports this without modification.

## Quick Start (slime Docker)

Inside `slimerl/slime` container (e.g., `slimerl/slime:v0.5.0rc0-cu126`):
```bash
# In container shell
pip install -e /root/slime
pip install rlve-gym  # shim that exposes Gym environments (no vendored slime)

export RLVE_CONFIG_PATH=/root/slime/examples/rlve/configs/starter_pack.yaml

# One-liner smoke (10 rollouts, instruct model recommended)
./examples/rlve/run_qwen3_8B_instruct.sh
```
Use the provided `run_qwen3_8B_instruct.sh` for full runs (2xH100); adjust paths/model as needed.

Note: the script generates a small dummy JSONL at runtime (`data/dummy_indices.jsonl`) to satisfy slime's global dataset API; all actual prompts are generated on-the-fly in `rlve_generate.py`.

## How It Works

```
slime rollout loop
      |
      v
rlve_generate.py    # Samples env, generates fresh prompt per rollout
      |
      v
rlve_reward.py      # Restores env state, calls verifier, returns accuracy
      |
      v
--reward-key accuracy   # Binary signal (0/1) for training
```

**Key insight**: The dummy JSONL only provides indices to satisfy slime's global dataset API; prompts are still generated on-the-fly by RLVE in the custom generate function, enabling curriculum learning without touching slime core.

## Files

| File | Purpose |
|------|---------|
| `rlve_prompt_provider.py` | Samples environments by weight, generates problems, tracks accuracy for curriculum |
| `rlve_generate.py` | Custom generate function; populates `sample.prompt` and `sample.metadata` |
| `rlve_reward.py` | Restores env from metadata, runs verifier, returns `{reward, accuracy, format_score}` |
| `configs/starter_pack.yaml` | Environment weights and difficulty parameters |

## Starter Pack Environments

| Environment | Weight | Key Parameter |
|-------------|--------|---------------|
| Multiplication | 1.0 | `digit_num` |
| Division | 1.0 | `divisor_digit_num`, `answer_digit_num` |
| Sorting | 1.0 | `N` (array length) |
| EuclidGame | 1.0 | `MAX_X_Y` |
| ShortestPath | 1.0 | `N`, `edge_density` |
| SpiralMatrix | 1.0 | `MAX_M_N` |
| LightUpPuzzle | 1.0 | `MAX_N_M`, `density` |

All have fast, deterministic verifiers suitable for true-on-policy training.

## Configuration

Edit `configs/starter_pack.yaml`:

```yaml
environments:
  Multiplication:
    weight: 1.0
    kwargs:
      digit_num: 3

# Format reward bonus (optional)
format_coef: 0.0  # Set > 0 to reward correct formatting

# Curriculum (Phase 2)
use_controllers: false  # Set true to enable adaptive difficulty
initial_difficulty: 0
difficulty_sliding_window_size: 1
min_metric_to_increase_difficulty: 0.5
min_prompts_before_difficulty_check: 8
```

### Adding Environments

```python
# Find available environments
from Gym.environments import identifier2environment
print(list(identifier2environment.keys()))
```

Add to YAML with weight and kwargs matching the environment's generator signature.

## Answer Format

Model must wrap answers with markers:

```
<answer>42</answer>
```

The prompt template instructs this. Verifier extracts and validates.

## Model Setup

**Use an instruct model** (e.g., Qwen3-8B-Instruct) to ensure well-formed responses with `<answer>...</answer>` markers. Base models may not follow the format reliably.

```bash
# Download and convert Qwen3-8B-Instruct
huggingface-cli download Qwen/Qwen3-8B-Instruct --local-dir /root/Qwen3-8B-Instruct

source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B-Instruct \
    --save /root/Qwen3-8B-Instruct_torch_dist
```

## Checkpoint/Resume

The provider supports state persistence for long training runs:

```python
from examples.rlve import get_provider

provider = get_provider()

# Save state (difficulty levels, accuracy counters, seed)
provider.save_state("/path/to/rlve_state.json")

# Load state to resume
provider.load_state("/path/to/rlve_state.json")
```

State includes `environment2difficulty`, accuracy counters, and `problem_generation_seed` - matching RLVE's tinker integration.

## Troubleshooting

**Gym not found**: `pip install rlve-gym` (shim). For full RLVE, clone and `PYTHONPATH=/path/to/RLVE:$PYTHONPATH`.

**Low success rate**: Reduce `kwargs` difficulty, use instruct model, ensure `<answer>` tags are present.

**Environment not found**: Check `identifier2environment.keys()` for valid IDs.
